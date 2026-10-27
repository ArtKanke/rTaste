"""The tools a proposer can use: search arXiv, read a paper, and browse the seed paper's code.

Every model gets the same tools. arXiv is the only literature source, and nothing posted after
the launch day can be found or read. The code of the seed paper can be read but not run.
"""

import fcntl
import hashlib
import html
import io
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pypdfium2 as pdfium
from pyarrow import parquet
from pypdf import PdfReader

from rtaste.models import ROOT

CACHE = ROOT / "runs" / "cache"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
READ_CHARS = 24_000  # how much of a paper or file one call returns (about 6,000 tokens)
PAGE_WIDTH = 980  # pixels; the width at which a seed paper's pages are shown to every model
TABLE_ROWS = 5  # rows of a data table shown by one call

_arxiv_lock = threading.Lock()


def _fetch(url: str) -> bytes:
    """Download a URL once and keep it on disk.

    arXiv asks for at most one request every 3 seconds. The time of the last request is kept in a
    locked file, so the pause also holds between separate runs that share the cache. After a
    rate-limit answer everyone waits a minute, and after a refusal (403) ten minutes.
    """
    cached = CACHE / hashlib.sha256(url.encode()).hexdigest()
    if cached.exists():
        return cached.read_bytes()
    CACHE.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        with _arxiv_lock, (CACHE / "arxiv.lock").open("a+") as turn:
            fcntl.flock(turn, fcntl.LOCK_EX)  # released when the file is closed
            if cached.exists():  # someone else fetched it while we waited
                return cached.read_bytes()
            turn.seek(0)
            time.sleep(max(0.0, min(600.0, float(turn.read() or 0) + 3.2 - time.time())))
            next_turn = time.time()
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "rtaste/0.1 (research harness)"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = response.read()
                cached.write_bytes(data)
                return data
            except urllib.error.HTTPError as error:
                if error.code in (403, 429):  # said in the log, because a search that fails makes a judgment worse
                    print(f"  arXiv refused a request (HTTP {error.code})", flush=True)
                if error.code == 403:  # arXiv treats quick requests after a refusal as an attack: ten minutes' rest
                    next_turn = time.time() + 600
                    raise
                if error.code not in (429, 500, 502, 503) or attempt == 4:
                    raise
                next_turn = time.time() + 60
            except (urllib.error.URLError, TimeoutError):
                if attempt == 4:
                    raise
            finally:
                turn.truncate(0)
                turn.write(str(next_turn))
        time.sleep(15 * (attempt + 1))  # outside the lock, so other requests are not held up


def _entries(params: dict) -> list[dict]:
    """Ask the arXiv API and return each paper's id, date, title, authors and abstract."""
    feed = ET.fromstring(_fetch("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)))
    papers = []
    for entry in feed.findall("a:entry", ATOM):
        if entry.find("a:id", ATOM) is None or entry.find("a:published", ATOM) is None:
            continue
        papers.append({
            "id": re.sub(r"v\d+$", "", entry.find("a:id", ATOM).text.split("/abs/")[-1]),
            "posted": entry.find("a:published", ATOM).text[:10],  # date of the first version
            "updated": entry.find("a:updated", ATOM).text[:10],  # date of the latest version
            "title": " ".join(entry.find("a:title", ATOM).text.split()),
            "authors": [a.find("a:name", ATOM).text for a in entry.findall("a:author", ATOM)],
            "abstract": " ".join(entry.find("a:summary", ATOM).text.split()),
        })
    return papers


def paper_info(arxiv_id: str) -> dict | None:
    """A paper's id, date, title, authors and abstract, or None if arXiv has no such paper."""
    found = _entries({"id_list": re.sub(r"v\d+$", "", arxiv_id.strip().removeprefix("arXiv:"))})
    return found[0] if found else None


def arxiv_search(query: str, launch_day: str, max_results: int = 10) -> str:
    """Search arXiv and list the best matches posted on or before the launch day."""
    if not re.search(r"\b(all|ti|abs|au|cat):", query):
        query = "all:" + query
    until = launch_day.replace("-", "") + "2359"
    papers = _entries({"search_query": f"({query}) AND submittedDate:[199101010000 TO {until}]",
                       "max_results": max_results, "sortBy": "relevance"})
    if not papers:
        return "No papers found."
    return "\n\n".join(
        f"[{p['id']}] {p['title']} ({p['posted']})\n{', '.join(p['authors'][:4])}"
        f"{' et al.' if len(p['authors']) > 4 else ''}\n{p['abstract']}" for p in papers)


def paper_text(arxiv_id: str) -> str:
    """The full text of a paper: arXiv's HTML version when there is one, otherwise the PDF."""
    try:
        page = _fetch(f"https://arxiv.org/html/{arxiv_id}").decode("utf-8", "replace")
        page = re.sub(r"<(script|style|nav|header|footer)\b.*?</\1>", " ", page, flags=re.S)
        page = re.sub(r'<math\b[^>]*?alttext="([^"]*)"[^>]*>.*?</math>', r" \1 ", page, flags=re.S)
        page = re.sub(r"</(p|div|h\d|li|tr|section|figcaption)>", "\n", page)
        text = html.unescape(re.sub(r"<[^>]+>", " ", page))
    except urllib.error.HTTPError as error:
        if error.code in (429, 500, 502, 503):  # arXiv is busy: fail now rather than show a different text later
            raise
        pdf = PdfReader(io.BytesIO(_fetch(f"https://arxiv.org/pdf/{arxiv_id}")))
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


_pages: dict[str, list[bytes]] = {}
_pages_lock = threading.Lock()


def paper_pages(arxiv_id: str) -> list[bytes]:
    """Every page of the paper's PDF as a JPEG image of the same width, so that its figures can be seen.
    We render the pages ourselves, so every model is shown exactly the same images."""
    with _pages_lock:  # the PDF library must not be used by two threads at once; each paper is rendered once
        if arxiv_id not in _pages:
            pdf = pdfium.PdfDocument(_fetch(f"https://arxiv.org/pdf/{arxiv_id}"))
            pages = []
            for page in pdf:
                image = io.BytesIO()
                page.render(scale=PAGE_WIDTH / page.get_width()).to_pil().convert("RGB").save(image, "JPEG", quality=80)
                pages.append(image.getvalue())
                page.close()
            pdf.close()
            _pages[arxiv_id] = pages
        return _pages[arxiv_id]


def seed_inputs(arxiv_id: str) -> dict:
    """Hashes of what every model is shown of a seed paper: its text, and the PDF its pages are rendered from."""
    return {"paper_sha256": hashlib.sha256(paper_text(arxiv_id).encode()).hexdigest(),
            "pdf_sha256": hashlib.sha256(_fetch(f"https://arxiv.org/pdf/{arxiv_id}")).hexdigest(),
            "pages": len(paper_pages(arxiv_id))}


def arxiv_read(arxiv_id: str, launch_day: str, start: int = 0) -> str:
    """Read part of a paper's full text, if it was posted on or before the launch day."""
    paper = paper_info(arxiv_id)
    if paper is None:
        return f"No arXiv paper with id {arxiv_id}."
    if paper["posted"] > launch_day:
        return f"Paper {arxiv_id} was posted after the launch day and cannot be read."
    # If the paper was revised after the launch day, only its first version may be read.
    version = "v1" if paper["updated"] > launch_day else ""
    return _page(paper_text(paper["id"] + version), start, f"{paper['title']} ({paper['posted']})")


def _page(text: str, start: int, title: str) -> str:
    part = text[start:start + READ_CHARS]
    more = f" Call again with start={start + READ_CHARS} for more." if start + READ_CHARS < len(text) else ""
    return f"{title}\nCharacters {start} to {start + len(part)} of {len(text)}.{more}\n\n{part}"


def _inside(repo: Path, path: str) -> Path | None:
    """Resolve a path inside the repository, refusing anything that points outside it."""
    target = (repo / path.lstrip("/")).resolve()
    return target if target == repo.resolve() or repo.resolve() in target.parents else None


def repo_ls(repo: Path, path: str = "") -> str:
    """List a directory of the seed paper's code repository."""
    folder = _inside(repo, path)
    if folder is None or not folder.is_dir():
        return f"No such directory: {path}"
    lines = []
    for item in sorted(folder.iterdir()):
        if item.name == ".git":
            continue
        lines.append(f"{item.name}/" if item.is_dir() else f"{item.name}  ({item.stat().st_size} bytes)")
    return "\n".join(lines) or "(empty)"


def repo_read(repo: Path, path: str, start: int = 0) -> str:
    """Read part of a file from the seed paper's code repository: text, or a few rows of a parquet table."""
    file = _inside(repo, path)
    if file is None or not file.is_file():
        return f"No such file: {path}"
    if file.suffix == ".parquet":
        return _table_rows(file, path, start)
    data = file.read_bytes()
    if b"\0" in data[:4000]:
        return f"{path} is a binary file ({len(data)} bytes)."
    return _page(data.decode("utf-8", "replace"), start, path)


def _table_rows(file: Path, path: str, start: int) -> str:
    """The columns of a parquet table and a few of its rows, so that the data can be looked at."""
    table = parquet.ParquetFile(file)
    rows, seen = [], 0
    for batch in table.iter_batches(batch_size=64):
        if seen + batch.num_rows > start and len(rows) < TABLE_ROWS:
            rows += batch.slice(max(0, start - seen), TABLE_ROWS - len(rows)).to_pylist()
        seen += batch.num_rows
        if len(rows) >= TABLE_ROWS:
            break
    columns = ", ".join(f"{column.name} ({column.type})" for column in table.schema_arrow)
    shown = [json.dumps({key: _short(value) for key, value in row.items()}, default=str, ensure_ascii=False)
             for row in rows]
    total = table.metadata.num_rows
    more = f" Call again with start={start + TABLE_ROWS} for more rows." if start + TABLE_ROWS < total else ""
    return (f"{path}: a table with {total} rows.\nColumns: {columns}\n"
            f"Rows {start} to {start + len(rows)}.{more}\n\n" + "\n".join(shown))[:READ_CHARS]


def _short(value, limit: int = 1500):
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return value if len(text) <= limit else text[:limit] + f" [... {len(text) - limit} more characters]"


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def arxiv_tools(launch_day: str) -> tuple[list[dict], dict]:
    """The literature tools: their descriptions for a model, and the functions that carry them out."""
    text, number = {"type": "string"}, {"type": "integer"}
    specs = [
        {"name": "arxiv_search",
         "description": "Search arXiv. Returns up to 10 papers with id, date, authors and abstract. Plain words "
                        "are OR-ed and ranked by relevance; put a phrase in double quotes and use AND to require "
                        "terms. Fields: ti: (title), abs: (abstract), au: (author), cat: (category).",
         "parameters": _schema({"query": text}, ["query"])},
        {"name": "arxiv_read",
         "description": f"Read the full text of an arXiv paper by id, {READ_CHARS} characters at a time.",
         "parameters": _schema({"arxiv_id": text, "start": number}, ["arxiv_id"])},
    ]
    run = {
        "arxiv_search": lambda a: arxiv_search(a["query"], launch_day),
        "arxiv_read": lambda a: arxiv_read(a["arxiv_id"], launch_day, a.get("start", 0)),
    }
    return specs, run


def proposer_tools(repo: Path, launch_day: str) -> tuple[list[dict], dict]:
    """A proposer's tools: the literature tools, and read access to the seed paper's code."""
    text, number = {"type": "string"}, {"type": "integer"}
    specs, run = arxiv_tools(launch_day)
    specs += [
        {"name": "repo_ls",
         "description": "List a directory of the seed paper's code repository. Use an empty path for the top level.",
         "parameters": _schema({"path": text}, ["path"])},
        {"name": "repo_read",
         "description": f"Read a file from the seed paper's code repository: a text file, {READ_CHARS} characters at "
                        f"a time, or a parquet table, {TABLE_ROWS} rows at a time with its columns (start is then a "
                        "row number).",
         "parameters": _schema({"path": text, "start": number}, ["path"])},
    ]
    run |= {
        "repo_ls": lambda a: repo_ls(repo, a.get("path", "")),
        "repo_read": lambda a: repo_read(repo, a["path"], a.get("start", 0)),
    }
    return specs, run
