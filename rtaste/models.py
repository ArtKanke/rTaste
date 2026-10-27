"""Talk to the models and keep track of what each call costs.

A `Chat` is one append-only conversation with one model. It hides the two providers' request
formats behind the same three things: send a message, get back text and tool calls, hand back
tool results. Every call is priced from the usage the provider reports and written to a spend log.
"""

import base64
import hashlib
import json
import os
import re
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

import anthropic
import httpx2
import openai
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parent.parent
SPEND_LOG = ROOT / "runs" / "spend.jsonl"
WRAP_UP = 0.8  # share of a budget after which no new tool calls are run
REPAIR_ROUNDS = 2  # how often an invalid answer goes back to the model


@dataclass(frozen=True)
class Model:
    provider: str  # the request format it speaks: "openai" (Responses API) or "anthropic" (Messages API)
    api_id: str
    # Dollars per million tokens: fresh input, cached input, cache write, output.
    # Anthropic's cache-write price is the one for a one-hour cache, which is what we ask for.
    price_in: float
    price_cached: float
    price_write: float
    price_out: float
    # For another company's model that speaks the OpenAI format: where it is served, and the name of
    # the environment variable that holds its key.
    base_url: str | None = None
    key: str | None = None
    # Some models bill the whole request at higher rates once the input passes a number of tokens:
    # (tokens, factor on input, factor on output).
    long_input: tuple[int, float, float] | None = None
    top_effort: str = "max"  # the highest reasoning effort the model accepts


OPENAI_LONG = (272_000, 2.0, 1.5)
MODELS = {
    "astra": Model("openai", "gpt-6-astra", 10.0, 1.0, 12.5, 50.0, long_input=OPENAI_LONG),
    "sol": Model("openai", "gpt-6.1-sol", 2.0, 0.10, 2.5, 10.0, long_input=OPENAI_LONG),
    "luna": Model("openai", "gpt-6-luna", 0.10, 0.01, 0.125, 0.50, long_input=OPENAI_LONG),
    "fable": Model("anthropic", "claude-fable-5-1", 10.0, 0.25, 20.0, 50.0),
    "opus": Model("anthropic", "claude-opus-5-5", 4.0, 0.20, 8.0, 20.0),
    "sonnet": Model("anthropic", "claude-sonnet-5-5", 2.0, 0.20, 4.0, 10.0),
    "muse": Model("openai", "muse-spark-1.3", 1.25, 0.15, 1.25, 4.25,
                  base_url="https://api.meta.ai/v1", key="META_API_KEY"),
    "grok": Model("openai", "grok-4.7", 2.0, 0.50, 2.0, 6.0, base_url="https://api.x.ai/v1", key="XAI_API_KEY",
                  long_input=(200_000, 2.0, 2.0), top_effort="xhigh"),
}


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    text: str
    calls: list[ToolCall]
    status: str  # "done", "tool_use", "truncated" or "refused"
    cost: float
    detail: str = ""  # why the model refused or was cut off, when the provider says


class OverBudget(Exception):
    """The conversation's budget is spent; no further request is sent."""


@dataclass
class Budget:
    """A spending limit that several conversations can share (for example a whole team)."""

    cap: float
    spent: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, cost: float) -> None:
        with self._lock:
            self.spent += cost

    @property
    def left(self) -> float:
        return self.cap - self.spent


def load_env() -> None:
    """Read keys from the repo's .env file into the environment, without overriding what is set."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_clients: dict[tuple, object] = {}
_log_lock = threading.Lock()


# Ask the system to probe a quiet connection once a minute. Without this a reply whose connection was
# lost on the way is waited for until the time limit. The Anthropic SDK sets the same options itself.
KEEP_ALIVE = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1), (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 60),
              (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5),
              (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPIDLE", 0x10), 60)]  # 0x10 is its number on macOS


def _client(model: Model):
    where = (model.provider, model.base_url)
    if where not in _clients:
        load_env()
        # We retry ourselves, so the SDKs' own retries are switched off.
        if model.provider == "anthropic":
            _clients[where] = anthropic.Anthropic(max_retries=0, timeout=1800)
        else:
            _clients[where] = openai.OpenAI(
                base_url=model.base_url, api_key=os.environ[model.key or "OPENAI_API_KEY"], max_retries=0, timeout=1800,
                http_client=openai.DefaultHttpxClient(transport=httpx2.HTTPTransport(socket_options=KEEP_ALIVE)))
    return _clients[where]


def transient(error: Exception) -> bool:
    """A rate limit, a server error or a lost connection: worth repeating later, and not the model's doing."""
    if isinstance(error, (openai.APIStatusError, anthropic.APIStatusError)):
        # Status 200 is an error that arrived in the middle of a streamed reply.
        return error.status_code in (200, 429) or error.status_code >= 500
    return isinstance(error, (openai.APIConnectionError, anthropic.APIConnectionError, httpx2.TransportError))


def _with_retries(request, attempts: int = 20):
    """Run a request, repeating it after rate limits, server errors and lost connections.

    The pauses grow from 5 seconds to a minute, about 16 minutes in all, which outlasts most
    incidents. A request that ran into the time limit is repeated only once. A failed request
    leaves the conversation untouched, so repeating it is safe.
    """
    timeouts = 0
    for attempt in range(attempts):
        try:
            return request()
        except Exception as error:
            timeouts += isinstance(error, (openai.APITimeoutError, anthropic.APITimeoutError, httpx2.TimeoutException))
            if not transient(error) or attempt == attempts - 1 or timeouts > 1:
                raise
            response = getattr(error, "response", None)
            wait = response.headers.get("retry-after") if response is not None else None
            pause = float(wait) if wait and wait.replace(".", "").isdigit() else min(60, 5 * 2**attempt)
            print(f"  waiting {pause:.0f} s after {type(error).__name__} (try {attempt + 1})", flush=True)
        time.sleep(pause)


def _base64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode()


def now() -> str:
    """The time in UTC, as written into every saved record."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log_spend(entry: dict) -> None:
    """Add one line to the spend log. Every paid call and every machine minute is recorded here."""
    with _log_lock:
        SPEND_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SPEND_LOG.open("a") as log:
            log.write(json.dumps(entry) + "\n")


class Chat:
    """One append-only conversation with one model.

    The system prompt and the tool list stay fixed for the whole conversation and earlier turns
    are never edited. Both providers need that to accept their own reasoning back, and it is
    also what makes prompt caching work.
    """

    def __init__(self, model: str, system: str, tools: list[dict] = (), effort: str = "high",
                 purpose: str = "", budget: Budget | None = None, max_tokens: int = 128_000, shared: str = ""):
        self.name = model
        self.model = MODELS[model]
        self.system = system
        self.tools = list(tools)  # each: {"name", "description", "parameters"}
        self.effort = effort
        self.purpose = purpose
        self.budget = budget
        self.max_tokens = max_tokens
        self.history: list = []  # provider-shaped; sent back as is
        self.transcript: list[dict] = []  # readable record, saved with the results
        # `shared` is an opening that many conversations begin with, word for word, for example a whole
        # paper. They then use one cache key, so that a provider can serve the opening from its cache.
        self.shared = shared
        self.cache_key = hashlib.sha256(f"{model}\n{shared}".encode()).hexdigest()[:32] if shared else uuid.uuid4().hex
        self.cost = 0.0
        self.turns = 0

    def send(self, text: str, images: list[bytes] = ()) -> Reply:
        """Send a user message and return the model's reply. JPEG images, if any, go before the text."""
        content = text
        if images and self.model.provider == "openai":
            content = [{"type": "input_image", "image_url": "data:image/jpeg;base64," + _base64(image)}
                       for image in images] + [{"type": "input_text", "text": text}]
        elif images:
            content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": _base64(image)}} for image in images]
            content.append({"type": "text", "text": text})
        elif self.shared and not self.history and not self.model.base_url:
            # Anthropic and OpenAI read a prompt from their cache only up to a marked place, so the shared
            # opening is a block of its own with the mark at its end. Other providers match the opening
            # of a prompt by themselves.
            assert text.startswith(self.shared), "the first message must begin with the shared opening"
            kind, mark = (("input_text", {"prompt_cache_breakpoint": {"mode": "explicit"}})
                          if self.model.provider == "openai" else
                          ("text", {"cache_control": {"type": "ephemeral", "ttl": "1h"}}))
            content = [{"type": kind, "text": self.shared, **mark}, {"type": kind, "text": text[len(self.shared):]}]
        self.history.append({"role": "user", "content": content})
        self.transcript.append({"role": "user", "text": text, "images": len(images)})
        return self._request()

    def send_results(self, results: list[tuple[ToolCall, str]]) -> Reply:
        """Return the outputs of the tools the model asked for, and get its next reply."""
        if self.model.provider == "openai":
            for call, output in results:
                self.history.append({"type": "function_call_output", "call_id": call.id, "output": output})
        else:
            self.history.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": call.id, "content": output} for call, output in results]})
        for call, output in results:
            self.transcript.append({"role": "tool", "name": call.name, "output": output[:4000]})
        return self._request()

    def retry(self, effort: str | None = None) -> Reply:
        """Ask again after a truncated reply. The truncated reply was never added to the history."""
        return self._request(effort=effort)

    def _request(self, effort: str | None = None) -> Reply:
        if self.budget is not None and self.budget.left <= 0:
            raise OverBudget(f"{self.purpose}: budget of ${self.budget.cap:.2f} is spent")
        effort = effort or self.effort
        if self.model.provider == "openai":
            reply, usage = self._request_openai(effort)
        else:
            reply, usage = self._request_anthropic(effort)
        self.turns += 1
        self.cost += reply.cost
        if self.budget is not None:
            self.budget.add(reply.cost)
        log_spend({"time": now(), "model": self.name,
                    "purpose": self.purpose, "status": reply.status, "cost": round(reply.cost, 6), **usage})
        self.transcript.append({"role": "assistant", "status": reply.status, "text": reply.text,
                                "detail": reply.detail,
                                "calls": [{"name": c.name, "args": c.args} for c in reply.calls]})
        return reply

    def _request_openai(self, effort: str) -> tuple[Reply, dict]:
        # These replies arrive whole, and one at full effort can take over an hour. Any other is given
        # 30 minutes, so that a stalled request is noticed while an executor's machine is still up.
        time_limit = 4500 if effort == "max" else 1800
        if effort == "max":
            effort = self.model.top_effort
        # With store=False nothing is kept on the server, so the model's reasoning comes back
        # encrypted and is returned to it with the history.
        payload = dict(model=self.model.api_id, instructions=self.system, input=self.history,
                       store=False, include=["reasoning.encrypted_content"], reasoning={"effort": effort},
                       truncation="disabled", max_output_tokens=self.max_tokens, prompt_cache_key=self.cache_key)
        if self.tools:
            payload["tools"] = [{"type": "function", "strict": False, **tool} for tool in self.tools]
            payload["parallel_tool_calls"] = True
            payload["prompt_cache_retention"] = "24h"  # the default cache can lapse during one long reply
        elif self.shared and not self.model.base_url:
            # Only the marked opening is written to the cache. Without this the rest of every message is
            # written too, at the higher price, and nothing would ever read it.
            payload["prompt_cache_options"] = {"mode": "explicit"}
        response = _with_retries(lambda: _client(self.model).responses.with_raw_response.create(
            **payload, timeout=time_limit))
        raw = json.loads(response.text)

        used = raw.get("usage") or {}
        details = used.get("input_tokens_details") or {}
        cached = details.get("cached_tokens", 0)
        written = details.get("cache_write_tokens", 0)
        total_in = used.get("input_tokens", 0)  # includes cached and written tokens
        out = used.get("output_tokens", 0)  # includes reasoning
        usage = {"input": total_in - cached - written, "cached": cached, "written": written, "output": out}
        limit, in_factor, out_factor = self.model.long_input or (total_in, 1.0, 1.0)
        cost = self._price(usage, in_factor, out_factor) if total_in > limit else self._price(usage)

        items = raw.get("output") or []
        messages = [item for item in items if item["type"] == "message" and item.get("phase") != "commentary"]
        parts = [part for item in messages for part in item.get("content", [])]
        text = "".join(part.get("text", "") for part in parts if part["type"] == "output_text")
        refusal = next((part["refusal"] for part in parts if part["type"] == "refusal"), None)
        if refusal is not None:
            return Reply(text, [], "refused", cost, refusal), usage
        if raw.get("status") != "completed":
            # A cut-off reply may hold half-written tool calls, so it is neither kept nor acted on.
            return Reply(text, [], "truncated", cost, str(raw.get("incomplete_details") or raw.get("error"))), usage
        # Reasoning items must go back exactly as they came, in order, with their tool calls.
        self.history.extend(items)
        calls = [ToolCall(item["call_id"], item["name"], _arguments(item["arguments"]))
                 for item in items if item["type"] == "function_call"]
        return Reply(text, calls, "tool_use" if calls else "done", cost), usage

    def _request_anthropic(self, effort: str) -> tuple[Reply, dict]:
        payload = dict(model=self.model.api_id, max_tokens=self.max_tokens, system=self.system,
                       messages=self.history, thinking={"type": "adaptive"}, output_config={"effort": effort})
        if self.tools:
            # A conversation with tools has many turns, so the part already sent is cached. The cache
            # is kept for an hour because one long reply can outlast the default five minutes.
            payload["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
            payload["tools"] = [{"name": tool["name"], "description": tool["description"],
                                 "input_schema": tool["parameters"]} for tool in self.tools]

        def request():
            # Streaming keeps long max-effort replies from hitting an HTTP timeout.
            with _client(self.model).messages.stream(**payload) as stream:
                return stream.get_final_message()

        message = _with_retries(request)
        used = message.usage
        usage = {"input": used.input_tokens, "cached": used.cache_read_input_tokens or 0,
                 "written": used.cache_creation_input_tokens or 0, "output": used.output_tokens}
        cost = self._price(usage)

        text = "".join(block.text for block in message.content if block.type == "text")
        if message.stop_reason == "refusal":
            # A refusal is a result of the study. We do not fall back to another model.
            return Reply(text, [], "refused", cost, str(getattr(message, "stop_details", ""))), usage
        if message.stop_reason in ("max_tokens", "model_context_window_exceeded"):
            return Reply(text, [], "truncated", cost, message.stop_reason), usage
        # Thinking blocks are only valid in this conversation and must go back unchanged. They are
        # kept as plain data, exactly as the SDK would send them, so the history can be saved.
        self.history.append({"role": "assistant", "content": [
            block.model_dump(mode="json", exclude_unset=True) for block in message.content]})
        calls = [ToolCall(block.id, block.name, block.input) for block in message.content if block.type == "tool_use"]
        return Reply(text, calls, "tool_use" if calls else "done", cost), usage

    def _price(self, usage: dict, in_factor: float = 1.0, out_factor: float = 1.0) -> float:
        m = self.model
        dollars_in = usage["input"] * m.price_in + usage["cached"] * m.price_cached + usage["written"] * m.price_write
        return (dollars_in * in_factor + usage["output"] * m.price_out * out_factor) / 1_000_000


# Read once, so that a prompt edited while a run is going is not used under the old code version.
PROMPTS = {path.stem: path.read_text().strip() for path in (ROOT / "prompts").glob("*.md")}


def prompt(name: str, **fields: str) -> str:
    """A prompt from the prompts folder, with its $fields filled in."""
    return Template(PROMPTS[name]).substitute(**fields)


def _arguments(text: str) -> dict:
    """A tool call's arguments. If they are not valid JSON, the tool fails and the model is told."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def work(chat: Chat, message: str, run_tool: dict, max_turns: int, spend_limit: float | None = None,
         images: list[bytes] = ()) -> Reply:
    """Send a message and carry out the model's tool calls until it gives a final answer.

    `run_tool` maps a tool's name to a function of its arguments. No new tool calls are run once
    the conversation has used `max_turns` model calls or has spent `spend_limit` or most of its
    budget; the model is then told to answer. There is no limit in tokens, because the two
    providers count the same text differently and such a limit would not be equal.
    """
    reply, notices, retried = chat.send(message, images), 0, False
    while True:
        if reply.status == "truncated" and not retried:
            # A reply cut off by the output limit is asked for once more, with less reasoning.
            reply, retried = chat.retry(effort="medium"), True
        elif reply.status == "tool_use" and notices < 3:
            over_spend = (spend_limit is not None and chat.cost >= spend_limit) or (
                chat.budget is not None and chat.budget.spent >= WRAP_UP * chat.budget.cap)
            if chat.turns >= max_turns or over_spend:
                # Every tool call needs an answer, so we answer, but with a closing notice.
                notices += 1
                # A submit call is still carried out, so that work already done can be handed in.
                results = [(call, _run(run_tool, call) if call.name == "submit" else
                            "No more tool calls: you have reached your limit. Give your final answer now.")
                           for call in reply.calls]
            else:
                results = [(call, _run(run_tool, call)) for call in reply.calls]
            reply = chat.send_results(results)
        else:
            return reply


def _run(run_tool: dict, call: ToolCall) -> str:
    if call.name not in run_tool:
        return f"There is no tool called {call.name}."
    try:
        return run_tool[call.name](call.args)
    except Exception as error:  # a failed tool is reported to the model, not raised
        return f"The tool failed: {type(error).__name__}: {error}"


def extract_json(text: str) -> dict:
    """Pull the JSON object out of a reply, whether or not it is wrapped in a code fence."""
    try:
        return json.loads(text[text.find("{"):text.rfind("}") + 1])
    except json.JSONDecodeError:  # for example two fenced blocks, or text with braces around the answer
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
        if not fenced:
            raise
        return json.loads(fenced.group(1))


def parse(text: str, form: type[BaseModel]) -> tuple[dict | None, list[str]]:
    """Read a JSON answer in the given form. Returns the answer, or None and what is wrong with it."""
    try:
        return form.model_validate(extract_json(text)).model_dump(), []
    except ValidationError as error:
        return None, [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()]
    except (json.JSONDecodeError, TypeError):
        return None, ["The reply is not a JSON object."]


def ask(chat: Chat, message: str, check, run_tool: dict | None = None, max_turns: int = 10,
        spend_limit: float | None = None, images: list[bytes] = ()):
    """Get an answer that passes `check`, sending a failing answer back at most twice.

    `check(text)` returns the accepted answer and a list of problems. `ask` returns the answer
    and "done", or None and the reason there is no answer.
    """
    try:
        for _ in range(REPAIR_ROUNDS + 1):
            reply = work(chat, message, run_tool or {}, max_turns, spend_limit, images)
            if reply.status != "done":
                return None, reply.status
            answer, problems = check(reply.text)
            if not problems:
                return answer, "done"
            images = ()  # the images are part of the conversation already
            message = ("This answer cannot be accepted:\n- " + "\n- ".join(problems) +
                       "\n\nReply again with the whole answer, corrected, in the same JSON form.")
    except OverBudget:
        return None, "spending limit reached"
    return None, "no valid answer"
