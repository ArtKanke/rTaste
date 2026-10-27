"""Reproduce the tables and the numbers in the text of "rTaste: Do Research Agents Have Taste? Evaluating Early
Stage Research Judgment" from the released dataset.

    python3 analysis/tables.py

Reads data/ (described in data/README.md) with the standard library alone. Prints Tables 1 to 4 and the appendix
tables, then the numbers stated in the text: those of Section 4 and Appendices B and D, and a few of Section 3 and
Appendix A. Each number is labeled with its section.

Every 95% CI is over the seven seed papers, as in the paper: a comparison is made within each condition (or within
each seed paper), averaged within each seed paper, and the seven seed paper values give a mean and a Student's t
interval. A share "of conditions" is averaged the same way.
"""

import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import mean, median, stdev

DATA = Path(__file__).resolve().parent.parent / "data"
MODELS = ("astra", "fable", "opus", "sol", "luna", "muse")
NAME = {"astra": "GPT-6 Astra", "fable": "Claude Fable 5.1", "opus": "Claude Opus 5.5", "sol": "GPT-6.1 Sol",
        "luna": "GPT-6 Luna", "muse": "Muse Spark 1.3", "grok": "Grok 4.7", "All": "All"}
INITIAL = dict(zip(MODELS, "AFOSLM"))
PROVIDER = {"astra": "OpenAI", "sol": "OpenAI", "luna": "OpenAI", "fable": "Anthropic", "opus": "Anthropic",
            "muse": "Meta"}
LARGER, SMALLER = ("astra", "fable", "opus", "sol"), ("luna", "muse")
REVIEWERS = ("grok", "opus", "sol")  # also the three auditors
HARNESSES = ("SR", "RT")
SEEDS = range(1, 8)
T_95 = 2.447  # Student's t for a two-sided 95% interval with 6 degrees of freedom (seven seed papers)
CALL_LIMIT = {"researcher": 60, "proposer1": 30, "proposer2": 30}  # model calls while proposing (rtaste/propose.py)


def read(name):
    with open(DATA / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def yes(row, column):
    return row[column] == "True"


def key(row):
    """The condition a row belongs to: (seed number, condition)."""
    return int(row["seed_number"]), row["condition"]


MVES = read("mves.csv")
ALL_CONDITIONS = read("conditions.csv")
COMMITTED = [c for c in ALL_CONDITIONS if c["status"] == "committed"]
CALLS = read("calls.csv")
DELIVERED = [r for r in MVES if yes(r, "delivered")]
SOUND = [r for r in MVES if yes(r, "sound")]
FIVE = defaultdict(list)  # the five MVEs of each committed condition, in the order written
for r in sorted(MVES, key=lambda r: int(r["mve"])):
    FIVE[key(r)].append(r)
# The base rate forecast for a seed paper: the event rate of the sound MVEs of the other six seed papers.
BASE_RATE = {s: mean(yes(r, "event") for r in SOUND if int(r["seed_number"]) != s) for s in SEEDS}
GPU = defaultdict(float)  # machine minutes by (stage, seed number, condition, mve); work thrown away is left out
for c in CALLS:
    if c["stage"] in ("implementation machine", "rerun machine") and c["machine_minutes"] and c["repeated"] == "False":
        GPU[(c["stage"], *key(c), c["mve"])] += float(c["machine_minutes"])


# ---- counting and intervals

def select(rows, model="All", harness=None):
    """The rows of one proposing model (or all six) under one harness (or both)."""
    return [r for r in rows if model in ("All", r["proposer"]) and harness in (None, r["harness"])]


def count(rows, column):
    return sum(yes(r, column) for r in rows)


def interval(per_seed):
    """The mean of seven seed paper values and its 95% CI."""
    middle = mean(per_seed)
    half = T_95 * stdev(per_seed) / len(per_seed) ** 0.5
    return middle, middle - half, middle + half


def over_seeds(by_condition):
    """Per-condition values averaged within each seed paper, then the interval over the seven."""
    return interval([mean(v for k, v in by_condition.items() if k[0] == s) for s in SEEDS])


def of_conditions(value):
    """The mean over seed papers of a value of each condition's five MVEs, such as whether its first choice is new."""
    return over_seeds({k: value(five) for k, five in FIVE.items()})[0]


def first_choice(five, rank_column, place="1"):
    return next(r for r in five if r[rank_column] == place)


def gpu_minutes(row, stage):
    return GPU[(stage, *key(row), row["mve"])]


# ---- printing

def pct(share):
    return f"{100 * share:.0f}%"


def with_ci(stat, digits=2, scale=1):
    middle, low, high = (scale * x for x in stat)
    return f"{middle:+.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def heading(title):
    print(f"\n\n{title}\n{'=' * len(title)}")


def say(label, value):
    print(f"  {label:<74} {value}")


def show(title, head, rows, notes=()):
    """Print a table. `head` holds the title lines of each column; a row of None is a rule; a cell that is a pair
    (SR, RT) is printed "SR / RT" with the slashes of a column in line."""
    sides = {}
    for i in range(1, len(head)):
        pairs = [r[i] for r in rows if r and isinstance(r[i], tuple)]
        if pairs:
            sides[i] = max(len(a) for a, _ in pairs), max(len(b) for _, b in pairs)
    text = lambda c, i: f"{c[0]:>{sides[i][0]}} / {c[1]:<{sides[i][1]}}" if isinstance(c, tuple) else c
    rows = [r and [text(c, i) for i, c in enumerate(r)] for r in rows]
    depth = max(len(h) for h in head)
    head = [("",) * (depth - len(h)) + tuple(h) for h in head]
    width = [max(len(x) for x in [*head[i], *(r[i] for r in rows if r)]) for i in range(len(head))]
    heading(title)
    for line in range(depth):
        print("  ".join([head[0][line].ljust(width[0])] + [h[line].center(w) for h, w in zip(head[1:], width[1:])]))
    rule = "-" * (sum(width) + 2 * len(width) - 2)
    print(rule)
    for r in rows:
        print(rule if r is None else "  ".join([r[0].ljust(width[0])] + [c.rjust(w) for c, w in zip(r[1:], width[1:])]))
    for note in notes:
        print(note)


def model_table(cells_of):
    """The rows of a table by proposing model: the six models, a rule, then All. cells_of(model) gives the cells."""
    return [None if m is None else [NAME[m]] + cells_of(m) for m in (*MODELS, None, "All")]


# ---- Table 1

def model_measures(model, harness):
    """The eight measures of Table 1 for one model (or All) under one harness, unrounded."""
    rows, sound = select(MVES, model, harness), select(SOUND, model, harness)
    costs = [float(c["condition_cost"]) for c in select(COMMITTED, model, harness)]
    good = count(rows, "good")
    return [good / len(costs), count(rows, "bno_2of3"), count(rows, "non_obvious_2of3"), count(rows, "on_list"),
            100 * count(sound, "informative_2of3") / len(sound), 100 * count(sound, "event") / len(sound),
            mean(costs), 100 * good / sum(costs)]


def team_minus_single(test, refused_as_zero):
    """RT minus SR in how many of the five MVEs pass a test: the mean over the six models within each seed paper,
    then the interval. The refused condition counts as zero of five, or its pair is left out."""
    per_seed = []
    for s in SEEDS:
        gaps = []
        for m in MODELS:
            single = FIVE.get((s, f"{m}-single"))
            if single is not None or refused_as_zero:
                gaps.append(sum(map(test, FIVE[s, f"{m}-team"])) - sum(map(test, single or [])))
        per_seed.append(mean(gaps))
    return interval(per_seed)


def table1():
    shown = ("{:.2f}", "{}", "{}", "{}", "{:.0f}", "{:.0f}", "{:.1f}", "{:.0f}")
    differences = ("{:+d}", "{:+d}", "{:+d}", "{:+.1f}", "{:+.1f}", "{:+.1f}", "{:+.0f}")  # of the unrounded cells
    rows = model_table(lambda m: [(f.format(a), f.format(b))
                                  for f, a, b in zip(shown, model_measures(m, "SR"), model_measures(m, "RT"))])
    rows[MODELS.index("opus")][0] += " *"
    sr, rt = model_measures("All", "SR"), model_measures("All", "RT")
    first = team_minus_single(lambda r: yes(r, "good"), refused_as_zero=True)
    rows.append(["RT minus SR", f"{first[0]:+.2f}"] + [f.format(b - a) for f, a, b in zip(differences, sr[1:], rt[1:])])
    rows.append(["", f"[{first[1]:.2f}, {first[2]:.2f}]"] + [""] * 7)
    head = [("", "", "", "SR / RT"), ("Sound,", "valuable", "and new,", "of five"),
            ("Beyond", "the paper,", "non", "obvious"), ("Rated", "non", "obvious"),
            ("On a", "list of", "obvious", "follow ups"), ("Informative", "evidence,", "% of sound"),
            ("Registered", "event", "occurred,", "% of sound"), ("Cost per", "condition,", "dollars"),
            ("Sound,", "valuable", "and new", "per $100")]
    show("Table 1. Model comparison", head, rows,
         ["* Its single researcher ran on six seed papers (30 MVEs); in the first column, RT minus SR counts the "
          "seventh as zero."])


# ---- Table 2

KINDS = ((("Checks", "the paper"), ("checks",)), (("Reruns, varies", "or explains"), ("reruns", "varies", "explains")),
         (("Builds", "on"), ("builds on",)), (("New", "direction"), ("new direction",)), (("No two", "agree"), ("",)))


def table2():
    rows = model_table(lambda m: [tuple(str(sum(r["relation_2of3"] in answers for r in select(DELIVERED, m, h)))
                                        for h in HARNESSES) for _, answers in KINDS])
    show("Table 2. Kinds of experiment, SR / RT: delivered MVEs by the answer of at least two of the three reviewers",
         [("MVEs of",)] + [title for title, _ in KINDS], rows)


# ---- Table 3

def probability(row, who):
    """The probability that `who` gave to the MVE's registered event."""
    if who == "agent":
        return float(row["agent_probability"])
    if who == "average":  # the mean of the six judges' probabilities
        return mean(probability(row, j) for j in MODELS)
    if who == "base rate":
        return BASE_RATE[int(row["seed_number"])]
    if who == "same model judge":
        who = row["proposer"]
    return float(row[f"judge_{who}_probability"])


def brier(rows, who):
    return mean((probability(r, who) - yes(r, "event")) ** 2 for r in rows)


def brier_over_seeds(score, model="All"):
    """A score of each condition's sound MVEs, such as a Brier score, averaged within each seed paper and then over
    the seven."""
    sound = defaultdict(list)
    for r in select(SOUND, model):
        sound[key(r)].append(r)
    return over_seeds({k: score(rows) for k, rows in sound.items()})


def table3():
    def cells(model):
        sound = select(SOUND, model)
        scores = [brier_over_seeds(lambda rows: brier(rows, who), model)[0] for who in ("agent", *MODELS, "average")]
        marks = [" ", *("*" if j == model else " " for j in MODELS), " "]
        return [f"{count(sound, 'event') / len(sound):.2f}"[1:]] + [f"{s:.3f}"[1:] + m for s, m in zip(scores, marks)]

    head = [("MVEs of",), ("Event", "rate"), ("", "Agent")] + [("", INITIAL[j]) for j in MODELS] + [("", "Avg.")]
    show("Table 3. Brier scores on the 369 sound MVEs, by whose MVEs are forecast (rows) and who forecasts (columns)",
         head, model_table(cells),
         ["* the same model judge. Judges: " + ", ".join(f"{INITIAL[j]} {NAME[j]}" for j in MODELS) + ".",
          "Avg.: the Brier score of the mean of the six judges' probabilities."])


# ---- Table 4

def table4():
    measures = (lambda r: sum(float(r[f"attempt{a}_output_tokens"] or 0) for a in "12") / 1000,
                lambda r: sum(float(r[f"attempt{a}_minutes"] or 0) for a in "12"),
                lambda r: gpu_minutes(r, "implementation machine"),
                lambda r: gpu_minutes(r, "rerun machine"))
    rows = model_table(lambda m: [tuple(f"{mean(map(f, select(MVES, m, h))):.1f}" for h in HARNESSES)
                                  for f in measures])
    head = [("", "MVEs of"), ("Executor output", "tokens, thousands"), ("Executor working", "time, minutes"),
            ("GPU minutes,", "implementation"), ("GPU minutes,", "rerun")]
    show("Table 4. Execution tokens and compute per registered MVE, SR / RT", head, rows)


# ---- Appendix tables

def app_seeds():
    rows = []
    for s in read("seeds.csv"):
        posted = date.fromisoformat(s["first_posted"])
        rows.append([f"{s['seed_number']}  {s['title']}", f"{posted.day} {posted:%b %Y}", s["pages"]])
    show("Appendix A (tab:app-seeds). The seven seed papers, in the order they were run",
         [("Seed paper",), ("Submitted on arXiv",), ("Pages",)], rows)


FORMS = (("A plain threshold", "plain threshold"), ("A bound of a confidence interval", "confidence bound"),
         ("A prerequisite or several conditions", "composite"), ("All", None))


def app_event_forms():
    rows = []
    for label, form in FORMS:
        if form is None:
            rows.append(None)
        mves = [r for r in MVES if form in (None, r["event_form"])]
        sound = [r for r in mves if yes(r, "sound")]
        rows.append([label] + [str(len(select(mves, m))) for m in (*MODELS, "All")] +
                    [f"{count(sound, 'event')} of {len(sound)}"])
    head = ([("Form of the registered event",)] + [tuple(NAME[m].rsplit(" ", 1)) for m in MODELS] +
            [("All",), ("Occurred,", "of sound")])
    show("Appendix A (tab:app-event-forms). The form of the registered event, by proposing model", head, rows)


def kappa(pairs, linear=False):
    """Cohen's kappa of two raters' answers; with linear=True the answers are numbers and a disagreement counts by
    its distance."""
    disagree = (lambda x, y: abs(x - y)) if linear else (lambda x, y: float(x != y))
    left, right = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    seen = mean(disagree(x, y) for x, y in pairs)
    chance = sum(left[x] * right[y] * disagree(x, y) for x in left for y in right) / len(pairs) ** 2
    return 1 - seen / chance


def app_agreement():
    valued = [r for r in DELIVERED if all(r[f"research_value_{a}"] for a in REVIEWERS)]  # 409: three lack one value
    value = lambda r, a: int(r[f"research_value_{a}"])
    judgments = (("Faithfulness (faithful or not)", DELIVERED, lambda r, a: r[f"fidelity_{a}"] == "faithful", False),
                 ("Validity (valid or not)", DELIVERED, lambda r, a: r[f"validity_{a}"] == "valid", False),
                 ("Research value, 0-3 (linear weights)", valued, value, True),
                 ("Research value of 2 or more", valued, lambda r, a: value(r, a) >= 2, False))
    pairs = (("grok", "opus"), ("grok", "sol"), ("opus", "sol"))
    rows = [[label] + [f"{kappa([(answer(r, a), answer(r, b)) for r in mves], linear):.2f}" for a, b in pairs]
            for label, mves, answer, linear in judgments]
    show("Appendix B (tab:app-agreement). Agreement between pairs of auditors (Cohen's kappa)",
         [("Judgment",)] + [(NAME[a] + ",", NAME[b]) for a, b in pairs], rows)


# ---- Numbers in the text

def section_3_and_appendix_a():
    print("\nSection 3 and Appendix A")
    cut_off = [c for c in CALLS if c["stage"] == "proposing" and c["status"] == "truncated"]
    whose = sorted({f"{c['role']} of {c['condition']}" for c in cut_off})
    say("3.2 seed papers on which a proposing reply was cut off; whose",
        f"{len({c['seed'] for c in cut_off})} of 7; {', '.join(whose)}")
    say("3.2 registrations written at medium effort", f"{5 * len({key(c) for c in cut_off})} of {len(MVES)}")
    same = sum(r["executor"] == r["proposer"] for r in DELIVERED)
    say("3.4 delivered MVEs implemented by the model that proposed them", f"{same} of {len(DELIVERED)}")
    seven = [r for r in MVES if r["seed_number"] == "7"]
    h200 = sum(r["implemented_on"] == r["rerun_on"] == "h200" for r in seven)
    say("3.4, A.3 MVEs of seed paper 7 implemented and rerun on H200 machines", f"{h200} of {len(seven)}")
    relations = Counter(r["relation"] for r in MVES)
    say("A.2 registered events with the relation greater / less", f"{relations['greater']} / {relations['less']}")


def section_4_1():
    print("\n4 and 4.1 Quality of the MVEs")
    say("4 conditions that produced MVEs", f"{len(COMMITTED)} of {len(ALL_CONDITIONS)}")
    say("4.1 MVEs registered / delivered / sound / sound, valuable and new",
        f"{len(MVES)} / {len(DELIVERED)} / {len(SOUND)} / {count(MVES, 'good')}")
    for label, group in (("four larger", LARGER), ("two smaller", SMALLER)):
        of_five = [mean(float(c["good"]) for c in select(COMMITTED, harness=h) if c["proposer"] in group)
                   for h in (None, *HARNESSES)]
        say(f"4.1 sound, valuable and new of five, {label} models (SR, RT)",
            f"{of_five[0]:.2f} ({of_five[1]:.2f}, {of_five[2]:.2f})")
    listed = [r for r in MVES if yes(r, "on_list")]
    say("4.1 MVEs on a list of obvious follow ups that are new", f"{count(listed, 'new')} of {len(listed)}")
    for m in ("luna", "sol", "muse"):
        sound = select(SOUND, m)
        say(f"4.1 {NAME[m]}: sound MVEs; informative evidence, % of sound",
            f"{len(sound)} of {len(select(MVES, m))}; {pct(count(sound, 'informative_2of3') / len(sound))}")


def beyond_and_non_obvious(row, reviewer):
    """One reviewer's answers: builds on or new direction, few or none of ten researchers would list it, and not
    shown to be in the paper (reported or named, with the quoted passage found)."""
    beyond = row[f"relation_{reviewer}"] in ("builds on", "new direction")
    in_paper = (row[f"in_the_paper_{reviewer}"] in ("reported", "named")
                and yes(row, f"in_the_paper_quote_found_{reviewer}"))
    return beyond and row[f"obvious_{reviewer}"] in ("few", "none") and not in_paper


def section_4_2():
    print("\n4.2 Kinds of experiments")
    relations = Counter(r["relation_2of3"] for r in DELIVERED)
    say("4.2 delivered MVEs that build on / check / open a new direction",
        " / ".join(pct(relations[a] / len(DELIVERED)) for a in ("builds on", "checks", "new direction")))
    say("4.2 MVEs on a list of obvious follow ups", pct(count(MVES, "on_list") / len(MVES)))
    first = [FIVE[4, f"{m}-{h}"][0] for m in LARGER for h in ("single", "team")]
    say("4.2 seed paper 4, larger models: the MVE written first checks the paper",
        f"{sum(r['relation_2of3'] == 'checks' for r in first)} of {len(first)} conditions")
    bno = [r for r in MVES if yes(r, "bno_2of3")]
    say("4.2 MVEs beyond the paper and non obvious; of them on a list",
        f"{len(bno)}; {count(bno, 'on_list')} ({pct(count(bno, 'on_list') / len(bno))})")
    astra_sol = sum(r["proposer"] in ("astra", "sol") for r in bno)
    say("4.2 of them proposed by GPT-6 Astra and GPT-6.1 Sol", f"{astra_sol} ({pct(astra_sol / len(bno))})")
    for v in REVIEWERS:
        by_model = Counter(r["proposer"] for r in DELIVERED if beyond_and_non_obvious(r, v))
        say(f"4.2 beyond the paper and non obvious by reviewer {NAME[v]} alone",
            ", ".join(f"{INITIAL[m]} {by_model[m]}" for m in MODELS))
    share = lambda s, group: mean(yes(r, "bno_2of3") for r in MVES if key(r)[0] == s and r["proposer"] in group)
    per_seed = [share(s, ("astra", "sol")) - share(s, ("fable", "opus")) for s in SEEDS]
    say("4.2 share beyond the paper and non obvious, Astra, Sol minus Fable, Opus",
        with_ci(interval(per_seed), 0, 100) + " points")
    say("4.2 seed papers on which that share is higher", f"{sum(x > 0 for x in per_seed)} of 7")
    for m in ("luna", "astra", "sol"):
        say(f"4.2 {NAME[m]}: builds on, % of delivered; rated non obvious",
            f"{pct(sum(r['relation_2of3'] == 'builds on' for r in select(DELIVERED, m)) / len(select(DELIVERED, m)))}; "
            f"{count(select(MVES, m), 'non_obvious_2of3')}")
    for label, rows in (("beyond the paper and non obvious", [r for r in DELIVERED if yes(r, "bno_2of3")]),
                        ("other delivered MVEs", [r for r in DELIVERED if not yes(r, "bno_2of3")])):
        sound = [r for r in rows if yes(r, "sound")]
        say(f"4.2 {label}: sound, valuable and new; event",
            f"{pct(count(rows, 'good') / len(rows))}; {pct(count(sound, 'event') / len(sound))} of sound")
    good = [r for r in MVES if yes(r, "good")]
    say("4.2 sound, valuable and new MVEs beyond the paper and non obvious",
        f"{count(good, 'bno_2of3')} of {len(good)}")
    larger_study = [mean(r["next_step_2of3"] == "larger study" for r in DELIVERED if r["proposer"] in group)
                    for group in (("fable", "opus"), ("astra", "sol"))]
    say("4.2 reviewers recommend a larger study: Fable and Opus / Astra and Sol",
        " / ".join(map(pct, larger_study)) + " of delivered")


# Seed paper 6: the four MVEs that tested whether a model's own prediction errors harm decoding more than random
# errors of the same size, found by reading the registrations.
OWN_ERRORS = ((6, "astra-single", 1), (6, "astra-team", 1), (6, "sol-single", 1), (6, "sol-team", 2))


def section_4_3():
    print("\n4.3 Registered events and evidence")
    bounds = [(m, sum(r["event_form"] == "confidence bound" for r in select(MVES, m)), len(select(MVES, m)))
              for m in MODELS]
    say("4.3 events comparing a confidence bound with the threshold, by model",
        ", ".join(f"{INITIAL[m]} {k}/{n}" for m, k, n in bounds))
    occurred = [mean(yes(r, "event") for r in SOUND if r["proposer"] in group)
                for group in (("astra", "sol"), ("fable", "opus"))]
    say("4.3 event occurred, % of sound: Astra and Sol / Fable and Opus", " / ".join(map(pct, occurred)))
    own = [FIVE[s, c][n - 1] for s, c, n in OWN_ERRORS]
    say("4.3 seed paper 6, own versus random errors: thresholds; events",
        f"{', '.join(r['threshold'] for r in own)}; {count(own, 'event')} of 4 occurred")
    say("4.3 of these, read as evidence for by all three reviewers",
        f"{sum(all(r[f'evidence_{v}'] == 'for' for v in REVIEWERS) for r in own)} of 4")
    missed = [r for r in SOUND if not yes(r, "event")]
    informative = [r for r in missed if yes(r, "informative_2of3")]
    against = sum(r["evidence_2of3"] == "against" for r in informative)
    say("4.3 sound MVEs whose event did not occur: informative; of them against",
        f"{pct(len(informative) / len(missed))}; {against} of {len(informative)}")
    say("4.3 sound MVEs whose event occurred: evidence inconclusive",
        pct(mean(r["evidence_2of3"] == "inconclusive" for r in SOUND if yes(r, "event"))))


def section_4_4():
    print("\n4.4 Forecasts")
    difference = brier_over_seeds(lambda rows: brier(rows, "agent") - brier(rows, "same model judge"))
    say("4.4 Brier, agent minus same model judge", with_ci(difference, 3))
    forecasters = ("agent", "same model judge", "base rate", "average")
    scores = [brier_over_seeds(lambda rows: brier(rows, who))[0] for who in forecasters]
    say("4.4 Brier: agent / same model judge / base rate / mean of the six judges",
        " / ".join(f"{s:.3f}" for s in scores))
    for m in MODELS:
        sound = select(SOUND, m)
        say(f"4.4 sound MVEs of {NAME[m]}: agent's mean probability / event rate",
            f"{mean(probability(r, 'agent') for r in sound):.2f} / {count(sound, 'event') / len(sound):.2f}")


def section_4_5():
    print("\n4.5 First choices")
    most_valued = lambda column: {key(c): float(c[column]) for c in COMMITTED}
    agent, written, judge = (most_valued(c) for c in ("agent_first_is_most_valued", "first_written_is_most_valued",
                                                      "same_model_judge_first_is_most_valued"))
    say("4.5 most valued: agent's first choice / written first / same model judge's",
        " / ".join(pct(over_seeds(x)[0]) for x in (agent, written, judge)))
    for label, other in (("the MVE written first", written), ("the same model judge's", judge)):
        say(f"4.5 most valued MVE, agent's first choice minus {label}",
            with_ci(over_seeds({k: agent[k] - other[k] for k in agent}), 0, 100) + " points")
    say("4.5 sound, valuable and new: agent's first choice / a random pick",
        f"{pct(of_conditions(lambda five: yes(first_choice(five, 'agent_rank'), 'good')))} / "
        f"{pct(of_conditions(lambda five: count(five, 'good') / 5))}")
    say("4.5 on a list of obvious follow ups: MVE written first / written fifth",
        " / ".join(pct(of_conditions(lambda five: yes(five[i], "on_list"))) for i in (0, 4)))
    say("4.5 agent's first choice is the MVE written first",
        pct(of_conditions(lambda five: first_choice(five, "agent_rank")["mve"] == "1")))
    say("4.5 event occurred: agent's first choice / last choice",
        f"{pct(of_conditions(lambda five: yes(first_choice(five, 'agent_rank'), 'event')))} / "
        f"{pct(of_conditions(lambda five: yes(first_choice(five, 'agent_rank', '5'), 'event')))}")
    judges = [of_conditions(lambda five: first_choice(five, f"judge_{j}_rank")["mve"] == "1") for j in MODELS]
    say("4.5 judge's first choice is the MVE written first, mean of six judges", pct(mean(judges)))
    bno = lambda column: of_conditions(lambda five: yes(first_choice(five, column), "bno_2of3"))
    shares = (bno("agent_rank"), mean(bno(f"judge_{j}_rank") for j in MODELS),
              of_conditions(lambda five: count(five, "bno_2of3") / 5))
    say("4.5 beyond the paper, non obvious: agent's / judges' first choice / random", " / ".join(map(pct, shares)))


def section_4_6():
    print("\n4.6 Single researchers and teams")
    say("4.6 RT minus SR, sound, valuable and new of five",
        with_ci(team_minus_single(lambda r: yes(r, "good"), refused_as_zero=True)))
    for m in (*SMALLER, "opus"):
        sr, rt = model_measures(m, "SR"), model_measures(m, "RT")
        say(f"4.6 {NAME[m]}: sound, valuable and new of five; cost, SR / RT",
            f"{sr[0]:.2f} / {rt[0]:.2f}; ${sr[6]:.2f} / ${rt[6]:.2f}")
    tests = (("check the paper", lambda r: r["relation_2of3"] == "checks"), ("on a list", lambda r: yes(r, "on_list")),
             ("beyond the paper, non obvious", lambda r: yes(r, "bno_2of3")),
             ("rated non obvious", lambda r: yes(r, "non_obvious_2of3")),
             ("confidence bound events", lambda r: r["event_form"] == "confidence bound"))
    for label, test in tests:
        say(f"4.6 RT minus SR of five (refused left out): {label}",
            with_ci(team_minus_single(test, refused_as_zero=False)))
    say("4.6 agent's mean probability, SR / RT",
        " / ".join(f"{mean(probability(r, 'agent') for r in select(MVES, harness=h)):.2f}" for h in HARNESSES))


def per_100(model, harness, test):
    """MVEs that pass a test per $100 of condition cost."""
    costs = sum(float(c["condition_cost"]) for c in select(COMMITTED, model, harness))
    return 100 * sum(map(test, select(MVES, model, harness))) / costs


def section_4_7():
    print("\n4.7 Execution and resources")
    sr, rt = model_measures("All", "SR"), model_measures("All", "RT")
    say("4.7 cost per condition, SR / RT", f"${sr[6]:.2f} / ${rt[6]:.2f}")
    most = max(MODELS, key=lambda m: per_100(m, None, lambda r: yes(r, "good")))
    say("4.7 sound, valuable and new per $100, SR / RT; most, by model", f"{sr[7]:.0f} / {rt[7]:.0f}; {NAME[most]}")
    proposing = {m: mean(float(c["proposing_cost"]) for c in select(COMMITTED, m)) for m in MODELS}
    say("4.7 proposing cost per condition, Claude Fable 5.1 / GPT-6 Luna",
        f"${proposing['fable']:.2f} / ${proposing['luna']:.2f} = {proposing['fable'] / proposing['luna']:.0f} times")
    execution = Counter()
    for c in CALLS:
        if c["stage"] in ("implementation", "implementation machine", "rerun machine") and c["repeated"] == "False":
            execution[c["proposer"]] += float(c["cost"])
    per_mve = {m: execution[m] / len(select(MVES, m)) for m in MODELS}
    high, low = max(per_mve, key=per_mve.get), min(per_mve, key=per_mve.get)
    say("4.7 implementing and rerunning an MVE, highest / lowest by model",
        f"${per_mve[high]:.2f} ({INITIAL[high]}) / ${per_mve[low]:.2f} ({INITIAL[low]}) = "
        f"{per_mve[high] / per_mve[low]:.1f} times")
    cells = [(m, h) for m in MODELS for h in HARNESSES]
    for label, test in (("sound and valuable", lambda r: yes(r, "sound") and yes(r, "valuable")),
                        ("beyond the paper, non obvious", lambda r: yes(r, "bno_2of3"))):
        best = max(cells, key=lambda cell: per_100(*cell, test))
        say(f"4.7 {label} per $100: Luna RT; Sol SR / RT; most",
            f"{per_100('luna', 'RT', test):.0f}; {per_100('sol', 'SR', test):.0f} / {per_100('sol', 'RT', test):.0f}; "
            f"{NAME[best[0]]} {best[1]}")
    first = sum(r["delivered_on_attempt"] == "1" for r in MVES)
    say("4.7 MVEs delivered on the first attempt", f"{first} of {len(MVES)}")
    say("4.7 median minutes of an implementation (first attempts)",
        f"{median(float(r['attempt1_minutes']) for r in MVES):.0f}")
    calls = Counter((key(c), c["role"]) for c in CALLS if c["stage"] == "proposing" and c["role"] in CALL_LIMIT)
    # An agent at its limit is asked for a final answer, one call more than the limit.
    reached = {k for (k, role), n in calls.items() if n > CALL_LIMIT[role] and k in FIVE}
    say("4.7 conditions in which an agent reached its tool call limit",
        f"{len(reached)} of {len(FIVE)} ({pct(len(reached) / len(FIVE))})")
    implementation = lambda group: mean(gpu_minutes(r, "implementation machine")
                                        for r in MVES if r["proposer"] in group)
    say("4.7 GPU minutes to implement an MVE, Fable and Opus over Astra and Sol",
        f"{implementation(('fable', 'opus')) / implementation(('astra', 'sol')):.1f} times")
    for executor in ("opus", "sol"):
        rows = [r for r in DELIVERED if r["executor"] == executor]
        shares = (1 - count(rows, "sound") / len(rows), count(rows, "informative_2of3") / len(rows))
        say(f"4.7 executor {NAME[executor]}: not sound; informative evidence",
            "; ".join(map(pct, shares)) + " of delivered")


def occurs(value, event):
    return value > event["threshold"] if event["relation"] == "greater" else value < event["threshold"]


def appendix_b():
    print("\nAppendix B Labels from models")
    check = read("auditor_check.csv")
    say("B.1 planted defects flagged by at least two auditors: copies / originals",
        f"{count(check, 'flagged_2of3')} of {len(check)} / {count(check, 'original_flagged_2of3')} of {len(check)}")
    both = identical = same_event = rule_holds = 0
    with open(DATA / "mves.jsonl", encoding="utf-8") as f:  # full precision: mves.csv rounds values to 6 decimals
        for line in f:
            mve = json.loads(line)
            outcome, event = mve["outcome"], mve["registration"]["event"]
            if not outcome["delivered"]:
                continue
            rule_holds += outcome["event_occurred"] == occurs(outcome["value"], event)
            if outcome["executor_value"] is not None:
                both += 1
                identical += outcome["executor_value"] == outcome["value"]
                same_event += occurs(outcome["executor_value"], event) == outcome["event_occurred"]
    say("B.1 rerun: MVEs with both values; values identical; same event",
        f"{both}; {identical}; {same_event}")
    say("B.1 rerun: saved event = registered relation and threshold on rerun value",
        f"{rule_holds} of {len(DELIVERED)}")
    for judge in ("opus", "sol"):
        labels = Counter(r[f"novelty_{judge}"] for r in MVES)
        say(f"B.2 novelty judge {NAME[judge]}: distinct / close variant / equivalent",
            " / ".join(str(labels[x]) for x in ("distinct", "close variant", "equivalent")))
    say("B.2 novelty judges' agreement, kappa", f"{kappa([(r['novelty_opus'], r['novelty_sol']) for r in MVES]):.2f}")


def places_higher(seed, columns):
    """How many places higher the auditors rank a sound MVE for each True/False column, by least squares with every
    value taken as a difference from its condition's mean, so that only comparisons within a condition count. Returns
    the first column's coefficient."""
    points = []
    for k, five in FIVE.items():
        rows = [r for r in five if k[0] == seed and yes(r, "sound") and r["mean_value_rank"]]
        if len(rows) < 2:
            continue
        ys = [-float(r["mean_value_rank"]) for r in rows]  # higher is better
        xs = [[float(yes(r, c)) for c in columns] for r in rows]
        y_mean, x_mean = mean(ys), [mean(x[i] for x in xs) for i in range(len(columns))]
        points += [(y - y_mean, [a - b for a, b in zip(x, x_mean)]) for y, x in zip(ys, xs)]
    n = len(columns)
    xx = [[sum(x[i] * x[j] for _, x in points) for j in range(n)] for i in range(n)]
    xy = [sum(x[i] * y for y, x in points) for i in range(n)]
    if n == 1:
        return xy[0] / xx[0][0]
    # Two columns: the first coefficient of the 2 x 2 normal equations, by Cramer's rule.
    return (xx[1][1] * xy[0] - xx[0][1] * xy[1]) / (xx[0][0] * xx[1][1] - xx[0][1] * xx[1][0])


def rank(row, who):
    if who == "auditors":
        return float(row["mean_value_rank"]) if row["mean_value_rank"] else None
    return int(row["agent_rank" if who == "agent" else f"judge_{who}_rank"])


def spearman(xs, ys):
    """Spearman's correlation of two rankings of the same five MVEs, without ties."""
    n = len(xs)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(xs, ys)) / (n * (n * n - 1))


def appendix_d():
    print("\nAppendix D Further analyses of the rankings")
    for label, columns in (("D sound MVEs whose event occurred: places higher by the auditors", ("event",)),
                           ("D the same, with informative evidence held fixed", ("event", "informative_2of3"))):
        say(label, with_ci(interval([places_higher(s, columns) for s in SEEDS])))
    say("D informative evidence, % of sound: event occurred / not",
        " / ".join(pct(mean(yes(r, "informative_2of3") for r in SOUND if yes(r, "event") == e)) for e in (True, False)))
    lean = {}
    for who in ("auditors", "agent", *MODELS):
        gaps = {}
        for k, five in FIVE.items():
            on = [rank(r, who) for r in five if yes(r, "on_list") and rank(r, who) is not None]
            off = [rank(r, who) for r in five if not yes(r, "on_list") and rank(r, who) is not None]
            if on and off:
                gaps[k] = mean(off) - mean(on)
        lean[who] = gaps
    say("D conditions with MVEs both on and off the lists", len(lean["auditors"]))
    for who, gaps in lean.items():
        say(f"D places higher for MVEs on a list: {NAME.get(who, 'the ' + who)}", with_ci(over_seeds(gaps)))
    agreement = defaultdict(dict)  # group of judges -> condition -> mean Spearman of the agent with those judges
    for k, five in FIVE.items():
        model = k[1].split("-")[0]
        groups = defaultdict(list)
        for j in MODELS:
            group = "same model" if j == model else "same provider" if PROVIDER[j] == PROVIDER[model] else "other"
            groups[group].append(spearman([rank(r, "agent") for r in five], [rank(r, j) for r in five]))
        for group, values in groups.items():
            agreement[group][k] = mean(values)
    say("D Spearman, agent with judge: same model / same provider / other providers",
        " / ".join(f"{over_seeds(agreement[g])[0]:.2f}" for g in ("same model", "same provider", "other")))
    paired = {k: v - agreement["other"][k] for k, v in agreement["same provider"].items()}
    say(f"D Spearman, same provider minus other providers ({len(paired)} conditions)", with_ci(over_seeds(paired)))


def main():
    for table in (table1, table2, table3, table4, app_seeds, app_event_forms, app_agreement):
        table()
    heading("Numbers in the text")
    for section in (section_3_and_appendix_a, section_4_1, section_4_2, section_4_3, section_4_4, section_4_5,
                    section_4_6, section_4_7, appendix_b, appendix_d):
        section()


if __name__ == "__main__":
    main()
