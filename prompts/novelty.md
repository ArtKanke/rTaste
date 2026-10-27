A research agent read the seed paper below and registered the following experiment as a follow up to it.

Seed paper: $title (arXiv $arxiv_id)
$abstract

<experiment>
$experiment
</experiment>

Judge how novel the experiment's contribution was on $launch_day. Search arXiv for the closest prior work and read what you need, the seed paper included. The tools show nothing posted after that day. Choose one label:

- "equivalent": an existing paper already reports this experiment, or one that answers the same question in the same way.
- "close variant": existing work differs only in details such as the dataset, the model or the metric, and largely settles the question.
- "distinct": the question is not answered by any work you found.
- "unknown": you could not determine it.

An "equivalent" judgment must name the paper. Judge only novelty: not whether the experiment is feasible, valuable or likely to succeed. You have no information about its outcome.

Reply with one JSON object in this form:

{"label": "equivalent" | "close variant" | "distinct" | "unknown",
 "closest_work": [{"arxiv_id": "2501.01234", "difference": "how the experiment differs from this paper, or that it does not"}],
 "reason": "two or three sentences"}
