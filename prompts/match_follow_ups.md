Below are two sets of follow-up experiments to the same paper (arXiv $arxiv_id).

The first is a numbered list of short experiment ideas.

<list>
$follow_ups
</list>

The second is five registered experiments.

<experiments>
$experiments
</experiments>

For each of the five registered experiments, say whether it is the same experiment as one on the list: it asks the same question and answers it with essentially the same manipulation and measurement. A shared topic, dataset or model is not enough.

Reply with one JSON object in this form:

{"matches": [
  {"experiment": 1, "same_as": <the number of the matching list entry, or null>, "reason": "one sentence"},
  ... one entry for each of the five experiments
]}
