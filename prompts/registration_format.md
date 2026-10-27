Reply with one JSON object in this form:

{"experiments": [
  {
    "title": "what the experiment is",
    "seed_connection": "how it follows from the seed paper",
    "hypothesis": "the one claim it tests",
    "contribution": "what it adds beyond the closest work",
    "closest_work": [{"arxiv_id": "2501.01234", "difference": "how this experiment differs"}],
    "inputs": ["public data, code or models, and where to get them"],
    "method": "what is implemented and run, in enough detail for an engineer who has only this registration",
    "event": {
      "description": "the registered event in words",
      "statistic": "the single number the experiment reports",
      "relation": "greater",
      "threshold": 0.0,
      "upper_threshold": null
    },
    "evaluation": {
      "procedure": "how the statistic is computed",
      "raw_outputs": ["outputs/scores.jsonl"],
      "repetitions": 1,
      "controls": "the controls, or why none are needed"
    },
    "resources": {"needs_gpu": true, "models": ["models that are run"], "expected_minutes": 30},
    "full_scale_study": "the larger experiment this MVE would justify if the registered event occurs"
  }
]}

"relation" is "greater", "less" or "within". For "within", the event occurs when the statistic lies between "threshold" and "upper_threshold". "raw_outputs" are plain file paths, relative to the working directory, that the evaluation writes; say what each file holds in "procedure", not in the path. "expected_minutes" is how long the whole experiment takes to run once on the machine. Each registration must stand on its own and must not refer to your other experiments.
