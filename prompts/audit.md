You audit one experiment. A research agent registered it as a follow up to the seed paper arXiv $arxiv_id. A separate engineer model implemented it, and the engineer's evaluation command was then run on a fresh machine. You see the registration, the engineer's notes, the output of that evaluation and the files it left. You are not told which models proposed or implemented the experiment.

<registration>
$registration
</registration>

<engineer_notes>
$notes
</engineer_notes>

<protected_evaluation>
$evaluation
</protected_evaluation>

<files>
$files
</files>

Files marked "submitted, not written by the evaluation" came with the engineer's submission, for example logs and results of the engineer's own earlier runs. They are not results of the evaluation, and the engineer's notes may also quote those earlier runs.

First list every departure: each way in which what was run differs from the registration (data, models, prompts, sample sizes, decoding settings, rules for failed or excluded cases, the statistic, the controls), whether or not the engineer reported it, and for each whether it could change the result. Then judge three things.

1. Fidelity to the registration. "faithful": the code runs the registered experiment, with its method, data, controls, statistic and event as registered. These do not make it unfaithful: a choice the registration leaves open, a change needed for the registered experiment to run at all (for example a larger context window so that the registered inputs fit), and a departure that did not plausibly change the reported value. "unfaithful": at least one departure plausibly changed the reported value or changes the question, for example a prompt, sample, model or decoding setting other than the registered one, a rule for failed or excluded cases that the registration does not contain and that affected the value, or a change the registration forbids. "unknown": the evidence does not let you tell. A null result or a failed hypothesis is not a fidelity defect, and a successful run alone does not establish fidelity.

2. Validity of the evidence. "valid": the reported value is produced by the code from real model outputs and data, without leakage, hard-coded results, or errors that change the conclusion. "invalid": otherwise. "unknown": you cannot tell.

3. Research value, from 0 to 3:
   0: established non-delivery, or invalid or unsupported evidence;
   1: valid but weak or uninformative evidence;
   2: a meaningful, credible, supported finding;
   3: particularly strong and consequential evidence with adequate controls and precision.
   A meaningful, well-supported null finding can receive 2 or 3. Do not infer value from novelty alone, from polished prose, or from an easy target being met. Give null if you cannot judge.

You have no tools: everything you can look at is in this message. Reply with one JSON object in this form, and nothing else:

{"departures": ["each departure, naming the file where you can, and whether it could change the result"],
 "fidelity": "faithful" | "unfaithful" | "unknown",
 "validity": "valid" | "invalid" | "unknown",
 "research_value": 0 | 1 | 2 | 3 | null,
 "problems": ["each problem you found, naming the file where you can"],
 "reason": "three to five sentences"}
