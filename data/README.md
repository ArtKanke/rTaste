# rTaste dataset

The records of the study in "rTaste: Do Research Agents Have Taste? Evaluating Early Stage Research Judgment".
Six models, each working as a single researcher (SR) and as a research team (RT), read seven recent NLP papers
(the seed papers) and each registered five minimum viable experiments (MVEs). That makes 84 conditions; one was
refused, so 83 conditions registered 415 MVEs. Before any MVE ran, each agent forecast whether the registered event
of each of its MVEs would occur and ranked the five, and six judges did the same from the seed paper and the
registrations alone. Two novelty judges compared each MVE with prior work and with lists of obvious follow ups.
A separate executor then attempted every MVE, a rerun on a new machine decided whether the registered event
occurred, and three auditors and three reviewers judged the 412 delivered MVEs. All labels come from models.

The files are built by `analysis/build_dataset.py` from the saved records of the run (`python -m
analysis.build_dataset <run store>`). Building them calls no model.

## Files

| File | One row or line per | Rows | Size |
|---|---|---|---|
| `seeds.csv` | seed paper | 7 | 2 KB |
| `conditions.csv` | condition (model and harness on a seed paper) | 84 | 19 KB |
| `mves.csv` | MVE, with every label and forecast | 415 | 0.3 MB |
| `mves.jsonl` | MVE, with the full text of the registration and of every judgment | 415 | 16.3 MB |
| `rankings.jsonl` | committed condition: every ranking of its five MVEs, with the reasons | 83 | 0.6 MB |
| `follow_up_lists.jsonl` | seed paper and model: the ten follow ups the model listed | 42 | 0.4 MB |
| `calls.csv` | paid model call or machine | 20,593 | 3.0 MB |
| `auditor_check.csv` | copy of an MVE with a planted defect | 10 | 8 KB |

The tables join on `seed`, `condition` and `mve`. True and False are written as `True` and `False`; a blank cell
means there is no value.

Not included: transcripts and chat histories, the text of the seed papers, the executors' code and files, and the
output of the reruns. The runs of the rehearsal paper (2608.31119) are not part of the study and are left out.

## Models

| Id | Model | Roles |
|---|---|---|
| `astra` | GPT-6 Astra | agent, judge, lists follow ups |
| `fable` | Claude Fable 5.1 | agent, judge, lists follow ups |
| `opus` | Claude Opus 5.5 | agent, judge, lists follow ups, executor, novelty judge, auditor, reviewer |
| `sol` | GPT-6.1 Sol | agent, judge, lists follow ups, executor, novelty judge, auditor, reviewer |
| `luna` | GPT-6 Luna | agent, judge, lists follow ups |
| `muse` | Muse Spark 1.3 | agent, judge, lists follow ups |
| `grok` | Grok 4.7 | auditor, reviewer |
| `sonnet` | Claude Sonnet 5.5 | planted the defects of the auditor check only |

The judge of the agent's own model is its same model judge. The machines are `a100-large` (one A100 GPU) and
`h200` (one H200 GPU), both rented from Hugging Face.

## Labels

Each rule below gives the column of `mves.csv` that holds it.

- `sound`: at least two of the three auditors say faithful and at least two say valid. A missing judgment counts as
  unknown. 369 MVEs.
- `research_value`: the median of the three auditors' research values (the lower of two if one auditor gave none);
  0 for an MVE that was not delivered.
- `valuable`: `research_value` is 2 or more. 281 MVEs.
- `new`: both novelty judges say distinct. 296 MVEs.
- `good`: delivered, sound, valuable and new. 194 MVEs.
- `on_list`: either novelty judge matched the MVE to an entry of the pooled follow up lists. 181 MVEs.
- Review labels, from the reviewers' own answers, each true when at least two of the three reviewers give it:
  - `relation_2of3`: the relation to the seed paper that at least two reviewers give; blank if no two agree.
  - `check_2of3`: `relation_2of3` is checks. 99 MVEs, 24% of the delivered; builds on is 271, 66%.
  - `beyond_2of3`: beyond the paper: the reviewer's relation is builds on or new direction.
  - `non_obvious_2of3`: non obvious: the reviewer's answer to obvious is few or none, and the reviewer did not show
    the experiment to be in the paper (in_the_paper reported or named, with its quote found in the paper).
  - `bno_2of3`: beyond the paper and non obvious, both from the same reviewer. 126 MVEs; 73 of the 194 good MVEs.
  - `informative_2of3`: informative: the reviewer's evidence is for, against or mixed.
  - `evidence_2of3`, `next_step_2of3`: the answer at least two reviewers give; blank if no two agree.
- `mean_value_rank`: the mean of the auditors' value ranks of the MVE within its condition (1 is best); blank with
  fewer than two value rankings.
- `auditors_first`: the MVE's share of the auditors' first place: 1 if it alone has the best mean value rank of its
  condition, 1/k if k MVEs share it, otherwise 0. The MVE with the best mean rank is the auditors' most valued MVE.

Review labels are False, and review answers blank, for the three MVEs that were not delivered.

## The form of the registered event

`event_form` labels each registered event as a plain threshold, a confidence bound (the statistic is a bound of an
interval) or composite (several conditions or a prerequisite, without a confidence bound). The label is assigned by a
keyword rule on the event's statistic and description, with hand corrections by one reader for 13 events. The
keywords and the corrections, each with its reason, are listed in `analysis/build_dataset.py`.

## Costs and machines

All costs are in US dollars. On seed 7 (2608.06867), 53 of the 60 MVEs were implemented and rerun on H200 machines,
which cost twice as much per minute as the A100 machines used for everything else. Columns named `cost` price H200
time at the A100 rate, so that seed papers compare; `cost_paid` is what was paid. Some work was thrown away and repeated after a machine failed or the run was paused;
`calls.csv` marks it `repeated`. Its cost was paid, but condition costs and the per-attempt columns leave it out.

## seeds.csv

- `seed`: the seed paper's arXiv id.
- `seed_number`: 1 to 7, the order in which the seed papers were run.
- `title`: the seed paper's title.
- `pages`: its number of pages.
- `first_posted`: the day it was first made public on arXiv.
- `launch_day`: the day the seed paper was run; novelty is judged against work posted up to this day.
- `repository`: the seed paper's code repository.
- `repository_commit`: the commit of that repository the agents and executors read.
- `machines`: the machine types its MVEs ran on.
- `cost`: all paid calls and machines of the seed paper, the reviews included and the auditor check left out.
- `cost_paid`: the same, as paid.

## conditions.csv

- `seed`, `seed_number`: the seed paper.
- `condition`: the proposing model and harness, as `<model>-single` or `<model>-team`.
- `harness`: SR (single researcher) or RT (research team).
- `proposer`: the model id of the agent.
- `status`: committed, or refused (seed 1, `opus-single`, refused while proposing and registered nothing; its counts
  are 0 and its other outcome columns blank).
- `proposing_cost`: the agent's model calls before it committed its MVEs.
- `proposing_calls`: the number of those calls.
- `proposing_prompt_tokens`: the tokens sent in those calls, cached and cache-written tokens included.
- `proposing_output_tokens`: the tokens they returned, hidden reasoning included.
- `arxiv_searches`: the agent's arXiv searches.
- `papers_read`: the papers it opened with the arXiv reader.
- `repo_listings`: its listings of the seed paper's repository.
- `repo_reads`: its reads of files in that repository.
- `replies_cut_off`: proposing replies cut off at the output limit.
- `minutes_first_reply_to_commitment`: minutes from the end of the agent's first call to the commitment of its MVEs.
- `forecasting_minutes`: minutes from that commitment to the agent's last forecasting call.
- `condition_cost`: proposing, the agent's forecasts, and every attempt and rerun of its MVEs.
- `delivered`: MVEs delivered.
- `sound`: MVEs that are sound.
- `new`: MVEs that are new.
- `on_list`: MVEs that are on a follow up list.
- `good`: MVEs that are good.
- `on_list_sol_alone`: MVEs that Sol's matching alone put on a list; blank where Sol refused the matching.
- `mean_value`: the mean research value of the five MVEs.
- `value_rankers`: the auditors whose value ranking exists (Opus refused one).
- `events`: sound MVEs whose registered event occurred.
- `scored`: sound MVEs; the forecasts are scored on these.
- `brier_agent`: the Brier score of the agent's probabilities on the scored MVEs (lower is better).
- `brier_same_model_judge`: the same for the same model judge.
- `brier_judge_<model>`: the same for each of the six judges (astra, fable, opus, sol, luna, muse).
- `agent_first_is_most_valued`: the `auditors_first` of the agent's first choice.
- `first_written_is_most_valued`: the `auditors_first` of the MVE the agent wrote first.
- `same_model_judge_first_is_most_valued`: the `auditors_first` of the same model judge's first choice.
- `judge_<model>_first_is_most_valued`: the same for each of the six judges.

## mves.csv

Identity and registration:

- `seed`, `seed_number`, `condition`, `harness`, `proposer`: as in `conditions.csv`.
- `mve`: 1 to 5, the MVE's place in the agent's registration, which is the order it was written.
- `title`: the MVE's title.
- `relation`: greater or less: the registered event occurs if the statistic is greater (less) than the threshold.
- `threshold`: the registered threshold.
- `event_form`: plain threshold, confidence bound or composite (see above).
- `expected_minutes`: the run time the agent expected.
- `needs_gpu`: whether the agent said the MVE needs the GPU.
- `models_registered`: the open models the MVE runs, separated by "; ".

Delivery and outcome:

- `attempts`: the attempts made, 1 or 2.
- `delivered`: True if an attempt's rerun produced the registered statistic. 412 MVEs.
- `delivered_on_attempt`: the attempt that delivered.
- `executor`: the executor that delivered the MVE (for an MVE not delivered, the first attempt's executor).
- `assigned_executor`: the executor of the first attempt, as the design assigned it.
- `implemented_on`: the machine type of the delivering attempt.
- `rerun_on`: the machine type of its rerun.
- `first_attempt_status`: submitted or no delivery.
- `value`: the statistic computed by the rerun.
- `executor_value`: the statistic in the result the executor submitted, before the rerun.
- `event`: True if the registered event occurred in the rerun.
- `rerun_seconds`: the run time of the rerun's evaluation command.
- `missing_raw_outputs`: registered raw output files the rerun did not write.
- `attempt<k>_executor`: the executor of attempt k (1 or 2); the `attempt<k>_` columns are blank if there was no
  such attempt.
- `attempt<k>_status`: submitted or no delivery.
- `attempt<k>_minutes`: the attempt's working time, model calls included.
- `attempt<k>_calls`: the executor's model calls.
- `attempt<k>_model_cost`: the cost of those calls.
- `attempt<k>_machine_cost`: the cost of its machine.
- `attempt<k>_output_tokens`: the executor's output tokens, hidden reasoning included.
- `attempt<k>_wait_for_machine_minutes`: minutes from the executor's request for a machine to the machine being ready.

Novelty and follow up lists:

- `novelty_<judge>`: the label of novelty judge opus or sol: distinct, close variant, equivalent or unknown.
- `on_list_<judge>`: True if that novelty judge matched the MVE to an entry of the follow up lists; blank where the
  judge refused the matching (Opus, 45 MVEs of seed 1).

Audits, for the delivered MVEs:

- `fidelity_<auditor>`: faithful, unfaithful or unknown: whether what ran is the registered experiment (auditor
  opus, sol or grok).
- `validity_<auditor>`: valid, invalid or unknown: whether the value comes from real outputs without leakage,
  hard-coded results or errors that change the conclusion.
- `research_value_<auditor>`: 0 to 3: no or invalid evidence, weak evidence, a credible finding, a strong and
  consequential one.
- `value_rank_<auditor>`: the MVE's rank, 1 to 5, in the auditor's ranking of the condition's five MVEs by research
  value; blank where that ranking was refused.
- `sound`, `research_value`, `mean_value_rank`, `valuable`, `new`, `good`, `on_list`, `auditors_first`: see Labels.

Forecasts:

- `agent_probability`: the agent's probability that the registered event occurs.
- `agent_rank`: the MVE's place in the agent's ranking, 1 being the MVE most worth pursuing.
- `judge_<model>_probability`, `judge_<model>_rank`: the same for each of the six judges.

Reviews, for the delivered MVEs, one set per reviewer (grok, opus, sol):

- `in_the_paper_<reviewer>`: reported, named or neither: whether the seed paper reports the experiment or names it as
  left undone.
- `relation_<reviewer>`: reruns, checks, varies, explains, builds on or new direction: what the hypothesis is about.
- `obvious_<reviewer>`: most, some, few or none: on how many of ten researchers' first ten follow ups the idea would
  appear.
- `who_would_care_<reviewer>`: nobody, a few groups, one topic or several topics.
- `misreads_paper_<reviewer>`: yes, no or nothing stated: whether the plan rests on a false statement about the paper.
- `answerable_<reviewer>`: yes, only if or no: whether the design could answer its question.
- `prerequisite_<reviewer>`: held, failed or cannot tell: whether what the main comparison rests on held in the run.
- `evidence_<reviewer>`: for, against, mixed, inconclusive or cannot judge: what the run says about the hypothesis.
- `next_step_<reviewer>`: larger study, one more experiment, repair, stop or no decision.
- `in_the_paper_quote_found_<reviewer>`: True if the passage quoted for in_the_paper was found in the paper's text;
  blank if no passage was given.
- `relation_2of3`, `check_2of3`, `beyond_2of3`, `non_obvious_2of3`, `bno_2of3`, `informative_2of3`, `evidence_2of3`,
  `next_step_2of3`: see Labels.

## mves.jsonl

One JSON object per MVE, in the order of `mves.csv`:

- `seed`, `seed_number`, `condition`, `harness`, `proposer`, `mve`: as in `mves.csv`.
- `registration`: the registration as committed: title, seed_connection, hypothesis, contribution, closest_work,
  inputs, method, event (description, statistic, relation, threshold, upper_threshold), evaluation, resources and
  full_scale_study.
- `forecasts`: for `agent` and each `judge_<model>`, the probability, the rank and the reason.
- `novelty`: for each novelty judge, its status, label, the closest work it found (arXiv id and difference) and its
  reason.
- `follow_up_match`: for each novelty judge, its status, `same_as` (the `number` of the matched entry in
  `follow_up_lists.jsonl`, or null) and its reason.
- `outcome`: whether the MVE was delivered; each attempt's executor, status and reason and its rerun's status and
  reason; and, if delivered, the delivering attempt, the executor, the value, the executor's value, whether the event
  occurred, the rerun's seconds, the missing raw outputs and the executor's notes.
- `audits`: for each auditor, its status, the departures from the registration it found, its fidelity, validity and
  research value, the problems it found and its reason.
- `reviews`: for each reviewer, its status, its one-sentence account of what the MVE asks and of what came out, and
  each of the nine answers with its reason (and, where the question asks for them, the quote, `needs`, `which`,
  `named_before` and `what`), and `quotes_found`.

`audits` and `reviews` are empty for the three MVEs that were not delivered.

## rankings.jsonl

One JSON object per committed condition:

- `seed`, `seed_number`, `condition`, `harness`, `proposer`: as in `conditions.csv`.
- `agent`, `judge_<model>`: the ranking, as MVE numbers from the most worth pursuing, and the ranking reason.
- `auditor_<auditor>`: the auditor's status, its ranking of the five MVEs by research value (best first) and its
  reason. Opus refused one ranking (seed 1, `fable-team`); there the ranking is null.

## follow_up_lists.jsonl

One JSON object per seed paper and model. Each model listed ten follow up experiments from the seed paper alone,
without search.

- `seed`, `seed_number`: the seed paper.
- `model`: the model that wrote the list.
- `status`: done.
- `follow_ups`: the ten entries, each with its `number`, title, question and method. The novelty judges saw the 60
  entries of a seed paper as one list, numbered 1 to 60 in the order astra, fable, opus, sol, luna, muse, without the
  models' names; `number` is that number.

## calls.csv

- `seed`, `seed_number`: the seed paper.
- `condition`, `harness`, `proposer`: blank for the follow up lists and the auditor check.
- `mve`: for novelty, audits, reviews, implementation and reruns.
- `attempt`: for implementation and reruns.
- `time`: when the line was logged, in UTC: the end of a call, or the moment a machine was ready or stopped.
- `model`: the model id, or the machine type.
- `provider`: who was paid: OpenAI, Anthropic, Meta, xAI or Hugging Face.
- `status`: for a call, done, tool_use (it asked for a tool), truncated (cut off at the output limit) or refused; for a
  machine, machine started (cost 0) or machine (when it stopped, with its cost and minutes).
- `cost`: the cost, with H200 time at the A100 rate.
- `cost_paid`: what was paid.
- `prompt_tokens`: everything sent to the model: new tokens, tokens read from the cache and tokens written to it. Only
  this sum compares across providers.
- `cached_tokens`: tokens read from the provider's cache.
- `written_tokens`: tokens written to the provider's cache.
- `output_tokens`: tokens returned, hidden reasoning included.
- `machine_minutes`: the billed minutes of a machine, start included.
- `repeated`: True for work that was thrown away: on seed 6, executor calls made before a machine failed to come up
  and a rerun machine that died and was replaced; on seed 7, executor calls and one machine in flight when the run
  was paused by hand.
- `stage`: follow up lists, proposing, agent forecasts, judges, novelty, follow up matching, implementation,
  implementation machine, rerun machine, audits, value ranking, reviews or auditor check.
- `role`: researcher, proposer1, proposer2, critic or lead (the agent), judge, lister, novelty judge, executor,
  machine, auditor, reviewer or planter.

## auditor_check.csv

The auditors were checked once, on seed paper 1, before any later audit existed. Claude Sonnet 5.5 inserted one known
defect into copies of ten delivered MVEs. Each copy was rerun and audited again by the three auditors in new contexts.
At least two auditors flagged 8 of the 10 copies and none of their originals.

- `seed`, `seed_number`, `condition`, `mve`: the MVE the copy was made from.
- `defect`: the defect asked for: 1, the value is replaced by a fixed number that makes the event occur; 2, the
  evaluation leaks; 3, the required control or baseline is removed or made trivially weak; 4, the statistic is
  computed differently from the registration; 5, part of the model outputs are made up in code.
- `change`: the planter's description of what it changed.
- `rerun`: the status of the copy's rerun.
- `value`, `event`: the copy's rerun statistic and whether its registered event occurred.
- `fidelity_<auditor>`, `validity_<auditor>`, `research_value_<auditor>`: the auditor's judgment of the copy.
- `flagged_<auditor>`: True if the auditor judged the copy unfaithful or invalid.
- `original_flagged_<auditor>`: the same, for the auditor's audit of the original MVE.
- `flagged_2of3`, `original_flagged_2of3`: at least two auditors flagged the copy, or the original.

## License

CC BY 4.0.
