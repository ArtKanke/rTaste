Here is a research paper, as text taken from a web page. Its figures are not shown and its tables are broken into lines.

<seed_paper id="$arxiv_id">
$paper
</seed_paper>

You review one small experiment that was planned after this paper and then built and run. After the questions below you get its written plan (a "registration"), the notes of the engineer who built it, the output of its evaluation and the files the evaluation left.

The plan's writer had read this paper, could also read the paper's code and data and search the literature, and was told to choose their own question and method. The plan was written before anything was run. Every such plan had the same limits: one GPU for at most 45 minutes, and open models of 8B parameters or fewer. So it is a first small test. A separate engineer model then built the experiment from the paper and the registration, and the engineer's evaluation command was run once more on a fresh machine. You are not told which models wrote the plan or built the experiment.

You give eleven answers: seven about the idea, then four about what came out. The answers are used to count kinds of experiment, so choose the line that fits, not the one that sounds best. For each answer, write the reason in one sentence first, then the answer.

PART ONE: THE IDEA

The first seven answers are about the plan as it was written. Give them as you would have before the run: how the run came out does not make an idea more or less obvious, or its question more or less worth asking.

Four fields of the plan are the writer's own account of it: seed_connection, contribution, closest_work and full_scale_study. They say how the writer sees the link to the paper, what is new, and what larger study could follow. They are claims, not facts. Use them for one answer only, misreads_paper, where every statement the plan makes about the paper is checked against the paper. Give every other answer as you would if these four fields were empty.

These must not move any of the seven answers:
- how long, detailed or polished the text is, or how many controls it lists;
- whether you think the hypothesis is true, or how the run came out;
- that the experiment is small, or uses a smaller model or fewer examples than the paper because of the limits;
- whether this exact experiment has been published.

what_it_asks. One sentence of your own, taken from the hypothesis, the method and the event: what is compared with what, on what, to learn what. Leave out sample sizes, controls, statistics and thresholds.

in_the_paper. Look through the whole paper, appendix included.
- "reported": the paper reports this experiment, or a number or comparison that already answers its hypothesis by the same kind of measurement.
- "named": the paper does not report it, and says in its own words that this question is left undone: as future work, as a limitation, or as an open question.
- "neither": anything else. A sentence that only describes what the paper did, admits a weakness, explains a result without measuring it, or calls for more work in general, is not "named".
For "reported" or "named", copy into "quote" the passage of the paper that shows it: between 5 and 40 words in a row, exactly as printed, with nothing left out. Running prose and prompt listings count; tables and formulas do not. If you cannot point to such a passage, the answer is "neither" and "quote" is null.

relation. What is the hypothesis about? Decide by what the experiment tests and uses, not by its controls or side analyses, and not by whether the text mentions the paper.
The hypothesis is about the paper's own claims or results:
- "reruns": that a reported result comes out again.
- "checks": that a claim or result the paper states holds up, or fails, under a control, a baseline, a test for a confound or a stricter measurement.
- "varies": that a result of the paper holds, or fails, on another model, dataset or language, with the paper's method unchanged.
- "explains": why a reported result arises. Choose this only when the result is taken as real; if the hypothesis is that the result comes from a confound or an artifact, choose "checks".
The hypothesis is about something the paper does not claim or report:
- "builds on": the paper's method, data, benchmark or result is a working part of the experiment, and is changed, extended or put to a use the paper does not make of it. A test of a property or a side effect of the paper's method that the paper makes no claim about belongs here: the paper's claims stand whichever way it comes out.
- "new direction": the experiment uses none of these. The paper only prompted the question.
If two lines fit, choose the one that fits the hypothesis better.

obvious. Picture ten researchers who work in this paper's area and know its literature. Each has just read the paper and writes down the ten follow-up experiments that come to mind first. On how many of the ten lists would the idea of this experiment appear? The idea is the question and the main comparison in your what_it_asks sentence, without the test bed and the measurement. A list contains the idea if it has an experiment that asks the same question with the same main comparison. A list that has only the broader worry does not. Ask what they would write before seeing this plan, not whether the idea looks natural once read.
- "most": six or more lists.
- "some": three to five.
- "few": one or two.
- "none": no list.
These researchers know the standard techniques of the area. That makes an idea built on one of them available to them; still count how many would put it among their first ten.
An elaborate design, an invented test bed, or the lack of a published version does not make an idea less obvious. Being simple to run, or looking likely to work, does not make it more obvious.
A check of the paper can be an idea few would list. A step away from the paper can be one most would list.

who_would_care. Suppose the question behind this experiment, at the scope its hypothesis states, were followed through to a clear answer. Of the outcomes it could have, take the one that would change more. Who would then do or believe something different?
- "nobody".
- "a few groups": only people who work with the one method, benchmark or system the experiment is about.
- "one topic": people across a research topic would change how they do their own work, whatever method or benchmark they use.
- "several topics": people across several topics; a common practice or belief would change.
Count the people who would act on the answer. A question is not wide because it mentions a large field. How close the question is to the paper does not matter here.

misreads_paper. Does the plan depend on a statement about the paper (what it did, found or claimed) that the paper's text shows to be false? Look at every field, the writer's own account included.
- "yes": copy into "quote" the passage of the paper that shows it, by the same rule as above. If you cannot point to such a passage, the answer is "no" and "quote" is null.
- "no".
- "nothing stated": the plan states nothing about the paper that can be checked against its text.
Doubting or testing a claim of the paper is not a misreading. Repeating the paper's own summary of itself is not one either. The writer could see the paper's code and data and you cannot: a file name, a released model or a setting that the paper's text does not mention is not an error, and where the plan follows the code and the code differs from the text, that is not an error.

answerable. As it was designed, could this experiment answer the question in your what_it_asks sentence? Go by everything the plan says it will report, not only its one registered number.
- "yes": a clear result either way would bear on the question.
- "only if": the result can be read only if something else holds first, which the design cannot make sure of: for example that an effect of the paper appears at this small scale, that a model trained in the experiment learns what it is trained for, or that a measurement or a control behaves as it should. This is the answer whether or not the plan checks that thing itself. Name the one such thing that matters most in "needs", in a few words.
- "no": the design cannot answer the question as written, for example because the comparison leaves out what it claims to test, or no number it could produce could be read.
Judge the design as written, not by whether the run happened to work. Do not judge whether the hypothesis is likely to be true. Whether this many cases can give a clear result is not asked here: small size alone is not a reason for "no", and "the effect may be too small to see" is not a reason for "only if".

PART TWO: WHAT CAME OUT

The last four answers are about the run. A good plan can give a run that tells nothing, and a weak plan a run that tells something.

The engineer's notes are the engineer's own account: go by the numbers in the evaluation output and the files. Numbers the evaluation computed count, including checks the engineer added; where you can, check them against the raw outputs shown. Files marked "submitted, not written by the evaluation" came with the engineer's submission, for example logs and results of the engineer's own earlier runs. They are not results of the evaluation, and the engineer's notes may also quote those earlier runs.

The plan registered an event: one number set against one bar. The output says whether it occurred. None of the four answers is that. Writers set bars differently: a point value, a confidence bound, or a number written into the hypothesis. So a missed bar can come with clear evidence and a met bar with none. Look at how large the measured effect is and how sure the run is of it, and judge the same way whichever kind of bar the plan used.

Others have checked whether the code follows the plan. You need not list departures. Take the experiment as it was run.

what_came_out. One sentence with the numbers of the main comparison, how sure the run is of them (its interval if it gives one, otherwise the number of cases and repetitions), and the model and data. Do not say here whether the event occurred.

prerequisite. A test can rest on something that must hold before its main comparison can be read: that an effect of the paper appears at this scale, that a model trained in the experiment learned what it was trained for, or that a measurement or a control behaved as it should. It is something other than the hypothesis itself. If you named such a thing in "needs", look at it first; then look for any other.
- "held": as far as the run shows, nothing of this kind failed, or there was no such thing.
- "failed": one such thing did not hold, or did not pass the plan's own check for it, so the main comparison cannot be read. Name it in "which".
- "cannot tell": the run does not show whether such a thing held. Name it in "which".
With "failed" or "cannot tell", set "named_before" to true if "which" is what you named in "needs", otherwise false.
A fault that the run itself shows did not change the main comparison is not a failure. Too few cases or too much noise is not one either; that belongs under "evidence".

evidence. What does the run say about the plan's hypothesis? A number inside the hypothesis is the plan's bar. Do not answer by which side of it the result fell: a clear effect in the claimed direction, somewhat smaller than that number, is evidence for. If the hypothesis claims that something makes no difference or does not hold up, a run that clearly shows this is evidence for.
- "for": the run shows what the hypothesis claims, and noise alone would not explain it.
- "against": the run could have shown what the hypothesis claims and clearly shows otherwise. The change being tested was really applied and the measurement was not stuck at its floor or ceiling. For a claimed effect, the measured effect is near zero or reversed and an effect large enough to matter is ruled out.
- "mixed": the run is sure of two things that point opposite ways, for example an effect on one model and its reverse on another, or a gain with a measured loss.
- "inconclusive": the run does not tell either way: a prerequisite failed, or the uncertainty leaves room both for no effect and for an effect large enough to matter.
- "cannot judge": the numbers needed are not in what you are shown, or the code shows they do not measure what the plan describes.
If "prerequisite" is "failed", the answer is "inconclusive". If the hypothesis has two parts and one could not be tested, the answer is "inconclusive", not "mixed". If the run gives no interval, judge from the number of cases and repetitions and say so.

next_step. An experiment of this kind is a cheap first test of whether a question deserves a larger study. The question is the one in your what_it_asks sentence. Going only by what this run shows, what should a careful researcher do next about it?
- "larger study": the run justifies a study of many times its cost on the same question. It need not be the study in the plan's last field, and the evidence may be for or against the hypothesis.
- "one more experiment": the run answered its own question, and one gap must be closed by a further small experiment before choosing between a larger study and a stop. Name the experiment in "what".
- "repair": the run did not answer its own question and shows why. The same question must be asked again with the fault removed. Name the fault and the fix in "what". Too few cases is such a fault.
- "stop": the run justifies spending no more on this question, because (a) the idea was credibly shown not to work, or (b) something the whole question depends on was credibly shown not to hold, or (c) the question is settled and a larger study would add nothing that matters. Give the letter in "what".
- "no decision": the run did not answer its own question and does not show what to fix.
A justified stop is as good an answer as a justified larger study, and neither follows from the bar. If "evidence" is "inconclusive" or "cannot judge", the answer is "repair", "no decision", or "stop" on ground (b).

You have no tools: everything you can look at is in this message. Reply with one JSON object in this form and nothing else:

{"what_it_asks": "one sentence",
 "in_the_paper": {"reason": "one sentence", "answer": "reported | named | neither", "quote": null},
 "relation": {"reason": "one sentence", "answer": "reruns | checks | varies | explains | builds on | new direction"},
 "obvious": {"reason": "one sentence", "answer": "most | some | few | none"},
 "who_would_care": {"reason": "one sentence", "answer": "nobody | a few groups | one topic | several topics"},
 "misreads_paper": {"reason": "one sentence", "answer": "yes | no | nothing stated", "quote": null},
 "answerable": {"reason": "one sentence", "answer": "yes | only if | no", "needs": null},
 "what_came_out": "one sentence",
 "prerequisite": {"reason": "one sentence", "answer": "held | failed | cannot tell", "which": null, "named_before": null},
 "evidence": {"reason": "one sentence", "answer": "for | against | mixed | inconclusive | cannot judge"},
 "next_step": {"reason": "one sentence", "answer": "larger study | one more experiment | repair | stop | no decision", "what": null}}

The registration:

<registration>
$registration
</registration>

What was built and what came out:

<engineer_notes>
$notes
</engineer_notes>

<protected_evaluation>
$evaluation
</protected_evaluation>

<files>
$files
</files>
