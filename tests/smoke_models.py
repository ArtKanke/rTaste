"""A paid smoke test (a few cents): each model answers through one tool call, at low and at max effort."""

import sys

from rtaste.models import MODELS, Chat
import rtaste.models

rtaste.models.SPEND_LOG = rtaste.models.ROOT / "runs" / "spend-tests.jsonl"  # test spending stays out of the study's log

ADD = {"name": "add", "description": "Add two integers.",
       "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                      "required": ["a", "b"], "additionalProperties": False}}

names = sys.argv[1:] or list(MODELS)
for name in names:
    for effort in ("low", "max"):
        try:
            chat = Chat(name, "You are a careful assistant. Use the add tool for arithmetic.", [ADD],
                        effort=effort, purpose=f"smoke-{effort}", max_tokens=8000)
            reply = chat.send("What is 17 + 25? Call the add tool, then answer with just the number.")
            steps = 0
            while reply.status == "tool_use" and steps < 3:
                results = [(call, str(call.args["a"] + call.args["b"])) for call in reply.calls]
                reply = chat.send_results(results)
                steps += 1
            print(f"{name:7} {effort:4} -> {reply.status:9} {reply.text.strip()[:40]!r:44} turns={chat.turns} cost=${chat.cost:.4f}")
        except Exception as error:
            print(f"{name:7} {effort:4} -> ERROR {type(error).__name__}: {str(error)[:300]}")
