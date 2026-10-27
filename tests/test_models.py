"""The helpers around a conversation, without any paid call."""

import json

import rtaste.models
from rtaste.models import Chat, Reply, ToolCall, extract_json, work


class FakeChat:
    """Stands in for a conversation and gives prepared replies."""

    budget, cost = None, 0.0

    def __init__(self, replies):
        self.replies, self.turns, self.results = list(replies), 0, []

    def _next(self):
        self.turns += 1
        return self.replies.pop(0)

    def send(self, text, images=()):
        return self._next()

    def send_results(self, results):
        self.results.append(results)
        return self._next()

    def retry(self, effort=None):
        return self._next()


def call(name="search"):
    return Reply("", [ToolCall("1", name, {"q": "x"})], "tool_use", 0.0)


def test_a_reply_cut_off_during_the_search_is_asked_for_again_and_the_search_goes_on():
    chat = FakeChat([call(), Reply("", [], "truncated", 0.0), call(), Reply("answer", [], "done", 0.0)])
    reply = work(chat, "go", {"search": lambda args: "found"}, max_turns=10)
    assert reply.text == "answer" and [r[0][1] for r in chat.results] == ["found", "found"]


def test_a_second_cut_off_is_returned_as_it_is():
    chat = FakeChat([Reply("", [], "truncated", 0.0), Reply("", [], "truncated", 0.0)])
    assert work(chat, "go", {}, max_turns=10).status == "truncated"


def test_tool_calls_past_the_limit_get_a_closing_notice_and_a_failing_tool_is_reported():
    def broken(args):
        raise KeyError("query")

    chat = FakeChat([call(), call(), Reply("answer", [], "done", 0.0)])
    work(chat, "go", {"search": broken}, max_turns=2)
    assert "The tool failed: KeyError" in chat.results[0][0][1] and "No more tool calls" in chat.results[1][0][1]


def test_json_is_found_with_or_without_a_code_fence():
    assert extract_json('Here:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('{"method": "prompt ```{question}``` then score"}')["method"].startswith("prompt")
    assert extract_json('```json\n{"a": 1}\n```\nand also\n```json\n{"b": 2}\n```') == {"a": 1}


def first_message(model, shared, monkeypatch):
    """What a conversation would send as its first message, without sending it."""
    monkeypatch.setattr(Chat, "_request", lambda self, effort=None: Reply("", [], "done", 0.0))
    chat = Chat(model, "system", shared=shared)
    chat.send(shared + "the rest")
    return chat.history[0]["content"], chat.cache_key


def test_a_shared_opening_is_a_marked_block_of_its_own_and_gives_one_cache_key(monkeypatch):
    blocks, key = first_message("opus", "a whole paper. ", monkeypatch)
    assert blocks == [{"type": "text", "text": "a whole paper. ", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                      {"type": "text", "text": "the rest"}]
    blocks, _ = first_message("sol", "a whole paper. ", monkeypatch)
    assert blocks == [{"type": "input_text", "text": "a whole paper. ", "prompt_cache_breakpoint": {"mode": "explicit"}},
                      {"type": "input_text", "text": "the rest"}]
    assert first_message("opus", "a whole paper. ", monkeypatch)[1] == key  # the same opening, the same key
    assert first_message("opus", "another paper. ", monkeypatch)[1] != key
    assert first_message("sol", "a whole paper. ", monkeypatch)[1] != key  # a cache belongs to one model


def test_a_provider_that_matches_openings_by_itself_gets_plain_text_and_no_opening_changes_nothing(monkeypatch):
    assert first_message("grok", "a whole paper. ", monkeypatch)[0] == "a whole paper. the rest"
    text, key = first_message("opus", "", monkeypatch)
    assert text == "the rest" and first_message("opus", "", monkeypatch)[1] != key  # a key of its own, as before


def test_only_the_first_message_is_split_and_a_correction_goes_as_plain_text(monkeypatch):
    def reply(self, effort=None):
        self.history.append({"role": "assistant", "content": "an answer"})
        return Reply("", [], "done", 0.0)

    monkeypatch.setattr(Chat, "_request", reply)
    chat = Chat("opus", "system", shared="a whole paper. ")
    chat.send("a whole paper. the rest")
    chat.send("This answer cannot be accepted")
    assert isinstance(chat.history[0]["content"], list) and chat.history[2]["content"] == "This answer cannot be accepted"


def sent_to_openai(model, shared, tools, monkeypatch, tmp_path):
    """The request a conversation in the OpenAI format would make, caught before it leaves."""
    caught = {}

    class Client:
        class responses:
            class with_raw_response:
                @staticmethod
                def create(**payload):
                    caught.update(payload)
                    return type("Response", (), {"text": json.dumps({
                        "status": "completed", "usage": {"input_tokens": 10, "output_tokens": 1},
                        "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]})})

    monkeypatch.setattr(rtaste.models, "_client", lambda model: Client)
    monkeypatch.setattr(rtaste.models, "SPEND_LOG", tmp_path / "spend.jsonl")
    Chat(model, "system", tools=tools, shared=shared).send(shared + "the rest")
    return caught


def test_with_a_shared_opening_openai_is_told_to_cache_the_opening_and_nothing_else(monkeypatch, tmp_path):
    tool = [{"name": "search", "description": "d", "parameters": {"type": "object", "properties": {}}}]
    assert sent_to_openai("sol", "a whole paper. ", (), monkeypatch, tmp_path)["prompt_cache_options"] == {"mode": "explicit"}
    assert "prompt_cache_options" not in sent_to_openai("sol", "", (), monkeypatch, tmp_path)  # as before
    assert "prompt_cache_options" not in sent_to_openai("grok", "a whole paper. ", (), monkeypatch, tmp_path)
    assert "prompt_cache_options" not in sent_to_openai("sol", "a whole paper. ", tool, monkeypatch, tmp_path)  # turns are cached

