import asyncio
import json

import pytest

from caption.llm_client import (
    OpenAIChatCompletionClient,
    validate_llm_completion_client,
)
from caption.llm_json import TranslationError, parse_review_response, parse_translated_lines
from caption.translator import LlmTranslator
from caption.types import SamplingParams, SubtitleLine, WordSpan

WORDS = [
    WordSpan("welcome", 0.0, 0.5),
    WordSpan("to", 0.5, 0.8),
    WordSpan("cs", 0.8, 1.2),
    WordSpan("thirtythree", 1.2, 2.0),
    WordSpan("thirtysix", 2.0, 2.8),
    WordSpan("this", 3.4, 3.7),
    WordSpan("is", 3.7, 3.9),
    WordSpan("tatsu", 3.9, 4.4),
]

SENTENCE_SPLIT = '{"sentence_ends":[5,8],"line_breaks":[3]}'
GLOSSARY = '{"topic":"A lecture.","terms":[{"source":"thirtythree thirtysix","target":"336"}]}'


class ScriptedClient:
    """Completion client that answers by prompt kind, recording every call."""

    def __init__(self, **responses: str | list[str]) -> None:
        self.responses = {kind: value if isinstance(value, list) else [value] for kind, value in responses.items()}
        self.calls: list[tuple[str, bool]] = []

    async def complete_json(self, system_prompt: str, user_prompt: str, *, thinking: bool) -> str:
        kind = _prompt_kind(user_prompt)
        self.calls.append((kind, thinking))
        answers = self.responses[kind]
        return answers[min(sum(1 for call in self.calls if call[0] == kind) - 1, len(answers) - 1)]

    def count(self, kind: str) -> int:
        return sum(1 for call in self.calls if call[0] == kind)


def _prompt_kind(user_prompt: str) -> str:
    if user_prompt.startswith("Find the sentence boundaries"):
        return "split"
    if user_prompt.startswith("Prepare the terminology"):
        return "glossary"
    if user_prompt.startswith("A reviewer scored"):
        return "revision"
    if user_prompt.startswith("Score each"):
        return "review"
    return "translation"


def _translator(client: ScriptedClient, **kwargs: object) -> LlmTranslator:
    defaults: dict[str, object] = {
        "target_language": "Chinese",
        "concurrency": 2,
        "retries": 2,
        "review_rounds": 0,
    }
    return LlmTranslator(completion_client=client, **{**defaults, **kwargs})


def test_translate_splits_the_transcript_and_translates_each_sentence() -> None:
    client = ScriptedClient(
        split=SENTENCE_SPLIT,
        glossary=GLOSSARY,
        translation=[
            '{"lines":[{"source":"Welcome to CS 336.","target":"欢迎来到 CS 336。"}]}',
            '{"lines":[{"source":"This is Tatsu.","target":"这位是 Tatsu。"}]}',
        ],
    )

    draft = _translator(client).translate(WORDS)

    assert [translated.sentence.text for translated in draft.sentences] == [
        "welcome to cs thirtythree thirtysix",
        "this is tatsu",
    ]
    assert [translated.target_text for translated in draft.sentences] == ["欢迎来到 CS 336。", "这位是 Tatsu。"]
    assert draft.sentences[0].sentence.start == 0.0
    assert draft.sentences[0].sentence.end == 2.8
    assert draft.sentences[1].sentence.start == 3.4
    assert draft.glossary.topic == "A lecture."
    assert client.count("translation") == 2


def test_translation_prompt_carries_topic_glossary_and_neighbouring_sentences() -> None:
    client = ScriptedClient(
        split=SENTENCE_SPLIT,
        glossary=GLOSSARY,
        translation='{"lines":[{"source":"x","target":"x"}]}',
    )
    prompts: list[str] = []
    original = client.complete_json

    async def record(system_prompt: str, user_prompt: str, *, thinking: bool) -> str:
        prompts.append(user_prompt)
        return await original(system_prompt, user_prompt, thinking=thinking)

    client.complete_json = record  # type: ignore[method-assign]

    _translator(client).translate(WORDS)

    second = next(prompt for prompt in prompts if "[CURRENT]\nthis is tatsu" in prompt)
    assert "[TOPIC]\nA lecture." in second
    assert "- thirtythree thirtysix -> 336" in second
    assert "welcome to cs thirtythree thirtysix" in second.split("[CURRENT]")[0]


def test_a_retry_tells_the_model_why_the_previous_response_was_rejected() -> None:
    client = ScriptedClient(
        split=SENTENCE_SPLIT,
        glossary=GLOSSARY,
        translation=[
            '{"lines":[]}',
            '{"lines":[{"source":"x","target":"x"}]}',
        ],
    )
    prompts: list[str] = []
    original = client.complete_json

    async def record(system_prompt: str, user_prompt: str, *, thinking: bool) -> str:
        prompts.append(user_prompt)
        return await original(system_prompt, user_prompt, thinking=thinking)

    client.complete_json = record  # type: ignore[method-assign]

    _translator(client).translate(WORDS)

    retried = [prompt for prompt in prompts if "previous response was rejected" in prompt]
    assert retried, "a retry must carry the rejection reason"
    assert "has 0 line(s) but the layout has 1" in retried[0]


def test_unusable_boundaries_degrade_instead_of_failing_the_batch() -> None:
    """Boundaries are advisory, so a nonsense answer still yields covered sentences."""
    client = ScriptedClient(
        split='{"sentence_ends":[99,-1],"line_breaks":[500]}',
        glossary=GLOSSARY,
        translation='{"lines":[{"source":"x","target":"x"}]}',
    )

    draft = _translator(client).translate(WORDS)

    assert client.count("split") == 1
    assert len(draft.sentences) == 1
    assert len(draft.sentences[0].sentence.words) == len(WORDS)


def test_a_malformed_split_response_is_still_retried_and_reported() -> None:
    client = ScriptedClient(
        split="not json",
        glossary=GLOSSARY,
        translation='{"lines":[{"source":"x","target":"x"}]}',
    )

    with pytest.raises(TranslationError, match="Sentence split"):
        _translator(client).translate(WORDS)
    assert client.count("split") == 2


def test_review_revises_only_rejected_sentences_and_stops_when_accepted() -> None:
    client = ScriptedClient(
        split=SENTENCE_SPLIT,
        glossary=GLOSSARY,
        translation=[
            '{"lines":[{"source":"Welcome to CS 336.","target":"欢迎来到三十三三十六。"}]}',
            '{"lines":[{"source":"This is Tatsu.","target":"这位是 Tatsu。"}]}',
        ],
        review=[
            '{"reviews":[{"id":1,"score":2,"issue":"Course number spelled out."},{"id":2,"score":5,"issue":""}]}',
            '{"reviews":[{"id":1,"score":5,"issue":""}]}',
        ],
        revision='{"lines":[{"source":"Welcome to CS 336.","target":"欢迎来到 CS 336。"}]}',
    )
    translator = _translator(client, review_rounds=3)

    reviewed = translator.review(translator.translate(WORDS))

    assert [sentence.target_text for sentence in reviewed] == ["欢迎来到 CS 336。", "这位是 Tatsu。"]
    assert client.count("revision") == 1
    assert client.count("review") == 2


def test_review_stops_at_the_round_budget_when_the_reviewer_never_accepts() -> None:
    client = ScriptedClient(
        split=SENTENCE_SPLIT,
        glossary=GLOSSARY,
        translation='{"lines":[{"source":"x","target":"x"}]}',
        review='{"reviews":[{"id":1,"score":1,"issue":"bad"},{"id":2,"score":1,"issue":"bad"}]}',
        revision='{"lines":[{"source":"y","target":"y"}]}',
    )
    translator = _translator(client, review_rounds=2)

    reviewed = translator.review(translator.translate(WORDS))

    assert [sentence.target_text for sentence in reviewed] == ["y", "y"]
    assert client.count("review") == 2
    assert client.count("revision") == 4


def test_reasoning_is_requested_only_for_tasks_that_need_it() -> None:
    client = ScriptedClient(
        split=SENTENCE_SPLIT,
        glossary=GLOSSARY,
        translation='{"lines":[{"source":"x","target":"x"}]}',
        review='{"reviews":[{"id":1,"score":1,"issue":"bad"},{"id":2,"score":1,"issue":"bad"}]}',
        revision='{"lines":[{"source":"y","target":"y"}]}',
    )
    translator = _translator(client, review_rounds=1)

    translator.review(translator.translate(WORDS))

    thinking_by_kind = {kind: thinking for kind, thinking in client.calls}
    assert thinking_by_kind == {
        "split": False,
        "glossary": True,
        "translation": False,
        "review": False,
        "revision": True,
    }


def test_translated_lines_fill_the_layout_and_keep_its_spans() -> None:
    layout = (SubtitleLine(0.0, 1.0, "a"), SubtitleLine(1.0, 2.0, "b"))
    response = '{"lines":[{"source":"A.","target":"甲"},{"source":"B.","target":"乙"}]}'

    filled = parse_translated_lines(response, layout)

    assert filled == [SubtitleLine(0.0, 1.0, "A.", "甲"), SubtitleLine(1.0, 2.0, "B.", "乙")]
    with pytest.raises(TranslationError, match="has 2 line\\(s\\) but the layout has 1"):
        parse_translated_lines(response, layout[:1])
    with pytest.raises(TranslationError, match="must not be empty"):
        parse_translated_lines('{"lines":[{"source":"a","target":""}]}', layout[:1])


def test_review_parsing_recovers_renumbered_batches_and_rejects_bad_scores() -> None:
    assert parse_review_response('{"reviews":[{"id":7,"score":3,"issue":"x"}]}', [1])[1].score == 3
    with pytest.raises(TranslationError, match="review score must be between"):
        parse_review_response('{"reviews":[{"id":1,"score":9,"issue":""}]}', [1])
    with pytest.raises(TranslationError, match="do not match"):
        parse_review_response('{"reviews":[{"id":1,"score":3,"issue":"x"}]}', [1, 2])


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = type("Message", (), {"content": content})()


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]


class RecordingCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    async def create(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        return FakeResponse(self.content)


class RecordingOpenAIClient:
    def __init__(self, content: str) -> None:
        self.chat = type("Chat", (), {"completions": RecordingCompletions(content)})()


def test_openai_client_sends_the_sampling_set_that_matches_the_mode() -> None:
    client = RecordingOpenAIClient('{"ok": true}')
    adapter = OpenAIChatCompletionClient(
        client=client,
        model="test-model",
        thinking_sampling=SamplingParams(1.0, 0.95, 20, 0.0, 0.0, 1.0),
        instruct_sampling=SamplingParams(0.7, 0.80, 40, 0.1, 1.5, 1.05),
        reasoning_effort="medium",
    )

    asyncio.run(adapter.complete_json("system", "user", thinking=True))
    asyncio.run(adapter.complete_json("system", "user", thinking=False))

    enabled, disabled = client.chat.completions.calls
    assert enabled["reasoning_effort"] == "medium"
    assert enabled["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert (enabled["temperature"], enabled["top_p"], enabled["presence_penalty"]) == (1.0, 0.95, 0.0)
    assert enabled["extra_body"]["top_k"] == 20
    assert "reasoning_effort" not in disabled
    assert disabled["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert (disabled["temperature"], disabled["top_p"], disabled["presence_penalty"]) == (0.7, 0.80, 1.5)
    assert disabled["extra_body"]["top_k"] == 40
    assert disabled["extra_body"]["min_p"] == 0.1
    assert disabled["extra_body"]["repetition_penalty"] == 1.05


def test_disabling_thinking_forces_the_instruct_sampling_set() -> None:
    client = RecordingOpenAIClient('{"ok": true}')
    adapter = OpenAIChatCompletionClient(
        client=client,
        model="test-model",
        thinking_sampling=SamplingParams(1.0, 0.95, 20, 0.0, 0.0, 1.0),
        instruct_sampling=SamplingParams(0.7, 0.80, 20, 0.0, 1.5, 1.0),
        enable_thinking=False,
    )

    asyncio.run(adapter.complete_json("system", "user", thinking=True))

    assert client.chat.completions.calls[0]["temperature"] == 0.7


class SlowCompletions:
    async def create(self, **kwargs: object) -> FakeResponse:
        await asyncio.sleep(5)
        return FakeResponse('{"ok": true}')


class SlowClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": SlowCompletions()})()


def test_openai_client_request_timeout_raises_translation_error() -> None:
    adapter = OpenAIChatCompletionClient(
        client=SlowClient(), model="test-model", enable_thinking=False, request_timeout=0.05
    )

    with pytest.raises(TranslationError, match="timed out"):
        asyncio.run(adapter.complete_json("system", "user", thinking=False))


class PreflightClient:
    def __init__(self, content: str) -> None:
        self.content = content

    async def complete_json(self, system_prompt: str, user_prompt: str, *, thinking: bool) -> str:
        assert thinking is False
        return self.content


def test_preflight_accepts_fenced_json_and_rejects_garbage() -> None:
    validate_llm_completion_client(PreflightClient('```json\n{"ok": true}\n```'))

    with pytest.raises(TranslationError, match="preflight"):
        validate_llm_completion_client(PreflightClient("not json"))

    with pytest.raises(TranslationError, match="preflight"):
        validate_llm_completion_client(PreflightClient(json.dumps({"ok": False})))
