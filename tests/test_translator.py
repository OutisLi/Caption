import asyncio

import pytest

from caption.translator import (
    OpenAIChatCompletionClient,
    OpenAICompatibleTranslator,
    TranslationError,
    apply_translations,
    parse_optimized_segments_response,
    parse_translation_response,
    validate_llm_completion_client,
)
from caption.types import SubtitleCue, WordSpan


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = type("Message", (), {"content": content})()


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, content: str | list[str]) -> None:
        self.contents = content if isinstance(content, list) else [content]
        self.calls: list[dict] = []

    async def create(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.contents) - 1)
        return FakeResponse(self.contents[index])


class FakeClient:
    def __init__(self, content: str | list[str]) -> None:
        self.chat = type("Chat", (), {"completions": FakeCompletions(content)})()


class FakeJsonClient:
    def __init__(self, content: str | list[str]) -> None:
        self.contents = content if isinstance(content, list) else [content]
        self.calls: list[tuple[str, str]] = []

    async def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        index = min(len(self.calls) - 1, len(self.contents) - 1)
        return self.contents[index]


def test_translation_response_parsing_and_application() -> None:
    response = '{"translations":[{"id":1,"text":"你好。"},{"id":2,"text":"世界。"}]}'

    assert parse_translation_response(response, expected_ids=[1, 2]) == {1: "你好。", 2: "世界。"}
    cues = [SubtitleCue(index=1, start=0.0, end=1.0, source_text="Hello.")]

    translated = apply_translations(cues, {1: "你好。"})

    assert translated == [SubtitleCue(index=1, start=0.0, end=1.0, source_text="Hello.", target_text="你好。")]
    with pytest.raises(TranslationError):
        parse_translation_response('{"translations":[{"id":1,"text":"你好。"}]}', expected_ids=[1, 2])

    assert parse_translation_response(
        '{"translations":[{"id":101,"text":"你好。"},{"id":102,"text":"世界。"}]}',
        expected_ids=[1, 2],
    ) == {1: "你好。", 2: "世界。"}


def test_translator_uses_batch_local_ids_for_non_contiguous_cue_indices() -> None:
    client = FakeClient(
        [
            '{"translations":[{"id":1,"text":"第二十一句。"}]}',
            '{"translations":[{"id":1,"text":"第二十二句。"}]}',
        ]
    )
    translator = OpenAICompatibleTranslator(client=client, model="test-model", target_language="Chinese", concurrency=2)
    cues = [
        SubtitleCue(index=21, start=20.0, end=21.0, source_text="Sentence twenty one."),
        SubtitleCue(index=22, start=21.0, end=22.0, source_text="Sentence twenty two."),
    ]

    translated = translator.translate(cues)

    assert [(cue.index, cue.target_text) for cue in translated] == [(21, "第二十一句。"), (22, "第二十二句。")]
    assert len(client.chat.completions.calls) == 2


def test_translator_retries_invalid_translation_json_and_preflight_handles_fenced_json() -> None:
    validate_llm_completion_client(FakeJsonClient('```json\n{"ok": true}\n```'))

    with pytest.raises(TranslationError, match="preflight"):
        validate_llm_completion_client(FakeJsonClient("not json"))

    client = FakeClient(
        [
            "not json",
            '{"translations":[{"id":1,"text":"你好。"}]}',
        ]
    )
    translator = OpenAICompatibleTranslator(
        client=client, model="test-model", target_language="Chinese", concurrency=1, optimization_retries=2
    )
    cues = [SubtitleCue(index=1, start=0.0, end=1.0, source_text="Hello.")]

    translated = translator.translate(cues)

    assert translated == [SubtitleCue(index=1, start=0.0, end=1.0, source_text="Hello.", target_text="你好。")]
    assert len(client.chat.completions.calls) == 2


def test_translator_can_optimize_segments_with_new_timestamps() -> None:
    client = FakeClient(
        '{"items":[{"start_token_id":1,"end_token_id":2,"source_text":"Hello, world.","target_text":"你好，世界。"}]}'
    )
    translator = OpenAICompatibleTranslator(client=client, model="test-model", target_language="Chinese", concurrency=1)
    cues = [
        SubtitleCue(index=7, start=0.0, end=1.0, source_text="Hello", target_text="你好"),
        SubtitleCue(index=8, start=1.0, end=2.0, source_text="world", target_text="世界"),
    ]

    optimized = translator.optimize(cues)

    assert optimized == [
        SubtitleCue(index=1, start=0.0, end=2.0, source_text="Hello, world.", target_text="你好，世界。")
    ]


def test_parse_optimized_segments_response_splits_with_token_timestamps() -> None:
    response = (
        '{"items":['
        '{"start_token_id":1,"end_token_id":3,"source_text":"Hello from one cue.","target_text":"你好呀。"},'
        '{"start_token_id":4,"end_token_id":6,"source_text":"Split by tokens.","target_text":"按词切。"}'
        "]}"
    )
    tokens = [
        WordSpan("Hello", 0.0, 0.8),
        WordSpan("from", 0.8, 1.5),
        WordSpan("one cue.", 1.5, 2.4),
        WordSpan("Split", 2.4, 3.0),
        WordSpan("by", 3.0, 3.4),
        WordSpan("tokens.", 3.4, 4.8),
    ]

    optimized = parse_optimized_segments_response(
        response,
        tokens,
        max_segment_seconds=5.0,
        max_target_chars=15,
    )

    assert optimized == [
        SubtitleCue(index=1, start=0.0, end=2.4, source_text="Hello from one cue.", target_text="你好呀。"),
        SubtitleCue(index=2, start=2.4, end=4.8, source_text="Split by tokens.", target_text="按词切。"),
    ]


def test_parse_optimized_segments_response_enforces_segment_constraints() -> None:
    with pytest.raises(TranslationError, match="duration"):
        parse_optimized_segments_response(
            '{"items":[{"start_token_id":1,"end_token_id":2,"source_text":"too long","target_text":"太长"}]}',
            [WordSpan("too", 0.0, 3.0), WordSpan("long", 3.0, 6.0)],
            max_segment_seconds=5.0,
            max_target_chars=15,
        )

    with pytest.raises(TranslationError, match="target text"):
        parse_optimized_segments_response(
            '{"items":[{"start_token_id":1,"end_token_id":1,"source_text":"short","target_text":"这是一段超过十五个字的中文字幕内容"}]}',
            [WordSpan("short", 0.0, 1.0)],
            max_segment_seconds=5.0,
            max_target_chars=15,
        )

    short_tokens = [
        WordSpan("that", 0.0, 0.3),
        WordSpan("so", 0.3, 0.6),
        WordSpan("many", 0.6, 0.9),
        WordSpan("people", 0.9, 1.4),
        WordSpan("wanted", 1.4, 2.0),
    ]

    with pytest.raises(TranslationError, match="too short"):
        parse_optimized_segments_response(
            '{"items":['
            '{"start_token_id":1,"end_token_id":2,"source_text":"that so","target_text":"如此"},'
            '{"start_token_id":3,"end_token_id":5,"source_text":"many people wanted","target_text":"多人想要"}'
            "]}",
            short_tokens,
            max_segment_seconds=5.0,
            max_target_chars=15,
            min_segment_seconds=1.2,
        )

    paused_tokens = [
        WordSpan("yeah", 0.0, 0.3),
        WordSpan("thanks", 0.3, 0.6),
        WordSpan("I", 1.4, 1.5),
        WordSpan("am", 1.5, 1.6),
        WordSpan("Tatsu", 1.6, 2.0),
    ]

    optimized = parse_optimized_segments_response(
        '{"items":['
        '{"start_token_id":1,"end_token_id":2,"source_text":"Yeah, thanks.","target_text":"好的，谢谢。"},'
        '{"start_token_id":3,"end_token_id":5,"source_text":"I am Tatsu.","target_text":"我是Tatsu。"}'
        "]}",
        paused_tokens,
        max_segment_seconds=5.0,
        max_target_chars=15,
        min_segment_seconds=1.2,
    )

    assert [cue.source_text for cue in optimized] == ["Yeah, thanks.", "I am Tatsu."]


def test_empty_target_language_skips_translation_but_can_optimize_source() -> None:
    client = FakeClient(
        '{"items":[{"start_token_id":1,"end_token_id":1,"source_text":"Hello, world.","target_text":""}]}'
    )
    translator = OpenAICompatibleTranslator(client=client, model="test-model", target_language="", concurrency=2)
    cues = [SubtitleCue(index=1, start=0.0, end=1.0, source_text="Hello world")]

    translated = translator.translate(cues)
    optimized = translator.optimize(translated)

    assert translated == cues
    assert optimized == [SubtitleCue(index=1, start=0.0, end=1.0, source_text="Hello, world.", target_text="")]


def test_optimizer_retries_invalid_timestamps_and_raises_after_retry_budget() -> None:
    client = FakeClient(
        [
            '{"items":[{"start_token_id":99,"end_token_id":99,"source_text":"Bad","target_text":"坏"}]}',
            '{"items":[{"start_token_id":1,"end_token_id":1,"source_text":"Hello, world.","target_text":"你好，世界。"}]}',
        ]
    )
    translator = OpenAICompatibleTranslator(
        client=client, model="test-model", target_language="Chinese", concurrency=2, optimization_retries=2
    )
    cues = [SubtitleCue(index=3, start=0.0, end=1.0, source_text="Hello", target_text="你好")]

    assert translator.optimize(cues) == [
        SubtitleCue(index=1, start=0.0, end=1.0, source_text="Hello, world.", target_text="你好，世界。")
    ]
    assert len(client.chat.completions.calls) == 2
    assert "Previous attempt failed validation" in client.chat.completions.calls[1]["messages"][1]["content"]

    client = FakeClient('{"items":[{"start_token_id":99,"end_token_id":99,"source_text":"Bad","target_text":"坏"}]}')
    translator = OpenAICompatibleTranslator(
        client=client, model="test-model", target_language="Chinese", concurrency=2, optimization_retries=2
    )
    cues = [SubtitleCue(index=3, start=0.0, end=1.0, source_text="Hello", target_text="你好")]

    with pytest.raises(TranslationError):
        translator.optimize(cues)
    assert len(client.chat.completions.calls) == 2


class SlowCompletions:
    async def create(self, **kwargs: object) -> FakeResponse:
        await asyncio.sleep(5)
        return FakeResponse('{"ok": true}')


class SlowClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": SlowCompletions()})()


def test_openai_client_request_timeout_raises_translation_error() -> None:
    client = OpenAIChatCompletionClient(
        client=SlowClient(), model="test-model", enable_thinking=False, request_timeout=0.05
    )

    with pytest.raises(TranslationError, match="timed out"):
        asyncio.run(client.complete_json("system", "user"))
