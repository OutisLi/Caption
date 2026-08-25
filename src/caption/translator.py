"""LLM subtitle translation and optimization."""

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, TypeVar

from anthropic import AnthropicError, AsyncAnthropic
from openai import AsyncOpenAI, OpenAIError

from caption.llm_json import (
    TranslationError,
    apply_translations,
    parse_optimized_segments_response,
    parse_translation_response,
    strip_markdown_json,
)
from caption.progress import log_step, progress_gather
from caption.prompts import build_optimization_prompt, build_translation_prompt
from caption.types import SubtitleCue, WordSpan

T = TypeVar("T")
ANTHROPIC_MAX_TOKENS = 8192
LLM_PREFLIGHT_SYSTEM_PROMPT = "You validate LLM connectivity and return only valid JSON."
LLM_PREFLIGHT_USER_PROMPT = 'Return exactly this JSON object: {"ok": true}'


class JsonCompletionClient(Protocol):
    """Provider-neutral JSON completion client."""

    async def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        """
        Complete a JSON-only prompt pair.

        Parameters
        ----------
        system_prompt : str
            System instruction.
        user_prompt : str
            User task prompt.

        Returns
        -------
        str
            Raw JSON text from the provider.
        """


@dataclass(frozen=True)
class OpenAIChatCompletionClient:
    """OpenAI-compatible chat completion adapter."""

    client: Any
    model: str
    enable_thinking: bool = True
    reasoning_effort: str = "high"
    request_timeout: float = 30.0

    async def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        """
        Complete a JSON task with an OpenAI-compatible chat API.

        Parameters
        ----------
        system_prompt : str
            System instruction.
        user_prompt : str
            User task prompt.

        Returns
        -------
        str
            Raw JSON text from the response.

        Raises
        ------
        TranslationError
            If the request times out, or the provider returns an empty response or request error.
        """
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    **self._thinking_kwargs(),
                ),
                timeout=self.request_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise TranslationError(f"LLM request timed out after {self.request_timeout:.0f}s") from exc
        except OpenAIError as exc:
            raise TranslationError(f"LLM request failed: {exc}") from exc
        content = response.choices[0].message.content
        if not content:
            raise TranslationError("LLM response content is empty")
        return content

    def _thinking_kwargs(self) -> dict[str, Any]:
        if not self.enable_thinking:
            return {}
        return {
            "reasoning_effort": self.reasoning_effort,
            "extra_body": {"thinking": {"type": "enabled"}},
        }


@dataclass(frozen=True)
class AnthropicMessagesClient:
    """Anthropic messages API adapter."""

    client: Any
    model: str
    max_tokens: int = ANTHROPIC_MAX_TOKENS

    async def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        """
        Complete a JSON task with Anthropic messages.

        Parameters
        ----------
        system_prompt : str
            System instruction.
        user_prompt : str
            User task prompt.

        Returns
        -------
        str
            Raw JSON text from the response.

        Raises
        ------
        TranslationError
            If the provider returns an empty response or request error.
        """
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.2,
            )
        except AnthropicError as exc:
            raise TranslationError(f"LLM request failed: {exc}") from exc
        return _anthropic_response_text(response)


@dataclass(frozen=True)
class OptimizationWindow:
    """Store one optimization request window."""

    cues: list[SubtitleCue]
    tokens: list[WordSpan]


class LlmTranslator:
    """Translate and optimize subtitle cues with a JSON completion client."""

    def __init__(
        self,
        completion_client: JsonCompletionClient,
        target_language: str,
        concurrency: int = 4,
        optimization_retries: int = 3,
        optimization_window_seconds: float = 30.0,
        max_segment_seconds: float = 5.0,
        max_target_chars: int = 22,
        min_segment_seconds: float = 2.0,
        pause_seconds: float = 1.0,
    ) -> None:
        """
        Create a translator.

        Parameters
        ----------
        completion_client : JsonCompletionClient
            Provider adapter that returns JSON text for prompt pairs.
        target_language : str
            Translation target language. Empty means source-only optimization.
        concurrency : int
            Maximum number of concurrent LLM requests.
        optimization_retries : int
            Maximum optimization attempts per time window.
        optimization_window_seconds : float
            Maximum seconds of subtitle cues per optimization request.
        max_segment_seconds : float
            Maximum duration for each optimized cue.
        max_target_chars : int
            Maximum target-language characters for each optimized cue.
        min_segment_seconds : float
            Minimum duration for optimized cues unless a pause supports a shorter cue.
        pause_seconds : float
            Pause length that allows a short standalone cue.
        """
        self.completion_client = completion_client
        self.target_language = target_language
        self.concurrency = max(1, concurrency)
        self.optimization_retries = optimization_retries
        self.optimization_window_seconds = optimization_window_seconds
        self.max_segment_seconds = max_segment_seconds
        self.max_target_chars = max_target_chars
        self.min_segment_seconds = min_segment_seconds
        self.pause_seconds = pause_seconds

    def translate(self, cues: Sequence[SubtitleCue]) -> list[SubtitleCue]:
        """
        Translate subtitle cues.

        Parameters
        ----------
        cues : Sequence[SubtitleCue]
            Source subtitle cues.

        Returns
        -------
        list[SubtitleCue]
            Cues with target_text filled. If target_language is empty, cues are returned unchanged.
        """
        if not self.target_language:
            return list(cues)
        return asyncio.run(self._translate_async(cues))

    async def _translate_async(self, cues: Sequence[SubtitleCue]) -> list[SubtitleCue]:
        translated_items = await progress_gather(
            list(cues), self.concurrency, "Translation", "request", self._translate_one
        )
        return apply_translations(cues, {cue.index: target_text for cue, target_text in translated_items})

    def optimize(self, cues: Sequence[SubtitleCue], tokens: Sequence[WordSpan] | None = None) -> list[SubtitleCue]:
        """
        Optimize subtitle text and semantic boundaries.

        Parameters
        ----------
        cues : Sequence[SubtitleCue]
            Translated subtitle cues.
        tokens : Sequence[WordSpan] | None
            Timestamped source tokens used for subtitle boundary decisions.

        Returns
        -------
        list[SubtitleCue]
            Optimized subtitle cues with sequential indices.
        """
        return asyncio.run(self._optimize_async(cues, tokens))

    async def _optimize_async(
        self, cues: Sequence[SubtitleCue], tokens: Sequence[WordSpan] | None
    ) -> list[SubtitleCue]:
        optimized: list[SubtitleCue] = []
        source_tokens = list(tokens) if tokens is not None else _tokens_from_cues(cues)
        windows = _optimization_windows(list(cues), source_tokens, max_window_seconds=self.optimization_window_seconds)
        optimized_windows = await progress_gather(
            windows, self.concurrency, "Optimization", "window", self._optimize_with_retries
        )
        for window in optimized_windows:
            optimized.extend(window)
        return [replace(cue, index=index) for index, cue in enumerate(optimized, start=1)]

    async def _translate_one(self, cue: SubtitleCue) -> tuple[SubtitleCue, str]:
        local_cue = replace(cue, index=1)
        translation = await self._complete_json_task(
            operation=f"Translation cue {cue.index}",
            attempts=self.optimization_retries,
            build_prompts=lambda last_error: (
                "You translate subtitles faithfully and return only valid JSON.",
                build_translation_prompt([local_cue], self.target_language),
            ),
            parse=lambda content: parse_translation_response(content, [1])[1],
        )
        return cue, translation

    async def _optimize_with_retries(self, window: OptimizationWindow) -> list[SubtitleCue]:
        local_cues = [replace(cue, index=index) for index, cue in enumerate(window.cues, start=1)]
        return await self._complete_json_task(
            operation=f"Optimization window {window.tokens[0].start:.1f}s-{window.tokens[-1].end:.1f}s",
            attempts=self.optimization_retries,
            build_prompts=lambda last_error: (
                "You generate polished subtitles and return only valid JSON.",
                build_optimization_prompt(
                    local_cues,
                    window.tokens,
                    self.target_language,
                    str(last_error) if last_error else None,
                    self.max_segment_seconds,
                    self.max_target_chars,
                    self.min_segment_seconds,
                    self.pause_seconds,
                ),
            ),
            parse=lambda content: parse_optimized_segments_response(
                content,
                window.tokens,
                max_segment_seconds=self.max_segment_seconds,
                max_target_chars=self.max_target_chars,
                min_segment_seconds=self.min_segment_seconds,
                pause_seconds=self.pause_seconds,
            ),
        )

    async def _complete_json_task(
        self,
        operation: str,
        attempts: int,
        build_prompts: Callable[[Exception | None], tuple[str, str]],
        parse: Callable[[str], T],
    ) -> T:
        retry_count = max(1, attempts)
        last_error: Exception | None = None
        for attempt in range(1, retry_count + 1):
            if last_error is not None:
                log_step(f"{operation} retry {attempt}/{retry_count}: {last_error}", icon="🔁")
            system_prompt, user_prompt = build_prompts(last_error)
            try:
                content = await self.completion_client.complete_json(system_prompt, user_prompt)
                return parse(content)
            except (TranslationError, asyncio.TimeoutError) as exc:
                last_error = exc
        if isinstance(last_error, TranslationError):
            raise last_error
        raise TranslationError(f"{operation} failed after {retry_count} attempt(s): {last_error}") from last_error


class OpenAICompatibleTranslator(LlmTranslator):
    """Translate and optimize subtitle cues with an OpenAI-compatible client."""

    def __init__(
        self,
        client: Any,
        model: str,
        target_language: str,
        concurrency: int = 4,
        enable_thinking: bool = True,
        reasoning_effort: str = "high",
        optimization_retries: int = 3,
        optimization_window_seconds: float = 30.0,
        max_segment_seconds: float = 5.0,
        max_target_chars: int = 22,
        min_segment_seconds: float = 2.0,
        pause_seconds: float = 1.0,
    ) -> None:
        """
        Create an OpenAI-compatible translator.

        Parameters
        ----------
        client : Any
            OpenAI-compatible client with ``chat.completions.create``.
        model : str
            Chat completion model name.
        target_language : str
            Translation target language. Empty means source-only optimization.
        concurrency : int
            Maximum number of concurrent LLM requests.
        enable_thinking : bool
            Whether to pass thinking parameters to compatible APIs.
        reasoning_effort : str
            Reasoning effort passed to compatible APIs.
        optimization_retries : int
            Maximum optimization attempts per time window.
        optimization_window_seconds : float
            Maximum seconds of subtitle cues per optimization request.
        max_segment_seconds : float
            Maximum duration for each optimized cue.
        max_target_chars : int
            Maximum target-language characters for each optimized cue.
        min_segment_seconds : float
            Minimum duration for optimized cues unless a pause supports a shorter cue.
        pause_seconds : float
            Pause length that allows a short standalone cue.
        """
        completion_client = OpenAIChatCompletionClient(
            client=client,
            model=model,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )
        super().__init__(
            completion_client=completion_client,
            target_language=target_language,
            concurrency=concurrency,
            optimization_retries=optimization_retries,
            optimization_window_seconds=optimization_window_seconds,
            max_segment_seconds=max_segment_seconds,
            max_target_chars=max_target_chars,
            min_segment_seconds=min_segment_seconds,
            pause_seconds=pause_seconds,
        )


def create_openai_client(api_key: str, base_url: str | None = None) -> AsyncOpenAI:
    """
    Create an OpenAI-compatible client.

    Parameters
    ----------
    api_key : str
        API key.
    base_url : str | None
        Optional OpenAI-compatible base URL.

    Returns
    -------
    AsyncOpenAI
        Configured client.
    """
    if base_url:
        return AsyncOpenAI(api_key=api_key, base_url=base_url)
    return AsyncOpenAI(api_key=api_key)


def create_anthropic_client(api_key: str, base_url: str | None = None) -> AsyncAnthropic:
    """
    Create an Anthropic client.

    Parameters
    ----------
    api_key : str
        API key.
    base_url : str | None
        Optional Anthropic-compatible base URL.

    Returns
    -------
    AsyncAnthropic
        Configured Anthropic client.
    """
    if base_url:
        return AsyncAnthropic(api_key=api_key, base_url=base_url)
    return AsyncAnthropic(api_key=api_key)


def create_llm_completion_client(
    provider: str,
    api_key: str,
    base_url: str | None,
    model: str,
    enable_thinking: bool = True,
    reasoning_effort: str = "high",
    request_timeout: float = 30.0,
) -> JsonCompletionClient:
    """
    Create a provider-specific JSON completion adapter.

    Parameters
    ----------
    provider : str
        LLM provider name: ``openai`` or ``anthropic``.
    api_key : str
        API key.
    base_url : str | None
        Optional provider-compatible base URL.
    model : str
        Model name.
    enable_thinking : bool
        Whether OpenAI-compatible thinking parameters are enabled.
    reasoning_effort : str
        Reasoning effort for OpenAI-compatible APIs.
    request_timeout : float
        Per-request timeout in seconds for OpenAI-compatible APIs.

    Returns
    -------
    JsonCompletionClient
        Provider-neutral completion adapter.

    Raises
    ------
    ValueError
        If the provider is unknown.
    """
    normalized_provider = provider.strip().lower()
    if normalized_provider == "openai":
        return OpenAIChatCompletionClient(
            client=create_openai_client(api_key=api_key, base_url=base_url),
            model=model,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            request_timeout=request_timeout,
        )
    if normalized_provider == "anthropic":
        return AnthropicMessagesClient(
            client=create_anthropic_client(api_key=api_key, base_url=base_url),
            model=model,
        )
    raise ValueError("llm.provider must be 'openai' or 'anthropic'")


def validate_llm_completion_client(completion_client: JsonCompletionClient) -> None:
    """
    Validate that the configured LLM can complete a minimal JSON request.

    Parameters
    ----------
    completion_client : JsonCompletionClient
        Provider-neutral completion adapter.

    Raises
    ------
    TranslationError
        If the request fails or does not return the expected JSON object.
    """
    asyncio.run(_validate_llm_completion_client_async(completion_client))


async def _validate_llm_completion_client_async(completion_client: JsonCompletionClient) -> None:
    content = await completion_client.complete_json(LLM_PREFLIGHT_SYSTEM_PROMPT, LLM_PREFLIGHT_USER_PROMPT)
    try:
        payload = json.loads(strip_markdown_json(content))
    except json.JSONDecodeError as exc:
        raise TranslationError(f"LLM preflight failed: invalid JSON response: {content}") from exc
    if payload != {"ok": True}:
        raise TranslationError(f"LLM preflight failed: expected {{'ok': true}}, got {payload}")


def _anthropic_response_text(response: Any) -> str:
    content = "".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text" and getattr(block, "text", "")
    ).strip()
    if not content:
        raise TranslationError("LLM response content is empty")
    return content


def _optimization_windows(
    cues: list[SubtitleCue], tokens: list[WordSpan], max_window_seconds: float
) -> list[OptimizationWindow]:
    if not tokens:
        return []
    windows: list[OptimizationWindow] = []
    current: list[WordSpan] = []
    window_start = tokens[0].start
    for token in tokens:
        if current and token.end - window_start > max_window_seconds:
            windows.append(OptimizationWindow(cues=_cues_for_tokens(cues, current), tokens=current))
            current = []
            window_start = token.start
        current.append(token)
    if current:
        windows.append(OptimizationWindow(cues=_cues_for_tokens(cues, current), tokens=current))
    return windows


def _cues_for_tokens(cues: list[SubtitleCue], tokens: list[WordSpan]) -> list[SubtitleCue]:
    start = tokens[0].start
    end = tokens[-1].end
    return [cue for cue in cues if cue.end > start and cue.start < end]


def _tokens_from_cues(cues: Sequence[SubtitleCue]) -> list[WordSpan]:
    return [WordSpan(cue.source_text, cue.start, cue.end) for cue in cues]
