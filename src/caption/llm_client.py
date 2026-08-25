"""Provider adapters that turn a prompt pair into JSON text."""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from anthropic import AnthropicError, AsyncAnthropic
from openai import AsyncOpenAI, OpenAIError

from caption.llm_json import TranslationError, strip_markdown_json
from caption.progress import log_step
from caption.types import SamplingParams

T = TypeVar("T")
ANTHROPIC_MAX_TOKENS = 8192

# Fallbacks for direct construction outside the CLI. The configured values in
# defaults.toml are what a normal run uses.
DEFAULT_THINKING_SAMPLING = SamplingParams(
    temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0
)
DEFAULT_INSTRUCT_SAMPLING = SamplingParams(
    temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
)
_RETRY_NOTICE = "Your previous response was rejected. Return a corrected response that fixes exactly this: "
LLM_PREFLIGHT_SYSTEM_PROMPT = "You validate LLM connectivity and return only valid JSON."
LLM_PREFLIGHT_USER_PROMPT = 'Return exactly this JSON object: {"ok": true}'


class JsonCompletionClient(Protocol):
    """Provider-neutral JSON completion client."""

    async def complete_json(self, system_prompt: str, user_prompt: str, *, thinking: bool) -> str:
        """
        Complete a JSON-only prompt pair.

        Parameters
        ----------
        system_prompt : str
            System instruction.
        user_prompt : str
            User task prompt.
        thinking : bool
            Whether the task benefits from reasoning tokens. Providers without a
            reasoning switch ignore it.

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
    thinking_sampling: SamplingParams = DEFAULT_THINKING_SAMPLING
    instruct_sampling: SamplingParams = DEFAULT_INSTRUCT_SAMPLING
    enable_thinking: bool = True
    reasoning_effort: str = "high"
    request_timeout: float = 30.0

    async def complete_json(self, system_prompt: str, user_prompt: str, *, thinking: bool) -> str:
        """
        Complete a JSON task with an OpenAI-compatible chat API.

        Parameters
        ----------
        system_prompt : str
            System instruction.
        user_prompt : str
            User task prompt.
        thinking : bool
            Whether to request reasoning tokens. Reasoning is requested only when the
            adapter is configured for it and the task asks for it.

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
                    response_format={"type": "json_object"},
                    **self._request_kwargs(thinking),
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

    def _request_kwargs(self, thinking: bool) -> dict[str, Any]:
        enabled = self.enable_thinking and thinking
        sampling = self.thinking_sampling if enabled else self.instruct_sampling
        # OpenAI-compatible servers disagree on how reasoning is toggled: vLLM reads
        # chat_template_kwargs.enable_thinking, while several hosted gateways read a
        # top-level thinking object. Both switches are sent so that one configuration
        # works across providers; a server ignores the switch it does not implement.
        # top_k, min_p, and repetition_penalty travel the same way, being extensions
        # rather than part of the OpenAI schema.
        extra_body = {
            "thinking": {"type": "enabled" if enabled else "disabled"},
            "chat_template_kwargs": {"enable_thinking": enabled},
            "top_k": sampling.top_k,
            "min_p": sampling.min_p,
            "repetition_penalty": sampling.repetition_penalty,
        }
        kwargs: dict[str, Any] = {
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "presence_penalty": sampling.presence_penalty,
            "extra_body": extra_body,
        }
        if enabled:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return kwargs


@dataclass(frozen=True)
class AnthropicMessagesClient:
    """Anthropic messages API adapter."""

    client: Any
    model: str
    thinking_sampling: SamplingParams = DEFAULT_THINKING_SAMPLING
    instruct_sampling: SamplingParams = DEFAULT_INSTRUCT_SAMPLING
    max_tokens: int = ANTHROPIC_MAX_TOKENS

    async def complete_json(self, system_prompt: str, user_prompt: str, *, thinking: bool) -> str:
        """
        Complete a JSON task with Anthropic messages.

        Parameters
        ----------
        system_prompt : str
            System instruction.
        user_prompt : str
            User task prompt.
        thinking : bool
            Selects the sampling set. It does not enable extended thinking, which needs a
            token budget this adapter does not configure.

        Returns
        -------
        str
            Raw JSON text from the response.

        Raises
        ------
        TranslationError
            If the provider returns an empty response or request error.
        """
        # The messages API accepts temperature, top_p, and top_k; the remaining sampling
        # parameters have no equivalent and are left out.
        sampling = self.thinking_sampling if thinking else self.instruct_sampling
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                top_k=sampling.top_k,
            )
        except AnthropicError as exc:
            raise TranslationError(f"LLM request failed: {exc}") from exc
        return _anthropic_response_text(response)


async def complete_json_task(
    completion_client: JsonCompletionClient,
    operation: str,
    system_prompt: str,
    user_prompt: str,
    thinking: bool,
    parse: Callable[[str], T],
    retries: int,
) -> T:
    """
    Run one JSON task, retrying while the response cannot be trusted.

    A retry covers both transport failures and responses that fail validation, since
    both are recovered the same way: ask the model again. Each task is scoped to a
    single sentence or a single batch, so a retry is cheap and a persistent failure
    surfaces immediately instead of stalling the run.

    The rejection reason is appended to the prompt on every retry. Sampling is nearly
    deterministic, so repeating an identical prompt reproduces an identical rejected
    response; telling the model what was wrong is what makes the attempt differ.

    Parameters
    ----------
    completion_client : JsonCompletionClient
        Provider adapter.
    operation : str
        Human-readable task label used in retry logs and the final error.
    system_prompt : str
        System instruction.
    user_prompt : str
        User task prompt.
    thinking : bool
        Whether the task benefits from reasoning tokens.
    parse : Callable[[str], T]
        Validator that converts raw JSON text into the task result.
    retries : int
        Maximum number of attempts.

    Returns
    -------
    T
        Parsed task result.

    Raises
    ------
    TranslationError
        If every attempt fails.
    """
    last_error: TranslationError | None = None
    for attempt in range(1, max(1, retries) + 1):
        prompt = user_prompt
        if last_error is not None:
            log_step(f"{operation} retry {attempt}/{retries}: {last_error}", icon="🔁")
            prompt = f"{user_prompt}\n\n{_RETRY_NOTICE}{last_error}"
        try:
            content = await completion_client.complete_json(system_prompt, prompt, thinking=thinking)
            return parse(content)
        except TranslationError as exc:
            last_error = exc
    raise TranslationError(f"{operation} failed after {retries} attempt(s): {last_error}") from last_error


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
    thinking_sampling: SamplingParams = DEFAULT_THINKING_SAMPLING,
    instruct_sampling: SamplingParams = DEFAULT_INSTRUCT_SAMPLING,
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
    thinking_sampling : SamplingParams
        Sampling parameters for tasks that request reasoning.
    instruct_sampling : SamplingParams
        Sampling parameters for tasks that do not.
    enable_thinking : bool
        Whether OpenAI-compatible reasoning parameters may be sent.
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
            thinking_sampling=thinking_sampling,
            instruct_sampling=instruct_sampling,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            request_timeout=request_timeout,
        )
    if normalized_provider == "anthropic":
        return AnthropicMessagesClient(
            client=create_anthropic_client(api_key=api_key, base_url=base_url),
            model=model,
            thinking_sampling=thinking_sampling,
            instruct_sampling=instruct_sampling,
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
    content = await completion_client.complete_json(
        LLM_PREFLIGHT_SYSTEM_PROMPT, LLM_PREFLIGHT_USER_PROMPT, thinking=False
    )
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
