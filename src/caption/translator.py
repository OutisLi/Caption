"""Sentence-level subtitle translation and review-driven refinement."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, replace

from caption.llm_client import JsonCompletionClient, complete_json_task
from caption.llm_json import parse_review_response, parse_translated_lines
from caption.progress import log_step, progress_gather
from caption.prompts import (
    REVIEW_SYSTEM_PROMPT,
    REVISION_SYSTEM_PROMPT,
    TRANSLATION_SYSTEM_PROMPT,
    build_review_prompt,
    build_revision_prompt,
    build_sentence_translation_prompt,
)
from caption.segment import layout_sentence
from caption.transcript import extract_glossary, split_sentences
from caption.types import (
    SentenceLayout,
    SentenceReview,
    TranscriptGlossary,
    WordSpan,
)

# Neighbouring sentences shown to the translator. A sentence is already a complete
# semantic unit, so context only has to resolve referents and terminology; a wider window
# lengthens every prompt without improving either.
CONTEXT_BEFORE = 4
CONTEXT_AFTER = 2

# Sentences scored per review request. Large enough to amortise the prompt, small enough
# that one malformed response is cheap to retry.
REVIEW_BATCH_SIZE = 10


@dataclass(frozen=True)
class TranslationDraft:
    """Carry a first-pass translation together with the context that produced it.

    Parameters
    ----------
    sentences : tuple[SentenceLayout, ...]
        Translated sentences in transcript order.
    glossary : TranscriptGlossary
        Topic and term renderings shared by every sentence of the transcript.
    """

    sentences: tuple[SentenceLayout, ...]
    glossary: TranscriptGlossary


class LlmTranslator:
    """Translate a transcript sentence by sentence and refine it through review rounds.

    Responsibilities are split so that the model never makes a structural decision. It
    judges meaning for one sentence at a time, informed by the transcript topic, a shared
    glossary, and the neighbouring sentences. Everything structural - which words a
    sentence owns, where a display line starts, what timestamp a cue carries, how many
    lines a response must have - is derived by the program from word timestamps. No
    amount of model error can therefore produce malformed timing or an unbounded loop.

    Refinement never re-segments anything. Each round scores the current translations in
    batches and re-translates only the sentences a reviewer rejected, feeding back the
    rejected text and the reported defect. The round budget bounds the total work.
    """

    def __init__(
        self,
        completion_client: JsonCompletionClient,
        target_language: str,
        concurrency: int = 4,
        retries: int = 3,
        max_line_seconds: float = 6.0,
        max_line_chars: int = 60,
        review_rounds: int = 2,
        review_pass_score: int = 4,
    ) -> None:
        """
        Create a translator.

        Parameters
        ----------
        completion_client : JsonCompletionClient
            Provider adapter that returns JSON text for prompt pairs.
        target_language : str
            Translation target language.
        concurrency : int
            Maximum number of concurrent LLM requests.
        retries : int
            Maximum attempts per LLM request before the run fails.
        max_line_seconds : float
            Longest time a single display line may stay on screen.
        max_line_chars : int
            Source-language character budget for one display line.
        review_rounds : int
            Maximum number of score-and-revise rounds.
        review_pass_score : int
            Lowest score accepted without revision, on the reviewer's 1-5 scale.
        """
        self.completion_client = completion_client
        self.target_language = target_language
        self.concurrency = max(1, concurrency)
        self.retries = max(1, retries)
        self.max_line_seconds = max_line_seconds
        self.max_line_chars = max_line_chars
        self.review_rounds = max(0, review_rounds)
        self.review_pass_score = review_pass_score

    def segment(self, words: Sequence[WordSpan]) -> list[SentenceLayout]:
        """
        Restore sentences and lay them out as display lines, without translating.

        Parameters
        ----------
        words : Sequence[WordSpan]
            Timestamped ASR tokens.

        Returns
        -------
        list[SentenceLayout]
            Sentences with untranslated display lines.
        """
        return asyncio.run(self._layout_async(words))

    def translate(self, words: Sequence[WordSpan]) -> TranslationDraft:
        """
        Restore sentences, build a glossary, and translate every sentence.

        Parameters
        ----------
        words : Sequence[WordSpan]
            Timestamped ASR tokens.

        Returns
        -------
        TranslationDraft
            First-pass translation and the transcript context behind it.
        """
        return asyncio.run(self._translate_async(words))

    def review(self, draft: TranslationDraft) -> list[SentenceLayout]:
        """
        Refine a draft through bounded review rounds.

        Parameters
        ----------
        draft : TranslationDraft
            First-pass translation.

        Returns
        -------
        list[SentenceLayout]
            Sentences with revised text. Line spans are unchanged.
        """
        return asyncio.run(self._review_async(draft))

    async def _translate_async(self, words: Sequence[WordSpan]) -> TranslationDraft:
        layouts = await self._layout_async(words)
        glossary = await extract_glossary(
            self.completion_client,
            [layout.sentence for layout in layouts],
            self.target_language,
            concurrency=self.concurrency,
            retries=self.retries,
        )
        log_step(f"Glossary built with {len(glossary.terms)} term(s)", icon="📗")
        translated = await progress_gather(
            list(range(len(layouts))),
            self.concurrency,
            "Translation",
            "sentence",
            lambda position: self._translate_one(layouts, position, glossary),
        )
        return TranslationDraft(sentences=tuple(translated), glossary=glossary)

    async def _layout_async(self, words: Sequence[WordSpan]) -> list[SentenceLayout]:
        sentences = await split_sentences(
            self.completion_client,
            words,
            concurrency=self.concurrency,
            retries=self.retries,
        )
        log_step(f"Transcript split into {len(sentences)} sentence(s)", icon="✂️")
        return [
            layout_sentence(sentence, self.max_line_chars, self.max_line_seconds)
            for sentence in sentences
        ]

    async def _review_async(self, draft: TranslationDraft) -> list[SentenceLayout]:
        current = list(draft.sentences)
        pending = list(range(len(current)))
        for round_number in range(1, self.review_rounds + 1):
            if not pending:
                break
            reviews = await self._score(current, pending, round_number)
            rejected = [position for position in pending if reviews[position].score < self.review_pass_score]
            if not rejected:
                log_step(f"Review round {round_number}: {len(pending)} sentence(s) accepted", icon="✅")
                break
            log_step(
                f"Review round {round_number}: {len(rejected)}/{len(pending)} sentence(s) rejected", icon="📝"
            )
            revisions = await self._revise(current, rejected, reviews, draft.glossary, round_number)
            for position, revised in zip(rejected, revisions):
                current[position] = revised
            pending = rejected
        return current

    async def _score(
        self, translated: list[SentenceLayout], positions: list[int], round_number: int
    ) -> dict[int, SentenceReview]:
        batches = [
            positions[start : start + REVIEW_BATCH_SIZE]
            for start in range(0, len(positions), REVIEW_BATCH_SIZE)
        ]
        scored = await progress_gather(
            batches,
            self.concurrency,
            f"Review round {round_number}",
            "batch",
            lambda batch: self._score_batch(translated, batch),
        )
        return {
            position: review
            for batch, reviews in zip(batches, scored)
            for position, review in zip(batch, reviews)
        }

    async def _score_batch(
        self, translated: list[SentenceLayout], positions: list[int]
    ) -> list[SentenceReview]:
        items = []
        for local_id, position in enumerate(positions, start=1):
            previous_sentences, next_sentences = self._context(translated, position)
            items.append(
                {
                    "id": local_id,
                    "context": " ".join([*previous_sentences, *next_sentences]),
                    "transcript": translated[position].sentence.text,
                    "source": translated[position].source_text,
                    "target": translated[position].target_text,
                }
            )
        local_ids = list(range(1, len(items) + 1))
        reviews = await complete_json_task(
            self.completion_client,
            operation=f"Review of sentence {positions[0] + 1}-{positions[-1] + 1}",
            system_prompt=REVIEW_SYSTEM_PROMPT,
            user_prompt=build_review_prompt(items, self.target_language),
            thinking=False,
            parse=lambda content: parse_review_response(content, local_ids),
            retries=self.retries,
        )
        return [reviews[local_id] for local_id in local_ids]

    async def _revise(
        self,
        translated: list[SentenceLayout],
        positions: list[int],
        reviews: dict[int, SentenceReview],
        glossary: TranscriptGlossary,
        round_number: int,
    ) -> list[SentenceLayout]:
        return await progress_gather(
            positions,
            self.concurrency,
            f"Revision round {round_number}",
            "sentence",
            lambda position: self._revise_one(translated, position, reviews[position], glossary),
        )

    async def _translate_one(
        self, layouts: Sequence[SentenceLayout], position: int, glossary: TranscriptGlossary
    ) -> SentenceLayout:
        layout = layouts[position]
        previous_sentences, next_sentences = self._context(layouts, position)
        return await self._fill_lines(
            layout,
            operation=f"Translation of sentence {layout.sentence.index}",
            system_prompt=TRANSLATION_SYSTEM_PROMPT,
            user_prompt=build_sentence_translation_prompt(
                layout.sentence,
                [line.source_text for line in layout.lines],
                previous_sentences,
                next_sentences,
                glossary,
                self.target_language,
            ),
            thinking=False,
        )

    async def _revise_one(
        self,
        translated: list[SentenceLayout],
        position: int,
        review: SentenceReview,
        glossary: TranscriptGlossary,
    ) -> SentenceLayout:
        layout = translated[position]
        previous_sentences, next_sentences = self._context(translated, position)
        return await self._fill_lines(
            layout,
            operation=f"Revision of sentence {layout.sentence.index}",
            system_prompt=REVISION_SYSTEM_PROMPT,
            user_prompt=build_revision_prompt(
                layout.sentence,
                [line.source_text for line in layout.lines],
                previous_sentences,
                next_sentences,
                glossary,
                self.target_language,
                [{"source": line.source_text, "target": line.target_text} for line in layout.lines],
                review,
            ),
            thinking=True,
        )

    async def _fill_lines(
        self, layout: SentenceLayout, operation: str, system_prompt: str, user_prompt: str, thinking: bool
    ) -> SentenceLayout:
        lines = await complete_json_task(
            self.completion_client,
            operation=operation,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            thinking=thinking,
            parse=lambda content: parse_translated_lines(content, layout.lines),
            retries=self.retries,
        )
        return replace(layout, lines=tuple(lines))

    def _context(self, layouts: Sequence[SentenceLayout], position: int) -> tuple[list[str], list[str]]:
        start = max(0, position - CONTEXT_BEFORE)
        end = min(len(layouts), position + 1 + CONTEXT_AFTER)
        return (
            [layout.sentence.text for layout in layouts[start:position]],
            [layout.sentence.text for layout in layouts[position + 1 : end]],
        )
