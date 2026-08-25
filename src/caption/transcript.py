"""Recover sentences and transcript-wide terminology from raw ASR output.

These two stages exist so that translation can work on semantically complete units and
still stay consistent across a document it never sees as a whole. Segmentation supplies
the unit; the glossary supplies the shared state that parallel, independent translations
would otherwise lack.
"""

from collections.abc import Sequence

from caption.llm_client import JsonCompletionClient, complete_json_task
from caption.llm_json import parse_glossary_response, parse_sentence_boundaries
from caption.progress import progress_gather
from caption.prompts import (
    GLOSSARY_SYSTEM_PROMPT,
    SENTENCE_SPLIT_SYSTEM_PROMPT,
    build_glossary_prompt,
    build_sentence_split_prompt,
)
from caption.sentence import renumber, sentences_from_boundaries, speech_words, split_word_batches
from caption.types import GlossaryTerm, Sentence, TranscriptGlossary, WordSpan

# Words per segmentation request. Small batches keep every word id in easy reach of the
# model and bound the cost of one poor answer.
SENTENCE_SPLIT_BATCH_WORDS = 400

# Words per terminology request. Sized so that a recording of about an hour needs a single
# pass, which avoids reconciling competing renderings of the same term across batches.
GLOSSARY_BATCH_WORDS = 6000


async def split_sentences(
    completion_client: JsonCompletionClient,
    words: Sequence[WordSpan],
    *,
    concurrency: int,
    retries: int,
) -> list[Sentence]:
    """
    Group a timestamped word stream into sentences.

    Parameters
    ----------
    completion_client : JsonCompletionClient
        Provider adapter.
    words : Sequence[WordSpan]
        Timestamped ASR tokens.
    concurrency : int
        Maximum number of concurrent LLM requests.
    retries : int
        Maximum attempts per batch.

    Returns
    -------
    list[Sentence]
        Sentences covering every pronounceable token exactly once, indexed from one.

    Raises
    ------
    TranslationError
        If a batch returns no usable JSON within the retry budget.
    """
    batches = split_word_batches(speech_words(words), SENTENCE_SPLIT_BATCH_WORDS)
    if not batches:
        return []
    split = await progress_gather(
        batches,
        concurrency,
        "Sentence split",
        "batch",
        lambda batch: _split_batch(completion_client, batch, retries),
    )
    return renumber([sentence for batch in split for sentence in batch])


async def extract_glossary(
    completion_client: JsonCompletionClient,
    sentences: Sequence[Sentence],
    target_language: str,
    *,
    concurrency: int,
    retries: int,
) -> TranscriptGlossary:
    """
    Extract the topic and required term renderings for the whole transcript.

    Parameters
    ----------
    completion_client : JsonCompletionClient
        Provider adapter.
    sentences : Sequence[Sentence]
        Restored sentences in transcript order.
    target_language : str
        Translation target language.
    concurrency : int
        Maximum number of concurrent LLM requests.
    retries : int
        Maximum attempts per batch.

    Returns
    -------
    TranscriptGlossary
        Topic taken from the first batch that reports one, and the union of the term
        renderings, keeping the first rendering seen for each source term.

    Raises
    ------
    TranslationError
        If a batch cannot be parsed within the retry budget.
    """
    batches = _sentence_batches(sentences, GLOSSARY_BATCH_WORDS)
    if not batches:
        return TranscriptGlossary(topic="")
    extracted = await progress_gather(
        batches,
        concurrency,
        "Glossary",
        "batch",
        lambda batch: _extract_batch(completion_client, batch, target_language, retries),
    )
    return _merge_glossaries(extracted)


async def _split_batch(
    completion_client: JsonCompletionClient, batch: list[WordSpan], retries: int
) -> list[Sentence]:
    def parse(content: str) -> list[Sentence]:
        sentence_ends, line_breaks = parse_sentence_boundaries(content)
        return sentences_from_boundaries(batch, sentence_ends, line_breaks)

    return await complete_json_task(
        completion_client,
        operation=f"Sentence split at {batch[0].start:.1f}s",
        system_prompt=SENTENCE_SPLIT_SYSTEM_PROMPT,
        user_prompt=build_sentence_split_prompt([word.text for word in batch]),
        thinking=False,
        parse=parse,
        retries=retries,
    )


async def _extract_batch(
    completion_client: JsonCompletionClient,
    batch: list[Sentence],
    target_language: str,
    retries: int,
) -> TranscriptGlossary:
    return await complete_json_task(
        completion_client,
        operation=f"Glossary at {batch[0].start:.1f}s",
        system_prompt=GLOSSARY_SYSTEM_PROMPT,
        user_prompt=build_glossary_prompt(" ".join(sentence.text for sentence in batch), target_language),
        thinking=True,
        parse=parse_glossary_response,
        retries=retries,
    )


def _sentence_batches(sentences: Sequence[Sentence], batch_words: int) -> list[list[Sentence]]:
    batches: list[list[Sentence]] = []
    current: list[Sentence] = []
    current_words = 0
    for sentence in sentences:
        if current and current_words + len(sentence.words) > batch_words:
            batches.append(current)
            current = []
            current_words = 0
        current.append(sentence)
        current_words += len(sentence.words)
    if current:
        batches.append(current)
    return batches


def _merge_glossaries(glossaries: Sequence[TranscriptGlossary]) -> TranscriptGlossary:
    topic = next((glossary.topic for glossary in glossaries if glossary.topic), "")
    terms: dict[str, GlossaryTerm] = {}
    for glossary in glossaries:
        for term in glossary.terms:
            terms.setdefault(term.source.casefold(), term)
    return TranscriptGlossary(topic=topic, terms=tuple(terms.values()))
