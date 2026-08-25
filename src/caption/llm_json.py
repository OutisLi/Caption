"""Parse and validate structured LLM responses."""

import json
from collections.abc import Sequence
from dataclasses import replace

from caption.types import GlossaryTerm, SentenceReview, SubtitleLine, TranscriptGlossary

MIN_REVIEW_SCORE = 1
MAX_REVIEW_SCORE = 5


class TranslationError(RuntimeError):
    """Raised when the LLM response cannot be trusted."""


def parse_sentence_boundaries(response_text: str) -> tuple[list[int], list[int]]:
    """
    Parse a sentence-boundary response.

    Only the JSON shape is checked here. Whether an individual id can be honoured depends
    on the batch it describes and is decided where the words are known; an id that cannot
    be is dropped rather than rejected, so this stage has no way to fail on content.

    Parameters
    ----------
    response_text : str
        Raw LLM response.

    Returns
    -------
    tuple[list[int], list[int]]
        Reported sentence-end ids and line-break ids, as given.

    Raises
    ------
    TranslationError
        If the JSON is invalid or either field is not a list of integers.
    """
    data = _load_json_object(response_text, "sentence split")
    return _int_list(data, "sentence_ends"), _int_list(data, "line_breaks")


def _int_list(data: dict, field: str) -> list[int]:
    values = data.get(field, [])
    if not isinstance(values, list):
        raise TranslationError(f"sentence split {field} must be a list")
    return [_required_int(value, f"sentence split {field} entry") for value in values]


def parse_glossary_response(response_text: str) -> TranscriptGlossary:
    """
    Parse a terminology extraction response.

    Parameters
    ----------
    response_text : str
        Raw LLM response.

    Returns
    -------
    TranscriptGlossary
        Transcript topic and required term renderings. Terms with an empty source or
        target are dropped, since they constrain nothing.

    Raises
    ------
    TranslationError
        If the JSON is invalid or the terms field is not a list.
    """
    data = _load_json_object(response_text, "glossary")
    raw_terms = data.get("terms", [])
    if not isinstance(raw_terms, list):
        raise TranslationError("glossary response terms must be a list")
    terms = []
    for item in raw_terms:
        if not isinstance(item, dict):
            raise TranslationError("glossary term must be an object with source and target")
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        if source and target:
            terms.append(GlossaryTerm(source=source, target=target))
    return TranscriptGlossary(topic=str(data.get("topic", "")).strip(), terms=tuple(terms))


def parse_translated_lines(response_text: str, layout: Sequence[SubtitleLine]) -> list[SubtitleLine]:
    """
    Parse a sentence translation response onto an existing display-line layout.

    The layout is fixed by the audio timing before the request is sent, so the response
    only supplies text. Requiring one entry per line makes the response verifiable
    without any arithmetic on the model's side.

    Parameters
    ----------
    response_text : str
        Raw LLM response.
    layout : Sequence[SubtitleLine]
        Display lines whose spans the response must fill.

    Returns
    -------
    list[SubtitleLine]
        Layout lines carrying the translated text.

    Raises
    ------
    TranslationError
        If the JSON is invalid, the entry count does not match the layout, or a line
        carries no text.
    """
    data = _load_json_object(response_text, "translation")
    items = data.get("lines")
    if not isinstance(items, list):
        raise TranslationError("translation response must contain a lines list")
    if len(items) != len(layout):
        raise TranslationError(
            f"translation response has {len(items)} line(s) but the layout has {len(layout)}"
        )

    lines: list[SubtitleLine] = []
    for item, line in zip(items, layout):
        if not isinstance(item, dict) or "source" not in item or "target" not in item:
            raise TranslationError("translation line must contain source and target")
        source_text = str(item["source"]).strip()
        target_text = str(item["target"]).strip()
        if not source_text or not target_text:
            raise TranslationError("translation line must not be empty")
        lines.append(replace(line, source_text=source_text, target_text=target_text))
    return lines


def parse_review_response(response_text: str, expected_ids: Sequence[int]) -> dict[int, SentenceReview]:
    """
    Parse a batched translation review response.

    Parameters
    ----------
    response_text : str
        Raw LLM response.
    expected_ids : Sequence[int]
        Item ids submitted for review.

    Returns
    -------
    dict[int, SentenceReview]
        Review by item id.

    Raises
    ------
    TranslationError
        If the JSON is invalid, a score is out of range, or the ids cannot be matched.
    """
    data = _load_json_object(response_text, "review")
    items = data.get("reviews")
    if not isinstance(items, list):
        raise TranslationError("review response must contain a reviews list")

    reviews: dict[int, SentenceReview] = {}
    ordered: list[SentenceReview] = []
    for item in items:
        if not isinstance(item, dict) or "id" not in item or "score" not in item:
            raise TranslationError("review item must contain id and score")
        review = SentenceReview(score=_review_score(item["score"]), issue=str(item.get("issue", "")).strip())
        reviews[_required_int(item["id"], "review id")] = review
        ordered.append(review)

    if set(reviews) != set(expected_ids):
        # Local models routinely renumber batch items. Positional recovery is sound
        # because the prompt requires one entry per input id in the input order.
        if len(ordered) == len(expected_ids):
            return dict(zip(expected_ids, ordered))
        raise TranslationError("review ids do not match the submitted items")

    return reviews


def strip_markdown_json(text: str) -> str:
    """
    Remove a Markdown JSON code fence if the model returned one.

    Parameters
    ----------
    text : str
        Raw model output.

    Returns
    -------
    str
        JSON text.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    return "\n".join(lines[1:-1]).strip()


def _load_json_object(response_text: str, kind: str) -> dict:
    try:
        data = json.loads(strip_markdown_json(response_text))
    except json.JSONDecodeError as exc:
        raise TranslationError(f"{kind} response is not valid JSON") from exc
    if not isinstance(data, dict):
        raise TranslationError(f"{kind} response must be a JSON object")
    return data


def _review_score(value: object) -> int:
    score = _required_int(value, "review score")
    if not MIN_REVIEW_SCORE <= score <= MAX_REVIEW_SCORE:
        raise TranslationError(f"review score must be between {MIN_REVIEW_SCORE} and {MAX_REVIEW_SCORE}")
    return score


def _required_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise TranslationError(f"{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TranslationError(f"{field_name} must be an integer") from exc
