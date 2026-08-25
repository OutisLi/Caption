"""Prompt builders for every LLM stage of subtitle generation."""

import json
from collections.abc import Sequence

from caption.types import Sentence, SentenceReview, TranscriptGlossary

SENTENCE_SPLIT_SYSTEM_PROMPT = (
    "You locate sentence boundaries in speech transcripts and report them by word id. "
    "You return only valid JSON."
)
GLOSSARY_SYSTEM_PROMPT = (
    "You prepare terminology references for subtitle translators. You return only valid JSON."
)
TRANSLATION_SYSTEM_PROMPT = (
    "You are a professional subtitle translator working from speech transcripts. You return only valid JSON."
)
REVIEW_SYSTEM_PROMPT = (
    "You are a strict subtitle quality reviewer. You judge translations against their source "
    "and return only valid JSON."
)
REVISION_SYSTEM_PROMPT = (
    "You are a professional subtitle translator revising a rejected translation. You return only valid JSON."
)

_LINES_JSON_SHAPE = 'Return JSON exactly like: {"lines":[{"source":"...","target":"..."}]}'


def build_sentence_split_prompt(words: Sequence[str]) -> str:
    """
    Build a sentence-boundary prompt for one batch of transcript words.

    Parameters
    ----------
    words : Sequence[str]
        Transcript words in order, exactly as recognised.

    Returns
    -------
    str
        Prompt text.
    """
    numbered = [{"id": index, "text": text} for index, text in enumerate(words, start=1)]
    return (
        "Find the sentence boundaries in the numbered transcript below.\n"
        "\n"
        "The words come from automatic speech recognition of continuous speech and carry no punctuation. You never\n"
        "rewrite them; you only report positions by word id.\n"
        "\n"
        "Report `sentence_ends`, the id of the last word of every sentence.\n"
        "1. A sentence is one complete spoken thought. Prefer several short sentences over one run-on, and end a\n"
        "   sentence where the speaker starts a new thought.\n"
        "2. Speech is full of repeated words, false starts, and abandoned clauses. They belong to the sentence they\n"
        "   occur in. Report where the thought ends, not where the wording is tidy.\n"
        "3. Every word belongs to exactly one sentence, so the ids increase and the last one is\n"
        f"   {len(words)}.\n"
        "\n"
        "Report `line_breaks`, the id of any word that could start a new subtitle line inside a sentence.\n"
        "4. A long sentence is shown across several lines and the program cuts it before one of these words.\n"
        "5. Report a word that opens a clause, a coordinating or subordinating conjunction, or a preposition that\n"
        "   opens a phrase.\n"
        "6. Never report a word that continues a compound noun, a name, a title, or a number, nor one that follows\n"
        "   a determiner, a preposition, or an auxiliary governing it.\n"
        "7. Report every acceptable id, not only the best one. The program chooses which to use and often uses\n"
        "   none, so an extra id costs nothing while a missing one cannot be recovered.\n"
        "\n"
        'Return JSON exactly like: {"sentence_ends":[14,29],"line_breaks":[8,22]}\n'
        f"Transcript:\n{json.dumps({'words': numbered}, ensure_ascii=False)}"
    )


def build_glossary_prompt(text: str, target_language: str) -> str:
    """
    Build a terminology extraction prompt for one batch of transcript text.

    Parameters
    ----------
    text : str
        Transcript text of the batch.
    target_language : str
        Translation target language.

    Returns
    -------
    str
        Prompt text.
    """
    return (
        f"Prepare the terminology reference a translator needs to render this transcript into {target_language}\n"
        "consistently. The transcript is translated one sentence at a time by workers that cannot see each other,\n"
        "so this reference is the only thing keeping their terminology aligned.\n"
        "\n"
        f"Report `topic` as one sentence in {target_language} describing what the recording is about, including the\n"
        "field and the setting. A translator reads it before every sentence, so it must resolve the ambiguities that\n"
        "a lone sentence cannot.\n"
        "\n"
        "Report `terms` for every recurring or easily mistranslated item: personal names, organisations, products,\n"
        "technical terms, and identifiers such as course or version numbers. For each, give the exact rendering\n"
        "every translator must use.\n"
        "\n"
        "Rules for renderings:\n"
        "- Personal, organisation, and product names written in Latin script stay in Latin script, unchanged.\n"
        "  Never transliterate them and never replace them with a same-sounding word.\n"
        f"- Technical terms use the established {target_language} rendering; keep the source term when none exists.\n"
        "- Numbers, course codes, and version identifiers are written as digits, and the `source` field records the\n"
        "  spelled-out form as the transcript shows it.\n"
        "- Speech recognition misspells names. When the context makes the intended name clear, record the\n"
        "  misrecognised form as `source` and the correct form as `target`.\n"
        "- Omit ordinary vocabulary that any translator renders correctly without help.\n"
        "\n"
        'Return JSON exactly like: {"topic":"...","terms":[{"source":"...","target":"..."}]}\n'
        f"Transcript:\n{text}"
    )


def build_sentence_translation_prompt(
    sentence: Sentence,
    lines: Sequence[str],
    previous_sentences: Sequence[str],
    next_sentences: Sequence[str],
    glossary: TranscriptGlossary,
    target_language: str,
) -> str:
    """
    Build a sentence translation prompt over a fixed display-line layout.

    Parameters
    ----------
    sentence : Sentence
        Sentence to translate.
    lines : Sequence[str]
        Source text of each display line, already fixed by the audio timing.
    previous_sentences : Sequence[str]
        Preceding source sentences, in reading order.
    next_sentences : Sequence[str]
        Following source sentences, in reading order.
    glossary : TranscriptGlossary
        Transcript topic and required term renderings.
    target_language : str
        Translation target language.

    Returns
    -------
    str
        Prompt text.
    """
    numbered = "\n".join(f"{number}. {line}" for number, line in enumerate(lines, start=1))
    return (
        f"Translate the CURRENT sentence into {target_language}.\n"
        "\n"
        f"{_glossary_block(glossary)}"
        f"{_context_block('CONTEXT BEFORE', previous_sentences)}"
        f"[CURRENT]\n{sentence.text}\n"
        f"{_context_block('CONTEXT AFTER', next_sentences)}"
        f"[DISPLAY LINES]\n{numbered}\n"
        "\n"
        f"The sentence is already divided into the {len(lines)} display line(s) above. Each line owns a span of the\n"
        "audio, so the division is fixed and cannot be merged, reordered, or added to.\n"
        f"Return exactly {len(lines)} entries, one per line, in the same order. Do this even when a line looks too\n"
        "short to stand alone or reads better joined to its neighbour: a missing entry leaves part of the audio\n"
        "with no subtitle. Each entry carries the `source` shown on screen and its `target` translation.\n"
        "\n"
        "Source text:\n"
        "1. Repair what speech recognition got wrong: write spoken numbers as digits, restore misheard names using\n"
        "   the glossary, and fix punctuation and casing.\n"
        "2. Remove disfluencies, false starts, and stuttered repetitions. Keep every content word.\n"
        "3. Change nothing else. Do not reword, reorder, condense, or complete an unfinished clause, and do not\n"
        "   move content from one line to another.\n"
        "\n"
        "Translation:\n"
        "4. Translate the sentence as a whole, then place each part on the line whose source it belongs to. Word\n"
        f"   order differs between the languages, so a phrase may sit on a neighbouring line when {target_language}\n"
        "   requires it, as long as the lines still read in order and no content is duplicated or lost.\n"
        "5. Use the glossary rendering for every term it lists.\n"
        f"6. Write natural {target_language} for on-screen reading. Follow target-language word order rather than\n"
        "   tracking the source word by word.\n"
        "7. Add no explanation, causal link, or fact the speaker did not state, and drop no content. When the\n"
        "   speaker abandoned a clause, leave the translation equally unfinished.\n"
        "8. Each line occupies one on-screen slot sized for its source, so keep every `target` close to the\n"
        "   information density of its own `source`. Never pad a line out and never let one swell past the rest.\n"
        "\n"
        f"{_LINES_JSON_SHAPE}"
    )


def build_review_prompt(items: Sequence[dict[str, object]], target_language: str) -> str:
    """
    Build a batched translation review prompt.

    Parameters
    ----------
    items : Sequence[dict[str, object]]
        Review items carrying ``id``, ``transcript``, ``source``, ``target``, and ``context``.
    target_language : str
        Translation target language.

    Returns
    -------
    str
        Prompt text.
    """
    return (
        f"Score each {target_language} subtitle translation.\n"
        "\n"
        "Each item gives `transcript` (the raw recognised sentence), `source` (that sentence cleaned for display),\n"
        "`target` (the translation), and `context` (neighbouring sentences, for reference only; do not score them).\n"
        "\n"
        "Scale:\n"
        "5 - accurate and idiomatic; ready to publish.\n"
        "4 - accurate; only minor stylistic roughness.\n"
        "3 - understandable but defective: awkward phrasing, wrong register, or non-standard terminology.\n"
        "2 - meaning error: content mistranslated, omitted, or invented.\n"
        "1 - unusable: wrong content, wrong language, or empty.\n"
        "\n"
        "Judge both fields:\n"
        "- `source` may only fix recognition errors, punctuation, casing, and disfluencies. Penalise it when it\n"
        "  rewords the speaker, drops content, or completes a clause the speaker abandoned.\n"
        "- `target` must carry exactly the content of `source`. Penalise invented content, dropped content, Latin\n"
        f"  script names that were transliterated or replaced, terminology that contradicts `context`, and\n"
        f"  {target_language} that no subtitle editor would ship.\n"
        "\n"
        "A translation that is unfinished because the speaker abandoned the clause is correct. Never penalise it.\n"
        "\n"
        "Set `issue` to a specific, actionable description of what to fix whenever the score is below 5, and to an\n"
        "empty string otherwise. Return one entry per input id, in the input order.\n"
        "\n"
        'Return JSON exactly like: {"reviews":[{"id":1,"score":5,"issue":""}]}\n'
        f"Input JSON:\n{json.dumps({'items': list(items)}, ensure_ascii=False)}"
    )


def build_revision_prompt(
    sentence: Sentence,
    lines: Sequence[str],
    previous_sentences: Sequence[str],
    next_sentences: Sequence[str],
    glossary: TranscriptGlossary,
    target_language: str,
    rejected_lines: Sequence[dict[str, str]],
    review: SentenceReview,
) -> str:
    """
    Build a targeted revision prompt for a rejected sentence translation.

    Parameters
    ----------
    sentence : Sentence
        Sentence to translate.
    lines : Sequence[str]
        Source text of each display line, already fixed by the audio timing.
    previous_sentences : Sequence[str]
        Preceding source sentences, in reading order.
    next_sentences : Sequence[str]
        Following source sentences, in reading order.
    glossary : TranscriptGlossary
        Transcript topic and required term renderings.
    target_language : str
        Translation target language.
    rejected_lines : Sequence[dict[str, str]]
        Display lines the reviewer rejected, each with ``source`` and ``target``.
    review : SentenceReview
        Reviewer score and defect description.

    Returns
    -------
    str
        Prompt text.
    """
    return (
        f"A reviewer scored this subtitle translation {review.score}/5 and reported:\n"
        f"{review.issue}\n"
        "\n"
        "Produce a corrected version. Repair exactly the reported defect, keep everything the reviewer did not\n"
        "fault, and do not restyle the translation for its own sake. If the report is mistaken about the source,\n"
        "return the previous attempt unchanged.\n"
        "\n"
        f"[PREVIOUS ATTEMPT]\n{json.dumps({'lines': list(rejected_lines)}, ensure_ascii=False)}\n"
        "\n"
        + build_sentence_translation_prompt(
            sentence,
            lines,
            previous_sentences,
            next_sentences,
            glossary,
            target_language,
        )
    )


def _glossary_block(glossary: TranscriptGlossary) -> str:
    topic = f"[TOPIC]\n{glossary.topic}\n" if glossary.topic else ""
    if not glossary.terms:
        return topic
    terms = "\n".join(f"- {term.source} -> {term.target}" for term in glossary.terms)
    return f"{topic}[GLOSSARY]\n{terms}\n"


def _context_block(label: str, lines: Sequence[str]) -> str:
    if not lines:
        return f"[{label}]\n(none)\n"
    rendered = "\n".join(f"- {line}" for line in lines)
    return f"[{label}]\n{rendered}\n"
