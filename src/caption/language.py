"""Resolve language labels used in filenames and Matroska track metadata."""

from dataclasses import dataclass


class LanguageError(ValueError):
    """Raised when a language label cannot be interpreted."""


@dataclass(frozen=True)
class Language:
    """A language as needed by output filenames and Matroska tracks.

    Parameters
    ----------
    tag : str
        Three-letter ISO 639-2/B code written into MKV metadata.
    code : str
        Short code used in subtitle filenames, usually ISO 639-1.
    title : str
        Human-readable name written into the track title.
    """

    tag: str
    code: str
    title: str


# Aliases → (ISO 639-2/B, filename code, English title). Keys are lowercase and
# may be ISO 639-1, 639-2, or a name that ASR and configs use.
_LANGUAGE_ALIASES: dict[str, tuple[str, str, str]] = {
    "ar": ("ara", "ar", "Arabic"),
    "ara": ("ara", "ar", "Arabic"),
    "arabic": ("ara", "ar", "Arabic"),
    "cs": ("cze", "cs", "Czech"),
    "cze": ("cze", "cs", "Czech"),
    "ces": ("cze", "cs", "Czech"),
    "czech": ("cze", "cs", "Czech"),
    "de": ("ger", "de", "German"),
    "ger": ("ger", "de", "German"),
    "deu": ("ger", "de", "German"),
    "german": ("ger", "de", "German"),
    "en": ("eng", "en", "English"),
    "eng": ("eng", "en", "English"),
    "english": ("eng", "en", "English"),
    "es": ("spa", "es", "Spanish"),
    "spa": ("spa", "es", "Spanish"),
    "spanish": ("spa", "es", "Spanish"),
    "fr": ("fre", "fr", "French"),
    "fre": ("fre", "fr", "French"),
    "fra": ("fre", "fr", "French"),
    "french": ("fre", "fr", "French"),
    "hi": ("hin", "hi", "Hindi"),
    "hin": ("hin", "hi", "Hindi"),
    "hindi": ("hin", "hi", "Hindi"),
    "it": ("ita", "it", "Italian"),
    "ita": ("ita", "it", "Italian"),
    "italian": ("ita", "it", "Italian"),
    "ja": ("jpn", "ja", "Japanese"),
    "jpn": ("jpn", "ja", "Japanese"),
    "japanese": ("jpn", "ja", "Japanese"),
    "日本語": ("jpn", "ja", "Japanese"),
    "ko": ("kor", "ko", "Korean"),
    "kor": ("kor", "ko", "Korean"),
    "korean": ("kor", "ko", "Korean"),
    "한국어": ("kor", "ko", "Korean"),
    "nl": ("dut", "nl", "Dutch"),
    "dut": ("dut", "nl", "Dutch"),
    "nld": ("dut", "nl", "Dutch"),
    "dutch": ("dut", "nl", "Dutch"),
    "pl": ("pol", "pl", "Polish"),
    "pol": ("pol", "pl", "Polish"),
    "polish": ("pol", "pl", "Polish"),
    "pt": ("por", "pt", "Portuguese"),
    "por": ("por", "pt", "Portuguese"),
    "portuguese": ("por", "pt", "Portuguese"),
    "ru": ("rus", "ru", "Russian"),
    "rus": ("rus", "ru", "Russian"),
    "russian": ("rus", "ru", "Russian"),
    "sv": ("swe", "sv", "Swedish"),
    "swe": ("swe", "sv", "Swedish"),
    "swedish": ("swe", "sv", "Swedish"),
    "th": ("tha", "th", "Thai"),
    "tha": ("tha", "th", "Thai"),
    "thai": ("tha", "th", "Thai"),
    "tr": ("tur", "tr", "Turkish"),
    "tur": ("tur", "tr", "Turkish"),
    "turkish": ("tur", "tr", "Turkish"),
    "uk": ("ukr", "uk", "Ukrainian"),
    "ukr": ("ukr", "uk", "Ukrainian"),
    "ukrainian": ("ukr", "uk", "Ukrainian"),
    "vi": ("vie", "vi", "Vietnamese"),
    "vie": ("vie", "vi", "Vietnamese"),
    "vietnamese": ("vie", "vi", "Vietnamese"),
    "zh": ("chi", "zh", "Chinese"),
    "chi": ("chi", "zh", "Chinese"),
    "zho": ("chi", "zh", "Chinese"),
    "cmn": ("chi", "zh", "Chinese"),
    "chinese": ("chi", "zh", "Chinese"),
    "中文": ("chi", "zh", "Chinese"),
    "zh-cn": ("chi", "zh", "Chinese"),
    "zh-hans": ("chi", "zh", "Chinese"),
    "zh-sg": ("chi", "zh", "Chinese"),
    "zh-tw": ("chi", "zh", "Chinese"),
    "zh-hk": ("chi", "zh", "Chinese"),
    "zh-hant": ("chi", "zh", "Chinese"),
    "yue": ("yue", "yue", "Cantonese"),
    "cantonese": ("yue", "yue", "Cantonese"),
    "mul": ("mul", "mul", "Bilingual"),
    "und": ("und", "und", "Unknown"),
}


def resolve_language(value: str) -> Language:
    """
    Map a user, ASR, or ISO language label onto filename and Matroska codes.

    Parameters
    ----------
    value : str
        Language code or name. Empty becomes ``und``.

    Returns
    -------
    Language
        Matroska tag, filename code, and display title.

    Raises
    ------
    LanguageError
        If the label cannot be interpreted as a language.
    """
    raw = value.strip()
    if not raw:
        return Language("und", "und", "Unknown")
    key = raw.lower().replace("_", "-")
    matched = _LANGUAGE_ALIASES.get(key) or _LANGUAGE_ALIASES.get(key.split("-", 1)[0])
    if matched is not None:
        tag, code, title = matched
        return Language(tag, code, title)
    if len(key) == 3 and key.isalpha():
        return Language(key, key, raw)
    raise LanguageError(f"unrecognized language {value!r}; use an ISO 639 code such as zh or en")


def filename_language_code(value: str) -> str:
    """
    Return the short language code used in a subtitle filename.

    Parameters
    ----------
    value : str
        Language code or name.

    Returns
    -------
    str
        Filename suffix such as ``zh`` or ``en``.
    """
    return resolve_language(value).code
