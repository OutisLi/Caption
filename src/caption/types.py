"""Shared data types for caption generation."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MediaJob:
    """Describe one media file and its output location.

    Parameters
    ----------
    input_path : Path
        Source audio or video file.
    output_dir : Path
        Base directory where generated subtitle files are written.
    stem : str
        Output filename stem.
    relative_output_dir : Path
        Input-root-relative directory preserved under each output stage.
    """

    input_path: Path
    output_dir: Path
    stem: str
    relative_output_dir: Path = Path()


@dataclass(frozen=True)
class OutputPaths:
    """Store generated subtitle output paths.

    Parameters
    ----------
    asr_srt : Path
        ASR-only source-language subtitle path.
    asr_txt : Path
        ASR-only source-language plain text path.
    source_srt : Path
        Source-language subtitle path.
    source_txt : Path
        Source-language plain text path.
    target_srt : Path
        Target-language subtitle path.
    target_txt : Path
        Target-language plain text path.
    bilingual_srt : Path
        Bilingual subtitle path.
    raw_source_srt : Path | None
        Pre-review source-language subtitle path.
    raw_source_txt : Path | None
        Pre-review source-language plain text path.
    raw_target_srt : Path | None
        Pre-review target-language subtitle path.
    raw_target_txt : Path | None
        Pre-review target-language plain text path.
    raw_bilingual_srt : Path | None
        Pre-review bilingual subtitle path.
    asr_json : Path | None
        Optional ASR debug JSON path.
    written_paths : tuple[Path, ...]
        Files written by the current run.
    """

    asr_srt: Path
    asr_txt: Path
    source_srt: Path
    source_txt: Path
    target_srt: Path
    target_txt: Path
    bilingual_srt: Path
    raw_source_srt: Path | None = None
    raw_source_txt: Path | None = None
    raw_target_srt: Path | None = None
    raw_target_txt: Path | None = None
    raw_bilingual_srt: Path | None = None
    asr_json: Path | None = None
    written_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class SamplingParams:
    """Sampling parameters for one generation mode.

    Models publish different recommendations for reasoning and non-reasoning generation,
    so each mode carries its own set and the active one follows the per-task reasoning
    switch.

    Parameters
    ----------
    temperature : float
        Softmax temperature.
    top_p : float
        Nucleus sampling mass.
    top_k : int
        Candidate cutoff by rank. Zero disables it.
    min_p : float
        Minimum token probability relative to the most likely token.
    presence_penalty : float
        Penalty applied to tokens that already occur in the output.
    repetition_penalty : float
        Multiplicative penalty applied to repeated tokens.
    """

    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float


@dataclass(frozen=True)
class WordSpan:
    """Represent one recognized word or character with timestamps.

    Parameters
    ----------
    text : str
        Recognized text.
    start : float
        Start time in seconds.
    end : float
        End time in seconds.
    """

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class SubtitleCue:
    """Represent one subtitle cue.

    Parameters
    ----------
    index : int
        One-based subtitle cue index.
    start : float
        Start time in seconds.
    end : float
        End time in seconds.
    source_text : str
        Source-language text.
    target_text : str
        Target-language text.
    """

    index: int
    start: float
    end: float
    source_text: str
    target_text: str = ""


@dataclass(frozen=True)
class Sentence:
    """Represent one semantically complete unit of speech.

    A sentence is the unit of translation. Its text carries restored punctuation and
    casing but the same words as the transcript, so it maps back onto ``words`` exactly.

    Parameters
    ----------
    index : int
        One-based sentence index within the transcript.
    text : str
        Punctuation-restored source text.
    words : tuple[WordSpan, ...]
        Timestamped words the sentence covers, in order.
    breaks : frozenset[int]
        Sentence-relative indices of words that may start a new display line. Advisory:
        the layout stage decides how many breaks to use, and may use none of them.
    """

    index: int
    text: str
    words: tuple[WordSpan, ...]
    breaks: frozenset[int] = frozenset()

    @property
    def start(self) -> float:
        """Return the sentence start time in seconds."""
        return self.words[0].start

    @property
    def end(self) -> float:
        """Return the sentence end time in seconds."""
        return self.words[-1].end

    @property
    def duration(self) -> float:
        """Return the sentence duration in seconds."""
        return self.end - self.start


@dataclass(frozen=True)
class SubtitleLine:
    """Represent one display line with its own time span.

    The program derives a line from word timestamps before any translation happens, and
    the span never moves afterwards. LLM stages only replace the text a line carries.

    Parameters
    ----------
    start : float
        Start time in seconds.
    end : float
        End time in seconds.
    source_text : str
        Source-language text for this line.
    target_text : str
        Target-language text for the same span.
    """

    start: float
    end: float
    source_text: str
    target_text: str = ""


@dataclass(frozen=True)
class SentenceLayout:
    """Pair a sentence with the display lines it occupies.

    Parameters
    ----------
    sentence : Sentence
        Source sentence with timestamps.
    lines : tuple[SubtitleLine, ...]
        Display lines covering the sentence in order.
    """

    sentence: Sentence
    lines: tuple[SubtitleLine, ...]

    @property
    def source_text(self) -> str:
        """Return the display source text of the whole sentence."""
        return " ".join(line.source_text for line in self.lines if line.source_text)

    @property
    def target_text(self) -> str:
        """Return the translated text of the whole sentence."""
        return "".join(line.target_text for line in self.lines)


@dataclass(frozen=True)
class GlossaryTerm:
    """Bind one source term to its required translation.

    Parameters
    ----------
    source : str
        Term as it appears in the transcript.
    target : str
        Rendering every translation of this transcript must use.
    """

    source: str
    target: str


@dataclass(frozen=True)
class TranscriptGlossary:
    """Carry the transcript-wide context that keeps parallel translations consistent.

    Parameters
    ----------
    topic : str
        One-sentence description of the recording.
    terms : tuple[GlossaryTerm, ...]
        Required renderings for recurring or easily mistranslated terms.
    """

    topic: str
    terms: tuple[GlossaryTerm, ...] = ()


@dataclass(frozen=True)
class SentenceReview:
    """Represent one quality judgement of a translated sentence.

    Parameters
    ----------
    score : int
        Quality score from 1 (unusable) to 5 (ready to publish).
    issue : str
        Concrete defect description. Empty when the translation needs no revision.
    """

    score: int
    issue: str


@dataclass(frozen=True)
class AsrResult:
    """Store normalized ASR output.

    Parameters
    ----------
    text : str
        Full recognized text.
    language : str
        Detected or forced language name.
    words : list[WordSpan]
        Word or character timestamps.
    chunks : list[dict]
        Optional backend chunk metadata.
    """

    text: str
    language: str
    words: list[WordSpan]
    chunks: list[dict]


@dataclass(frozen=True)
class CaptionConfig:
    """Configure caption generation.

    Parameters
    ----------
    target_language : str | None
        Translation target language. None means source-only optimization.
    source_language : str | None
        Optional forced source language for ASR.
    translation_position : str
        Where translated text appears in bilingual subtitles.
    max_chars_per_cue : int
        Maximum source-language characters per cue in the ASR-only output.
    max_seconds_per_cue : float
        Maximum cue duration in seconds.
    plain_text : bool
        Whether to stop after raw source SRT/TXT generation.
    write_text : bool
        Whether to write TXT sidecar files outside plain-text mode.
    review : bool
        Whether translations are refined through review rounds after the first pass.
    """

    target_language: str | None
    source_language: str | None = None
    translation_position: str = "bottom"
    max_chars_per_cue: int = 60
    max_seconds_per_cue: float = 6.0
    plain_text: bool = False
    write_text: bool = False
    review: bool = False
