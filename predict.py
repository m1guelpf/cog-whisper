from typing import Optional

import whisper
import numpy as np
from cog import BasePredictor, Input, Path

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "iw": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
}


class Predictor(BasePredictor):
    def predict(
        self,
        audio_path: Path = Input(description="Audio file to transcribe"),
        model_name: str = Input(
            default="base",
            choices=whisper.available_models(),
            description="Name of the Whisper model to use.",
        ),
        task: str = Input(
            default="transcribe",
            choices=["transcribe", "translate"],
            description="Whether to transcribe or translate the audio.",
        ),
        language: Optional[str] = Input(
            default=None,
            choices=LANGUAGES.keys() + TO_LANGUAGE_CODE.keys(),
            description="Original language for the audio, or None for language detection.",
        ),
        temperature: float = Input(
            default=0,
            description="Temperature to use for sampling.",
        ),
        best_of: Optional[int] = Input(
            default=5,
            description="Number of candidates when sampling with non-zero temperature.",
        ),
        beam_size: Optional[int] = Input(
            default=5,
            description="Number of beams in beam search, only applicable when temperature is zero.",
        ),
        patience: float = Input(
            default=0.0,
            description="Optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (0.0) is equivalent to not using patience.",
        ),
        length_penalty: float = Input(
            default=None,
            description="Optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple lengt normalization by default.",
        ),
        suppress_tokens: str = Input(
            default="-1",
            description="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations.",
        ),
        temperature_increment_on_fallback: Optional[float] = Input(
            default=0.2,
            description="Temperature to increase when falling back when the decoding fails to meet either of the thresholds below.",
        ),
        compression_ratio_threshold: Optional[float] = Input(
            default=2.4,
            description="If the gzip compression ratio is higher than this value, treat the decoding as failed.",
        ),
        logprob_threshold: Optional[float] = Input(
            default=-1.0,
            description="If the average log probability is lower than this value, treat the decoding as failed.",
        ),
        no_caption_threshold: Optional[float] = Input(
            default=0.6,
            description="If the probability of the <|nocaptions|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence.",
        ),
    ) -> str:
        """Run a single prediction on the model"""

        if model_name.endswith(".en") and language != "en":
            raise Exception(
                f"{model_name} is an English-only model but receipted '{language}'."
            )

        if temperature_increment_on_fallback is not None:
            temp = tuple(np.arange(
                temperature, 1.0 + 1e-6, temperature_increment_on_fallback
            ))
        else:
            temp = [temperature]

        model = whisper.load_model(
            model_name, download_root="whisper-cache"
        ).to("cuda")

        result = whisper.transcribe(
            model,
            audio_path,
            verbose=True,
            temperature=temp,
            logprob_threshold=logprob_threshold,
            no_captions_threshold=no_caption_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            decode_options={
                task, language, temperature, best_of, beam_size, patience, length_penalty, suppress_tokens
            }
        )

        return result["text"]
