import whisper
from typing import Optional
from whisper.tokenizer import LANGUAGES
from cog import BasePredictor, Input, Path, BaseModel


class ModelOutput(BaseModel):
    language: str
    text: Optional[str]
    subtitles: Optional[str]


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
        output: str = Input(
            default="text",
            choices=["text", "vtt"],
            description="Whether to return raw text or a VTT file.",
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""

        model = whisper.load_model(
            model_name, download_root="whisper-cache", device="cuda"
        )

        result = model.transcribe(
            str(audio_path),
            verbose=True,
            task=task
        )

        if (output == "text"):
            return ModelOutput(
                text=result["text"],
                language=LANGUAGES[result["language"]],
            )

        vtt = "WEBVTT\n"
        for segment in result['segments']:
            vtt += f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            vtt += f"{segment['text'].replace('-->', '->')}\n"

        return ModelOutput(
            subtitles=vtt,
            language=LANGUAGES[result["language"]],
        )


def format_timestamp(seconds: float):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    return (f"{hours}:" if hours > 0 else "") + f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
