import whisper
from whisper.tokenizer import LANGUAGES
from cog import BasePredictor, Input, Path, BaseModel


class ModelOutput(BaseModel):
    text: str
    language: str
    subtitles: str


class Predictor(BasePredictor):
    def predict(
        self,
        audio_path: Path = Input(description="Audio file to transcribe"),
        model_name: str = Input(
            default="base",
            choices=whisper.available_models(),
            description="Name of the Whisper model to use.",
        ),
        format: str = Input(
            default="vtt",
            choices=["srt", "vtt"],
            description="Whether to generate subtitles on the SRT or VTT format.",
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""

        model = whisper.load_model(
            model_name, download_root="whisper-cache", device="cuda"
        )

        result = model.transcribe(
            str(audio_path),
            verbose=True,
        )

        if (format == 'vtt'):
            subtitles = generate_vtt(result)
        else:
            subtitles = generate_srt(result)

        return ModelOutput(
            text=result["text"],
            subtitles=subtitles,
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


def generate_vtt(result: dict):
    vtt = "WEBVTT\n"
    for segment in result['segments']:
        vtt += f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
        vtt += f"{segment['text'].replace('-->', '->')}\n"
    return vtt


def generate_srt(result: dict):
    srt = ""
    for i, segment in enumerate(result['segments'], start=1):
        srt += f"{i}\n"
        srt += f"{format_timestamp(segment['start'], always_include_hours=True)} --> {format_timestamp(segment['end'], always_include_hours=True)}\n"
        srt += f"{segment['text'].strip().replace('-->', '->')}\n"
    return srt
