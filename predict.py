import whisper
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.models = {}
        for model in ["tiny", "base", "small", "medium", "large"]:
            self.models[model] = whisper.load_model(
                model, download_root="whisper-cache"
            )

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
    ) -> str:
        """Run a single prediction on the model"""

        model = self.models[model_name].to("cuda")

        result = model.transcribe(
            model,
            audio_path,
            verbose=True,
            decode_options={task}
        )

        if (output == "text"):
            return result['text']

        vtt = "WEBVTT\n"
        for segment in result['segments']:
            vtt += f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            vtt += f"{segment['text'].replace('-->', '->')}\n"

        return vtt


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
