# Whisper Cog model

This is an implementation of [OpenAI's Whisper](https://openai.com/blog/whisper/) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i audio_path=@/path/to/audio.mp3 -i output=vtt

Or, build a Docker image:

    cog build

Or, [push it to Replicate](https://replicate.com/docs/guides/push-a-model):

    cog push r8.im/...
# cog-whisper
