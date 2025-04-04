# Qwen2.5-Omni-7B

A multimodal model by Alibaba capable of processing text, image, audio, and video inputs.

## Model Description

Qwen2.5-Omni-7B is a powerful multimodal AI model developed by the Qwen Team at Alibaba Group. It can:

- Process and understand images, audio, and video
- Generate text responses based on multimodal inputs
- Produce speech output in different voices

## Input Types

- **Text**: Ask questions or provide prompts as regular text
- **Images**: Upload images for visual understanding
- **Audio**: Submit audio files for analysis
- **Video**: Provide video content for the model to process

## Output

The model returns:
- Text response
- Optional audio response (when `generate_audio` is enabled)

## Examples

### Text + Image
```
model.predict(
    prompt="What's in this image?",
    image="path/to/image.jpg"
)
```

### Video Analysis
```
model.predict(
    prompt="Describe what's happening in this video",
    video="path/to/video.mp4"
)
```

### Audio Processing
```
model.predict(
    prompt="Transcribe and analyze this audio",
    audio="path/to/audio.mp3"
)
```

## About

This model is based on Qwen2.5-Omni-7B and is deployed as a [Cog](https://github.com/replicate/cog) model on [Replicate](https://replicate.com). 