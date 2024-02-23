# Critical Dream

*Visualizing critical role episodes*

**Objective**: To create a visual companion to critical role episodes that
renders scenes, characters, and environments to accompany the episode audio.

This project will involve several components that need to work together:

- **Caption data ingestion:** get captions from critical role episodes on YouTube.
- **Image data ingestion:** get image data from fandom sites, wikis, fan-made art.
- **Generative image model training:** train/fine-tune an image generation model
  that can generate character-specific images given a text prompt.
- **Prompt engineering:** use an LLM to parse text captions and write prompts
  for the image generator to create images of the scene being described within
  some time window in each episode.

## Environment Setup

```bash
conda create -n critical-dream python=3.11 -y
pip install -r requirements.txt
```

## Caption data ingestion

Use the [`youtube_transcript_api`](https://pypi.org/project/youtube-transcript-api/)
package to extract transcripts based on a list of video ids.
