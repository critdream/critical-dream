<p align="center">
    <img src="static/critdream-logo.png" alt="Critical Dream Logo" width="400">
</p>
<p align="center">
    A trippy, visual companion to critical role episodes
</p>

---

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

Export secrets:

```bash
export $(grep -v '^#' secrets.txt | xargs)
```

## Caption data ingestion

Use the [`youtube_transcript_api`](https://pypi.org/project/youtube-transcript-api/)
package to extract transcripts based on a list of video ids.

```bash
python critical_dream/captions.py data/captions
```

## Compose scenes from transcripts

```bash
python critical_dream/compose_scenes.py data/captions data/scenes
```

## Get image data

To download example images of each character, do:

```bash
python critical_dream/image_data.py data/images --multi_instance_data_config config/mighty_nein_instances.yaml
```

## Dreambooth fine-tuning

Export secrets:

```bash
export HF_HUB_TOKEN="..."
export WANDB_API_KEY="..."
```

You can fine-tune various models and fine-tuning options:


<details>
<summary>Stable Diffusion XL Base 1.0 LoRA fine-tuning</summary>

```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_PATH="stabilityai/sdxl-vae"
export OUTPUT_DIR="models/model_sd1xl_lora_critdream"
export HUB_MODEL_ID="cosmicBboy/stable-diffusion-xl-base-1.0-lora-dreambooth-critdream"

accelerate launch critical_dream/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --multi_instance_data_config=config/mighty_nein_instances.yaml \
  --multi_instance_subset=fjord \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --with_prior_preservation \
  --output_dir=$OUTPUT_DIR \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=10 \
  --max_train_steps=10 \
  --validation_prompt="a picture of [critrole-fjord], a half-orc with a top hat" \
  --validation_epochs=25 \
  --checkpointing_steps=500 \
  --hub_model_id=$HUB_MODEL_ID \
  --seed="0" \
  --push_to_hub
```
</details>


<details>
<summary>Stable Diffusion 1.4 full fine-tuning</summary>

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="data/fjord"
export CLASS_DIR="data/half_orc"
export OUTPUT_DIR="models/model_sd1_fjord"
export HUB_MODEL_ID="cosmicBboy/stable-diffusion-v1-4-dreambooth-critdream-fjord"

accelerate launch critical_dream/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a picture of [critrole-fjord], a half-orc warlock" \
  --class_prompt="a picture of a half-orc warlock" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=1000 \
  --validation_prompt="a picture of [critrole-fjord], a half-orc with a top hat" \
  --validation_steps=250 \
  --checkpointing_steps=1000 \
  --hub_model_id=$HUB_MODEL_ID \
  --push_to_hub \
  --report_to="wandb"
```
</details>


<details>
<summary>Stable Diffusion 1.4 LoRA fine-tuning</summary>

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="data/fjord"
export CLASS_DIR="data/half_orc"
export OUTPUT_DIR="models/model_sd1_lora_fjord"
export HUB_MODEL_ID="cosmicBboy/stable-diffusion-v1-4-lora-dreambooth-critdream-fjord"

accelerate launch critical_dream/train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a picture of [critrole-fjord], a half-orc" \
  --class_prompt="a picture of a half-orc" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=1500 \
  --validation_prompt="a picture of [critrole-fjord], a half-orc with a top hat" \
  --validation_epochs=25 \
  --checkpointing_steps=1000 \
  --hub_model_id=$HUB_MODEL_ID \
  --push_to_hub \
  --report_to="wandb"
```
</details>


<details>
<summary>Stable Diffusion 2 full fine-tuning</summary>

```bash
export MODEL_NAME="stabilityai/stable-diffusion-2"
export INSTANCE_DIR="data/fjord"
export CLASS_DIR="data/half_orc"
export OUTPUT_DIR="models/model_sd2_lora_fjord"
export HUB_MODEL_ID="cosmicBboy/stable-diffusion-2-dreambooth-critdream-fjord"

accelerate launch critical_dream/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a picture of [critrole-fjord], a half-orc" \
  --class_prompt="a picture of a half-orc" \
  --resolution=1024 \
  --train_batch_size=1 \
  --report_to="wandb" \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=100 \
  --validation_prompt="a picture of [critrole-fjord], a half-orc with a top hat" \
  --validation_steps=100 \
  --checkpointing_steps=100 \
  --hub_model_id=$HUB_MODEL_ID \
  --seed="0" \
  --push_to_hub \
  --report_to="wandb"
```
</details>



<details>
<summary>Stable Diffusion 2 LoRA fine-tuning</summary>

```bash
export MODEL_NAME="stabilityai/stable-diffusion-2"
export INSTANCE_DIR="data/fjord"
export CLASS_DIR="data/half_orc"
export OUTPUT_DIR="models/model_sd2_lora_fjord"
export HUB_MODEL_ID="cosmicBboy/stable-diffusion-2-lora-dreambooth-critdream-fjord"

accelerate launch critical_dream/train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a picture of [critrole-fjord], a half-orc" \
  --class_prompt="a picture of a half-orc" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=2000 \
  --validation_prompt="a picture of [critrole-fjord], a half-orc with a top hat" \
  --validation_epochs=25 \
  --checkpointing_steps=250 \
  --hub_model_id=$HUB_MODEL_ID \
  --push_to_hub \
  --report_to="wandb"
```
</details>
