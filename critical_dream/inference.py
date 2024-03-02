from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline
import torch

lora_model_id = "cosmicBboy/stable-diffusion-xl-base-1.0-lora-dreambooth-critdream-v0"
card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
pipe = pipe.to("mps")
pipe.load_lora_weights(lora_model_id)
image = pipe("a picture of [critrole-fjord], a male half-orc warlock", num_inference_steps=25).images[0]
image.save("fjord.png")
