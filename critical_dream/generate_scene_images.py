"""Create images"""

import json
import re

from argparse import ArgumentParser
from pathlib import Path
from typing import Iterator

from compel import Compel, ReturnedEmbeddingsType
from datasets import load_dataset
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from huggingface_hub.repocard import RepoCard

import torch


DEFAULT_LORA_MODEL_ID = "cosmicBboy/stable-diffusion-xl-base-1.0-lora-dreambooth-critdream-v0.4"
DEFAULT_DATASET_ID = "cosmicBboy/critical-dream-scenes-mighty-nein"
REFINER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"

VETH_EPISODE = 97
MOLLYMAUK_EPISODE = 26


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def get_dtype():
    return (
        torch.float16
        if torch.cuda.is_available() or torch.backends.mps.is_available()
        else torch.float32
    )


def load_pipeline(lora_model_id: str):
    card = RepoCard.load(lora_model_id)
    base_model_id = card.data.to_dict()["base_model"]

    pipe = DiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=get_dtype(),
    )
    pipe.load_lora_weights(lora_model_id)

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    if torch.backends.mps.is_available():
        pipe = pipe.to("mps")

    return pipe


def load_refiner():
    # dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        REFINER_MODEL,
        torch_dtype=get_dtype(),
        use_safetensors=True,
        variant="fp16",
    )
    if torch.cuda.is_available():
        refiner.to("cuda")
    if torch.backends.mps.is_available():
        refiner = refiner.to("mps")

    return refiner


SINGLE_CHARACTER_MAP = {
    "marisha": "beau",
    "laura": "jester",
    "travis": "fjord",
    "liam": "caleb",
    "ashley": "yasha",
}

PLAYER_CHARACTERS = frozenset(
    [
        *SINGLE_CHARACTER_MAP.values(),
        "nott",
        "veth",
        "mollymauk",
        "caduceus",
    ]
)

SPECIAL_CHARACTERS = {
    "fjord": "[critrole-fjord], a male half-orc warlock,",
    "beau": "[critrole-beau], a female human monk,",
    "jester": "[critrole-jester], a female tiefling cleric,",
    "caleb": "[critrole-caleb], a male human wizard,",
    "caduceus": "[critrole-caduceus], a male firbolg cleric,",
    "nott": "[critrole-nott], a female goblin rogue",
    "veth": "[critrole-veth], a female halfling rogue/wizard,",
    "yasha": "[critrole-yasha], a female aasimar barbarian,",
    "mollymauk": "[critrole-mollymauk], a male tiefling blood hunter",
}

PROMPT_PREFIX = "fantasy art style, professional, high quality, highly detailed, sharp focus"


def fix_character_name(
    character: str,
    description: str,episode_name: str,
) -> tuple[str, str, str]:
    char = character.lower().strip()

    if char == "matt":
        return "dungeon-master", "[dungeon-master]", description
    
    correct_char = (
        char
        if char in PLAYER_CHARACTERS
        else SINGLE_CHARACTER_MAP.get(char, None)
    )

    if correct_char is None:
        episode_num = int(episode_name.split("e")[-1])
        if char == "sam":
            correct_char = "veth" if episode_num > VETH_EPISODE else "nott"
        elif char == "taliesin":
            correct_char = "caduceus" if episode_num > MOLLYMAUK_EPISODE else "mollymauk"
        else:
            correct_char = char
    
    special_character = SPECIAL_CHARACTERS.get(correct_char, correct_char)

    patt = (
        re.compile(re.escape(char), re.IGNORECASE)
        if char != correct_char
        else re.compile(re.escape(correct_char), re.IGNORECASE)
    )
    description = re.sub(patt, special_character, description)
    return correct_char, special_character, description


def create_prompt(description: str) -> str:
    return f"{PROMPT_PREFIX}. {description}"


def process_scene(scene: dict):
    fixed_character, special_character, description = fix_character_name(
        scene["character"],
        scene["scene_description"],
        scene["episode_name"],
    )
    scene["fixed_character"] = fixed_character
    scene["special_character"] = special_character
    scene["prompt"] = create_prompt(description)
    return scene


def load_scene_dataset(dataset_id: str):
    dataset = load_dataset(dataset_id)["train"]
    return dataset.map(process_scene, batched=False)


def get_scene_dir(output_dir: Path, scene: dict, scene_num: int) -> Path:
    return output_dir / scene["episode_name"] / f"scene_{scene_num:04}"


def generate_scene_images(
    pipe,
    dataset,
    output_dir: Path,
    num_images_per_prompt: int = 8,
    num_inference_steps: int = 30,
    manual_seed: int = 0,
) -> Iterator[tuple[dict, Path]]:
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        None
    )
    generator = torch.Generator(device).manual_seed(manual_seed)

    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True]
    )

    for scene_num, scene in enumerate(dataset):
        scene_dir = get_scene_dir(output_dir, scene, scene_num)
        if scene_dir.exists():
            print(f"Scene images {scene_dir} exists. Skipping.")
            continue

        conditioning, pooled = compel(scene["prompt"])
        images = pipe(
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
            generator=generator,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
        ).images
        scene["images"] = images
        yield scene, scene_dir


def main(
    lora_model_id: str,
    dataset_id: str,
    output_dir: Path,
    num_images_per_prompt: int,
    num_inference_steps: int,
):
    pipe = load_pipeline(lora_model_id)
    dataset = load_scene_dataset(dataset_id)
    for scene, scene_dir in generate_scene_images(
        pipe, dataset, output_dir, num_images_per_prompt, num_inference_steps,
    ):
        scene_dir.mkdir(exist_ok=True, parents=True)
        images = scene.pop("images")

        with (scene_dir / "metadata.json").open("w") as f:
            json.dump(scene, f)

        for image_num, image in enumerate(images):
            image.save(scene_dir / f"image_{image_num:02}.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--lora_model_id",
        type=str,
        default=DEFAULT_LORA_MODEL_ID,
        help="Model ID of LoRA model.",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default=DEFAULT_DATASET_ID,
        help="Huggingface dataset ID.",
    )
    parser.add_argument("--output_dir", type=Path, help="Path to save images.")
    parser.add_argument("--num_images_per_prompt", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    args = parser.parse_args()
    main(
        args.lora_model_id,
        args.dataset_id,
        Path(args.output_dir),
        args.num_images_per_prompt,
        args.num_inference_steps,
    )
