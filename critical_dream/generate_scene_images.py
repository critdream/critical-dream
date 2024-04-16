"""Create images"""

import json
import logging

from argparse import ArgumentParser
from pathlib import Path
from typing import Iterator

from compel import Compel, ReturnedEmbeddingsType
from datasets import load_dataset
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub.repocard import RepoCard
from packaging import version

import torch


logger = logging.getLogger(__name__)


DEFAULT_LORA_MODEL_ID = "cosmicBboy/stable-diffusion-xl-base-1.0-lora-dreambooth-critdream-v0.4"
DEFAULT_DATASET_ID = "cosmicBboy/critical-dream-scenes-mighty-nein"
DEFAULT_REFINER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"

VETH_EPISODE = 97
MOLLYMAUK_EPISODE = 26

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

CHARACTER_TOKENS = {
    "fjord": "[critrole-fjord]",
    "beau": "[critrole-beau]",
    "jester": "[critrole-jester]",
    "caleb": "[critrole-caleb]",
    "caduceus": "[critrole-caduceus]",
    "nott": "[critrole-nott]",
    "veth": "[critrole-veth]",
    "yasha": "[critrole-yasha]",
    "mollymauk": "[critrole-mollymauk]",
    "essek": "[critrole-essek]",
}

ADDITIONAL_PROMPTS = {
    "fjord": "a male half-orc warlock with black hair",
    "beau": "a female human monk with round ears and no tattoos",
    "jester": "female tiefling cleric with blue skin, and blue hair",
    "caleb": "male human wizard. round ears",
    "caduceus": "a male firbolg cleric",
    "nott": "a female goblin rogue",
    "veth": "a female halfling rogue/wizard",
    "yasha": "a female aasimar barbarian with black hair",
    "mollymauk": "a male tiefling blood hunter with purple skin",
    "essek": "a male drow wizard.",
}

ADDITIONAL_NEGATIVE_PROMPTS = {
    "fjord": "",
    "beau": "",
    "jester": "",
    "caleb": "pointy ears. earrings.",
    "caduceus": "",
    "nott": "",
    "veth": "",
    "yasha": "black hair. wings.",
    "mollymauk": "",
}

PROMPT_AUGMENTATION = (
    "D&D fantasy art style. high quality. sharp focus. artstation. professional"
)

DEFAULT_NEGATIVE_PROMPT = (
    "nsfw, blank background, plain background, letters, words, copy, watermark, "
    "ugly, distorted, deformed, duplicate characters, repeated patterns, uncanny valley, "
    "distorted facial features, distorted hands, extra limbs, distorted fingers, "
    "(concept art)1.5, "
    "worst quality, low quality, normal quality, low resolution, "
    "worst resolution, normal resolution, collage, bad anatomy of fingers, "
    "error hands, error fingers"
)


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


def load_refiner(refiner_model_id: str):
    # dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        refiner_model_id,
        torch_dtype=get_dtype(),
        use_safetensors=True,
        variant="fp16",
    )
    if torch.cuda.is_available():
        refiner.to("cuda")
    if torch.backends.mps.is_available():
        refiner = refiner.to("mps")

    return refiner


def add_prompts(scene: dict) -> dict:

    character = scene["character"]
    description = scene["scene_description"]
    episode_name = scene["episode_name"]

    char = character.lower().strip()

    if " as " in char:
        _, char = char.split(" as ")

    if char == "matt":
        scene.update({
            "correct_char": "dm-matt-mercer",
            "character_tokens": "[dm-matt-mercer]",
            "prompt": f"[dm-matt-mercer] wearing a hooded cloak. {description}",
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        })
        return scene
    
    split_char = char.split(" ")
    if char in PLAYER_CHARACTERS:
        correct_char = char
    elif len(split_char) > 1 and split_char[0] in PLAYER_CHARACTERS:
        correct_char = split_char[0]
    elif len(split_char) > 1 and split_char[0] in SINGLE_CHARACTER_MAP:
        correct_char = SINGLE_CHARACTER_MAP[split_char[0]]
    else:
        correct_char = SINGLE_CHARACTER_MAP.get(char, None)

    if correct_char is None:
        episode_num = int(episode_name.split("e")[-1])
        if char == "sam":
            correct_char = "veth" if episode_num > VETH_EPISODE else "nott"
        elif char == "veth" and episode_num < VETH_EPISODE:
            correct_char = "nott"
        elif char == "taliesin":
            correct_char = "caduceus" if episode_num > MOLLYMAUK_EPISODE else "mollymauk"
        elif char == "beauregard":
            correct_char = "beau"
        else:
            correct_char = char
    
    character_tokens = CHARACTER_TOKENS.get(correct_char, correct_char)
    addtl_prompts = ADDITIONAL_PROMPTS.get(correct_char, "")
    addtl_neg_prompts = ADDITIONAL_NEGATIVE_PROMPTS.get(correct_char, "")

    # format the prompt
    if correct_char in PLAYER_CHARACTERS:
        full_character_desc = character_tokens
        if addtl_prompts:
            full_character_desc = f"{full_character_desc}, {addtl_prompts}"
        prompt = (
            f"{full_character_desc}, "
            f"{scene['action']}, "
            f"{scene['poses']}. "
            f"({scene['background']} background, fantasy world)++++ ."
            f"{PROMPT_AUGMENTATION}"
        )
    else:
        prompt = f"{description}, {PROMPT_AUGMENTATION}"

    # format the negative prompt
    if addtl_neg_prompts:
        negative_prompt = f"{addtl_neg_prompts}, {DEFAULT_NEGATIVE_PROMPT}"
    else:
        negative_prompt = DEFAULT_NEGATIVE_PROMPT

    # TODO: use the 'object' field to create another prompt for the thing being
    # interacted with in the scene.
    scene.update({
        "correct_char": correct_char,
        "character_tokens": character_tokens,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
    })

    return scene


def process_scene(scene: dict):
    scene = add_prompts(scene)
    scene.pop("turns")
    return scene


def load_scene_dataset(dataset_id: str):
    dataset = load_dataset(dataset_id)["train"]
    return dataset.map(process_scene, batched=False)


def get_scene_dir(output_dir: Path, scene: dict) -> Path:
    return output_dir / scene["episode_name"]


def generate_scene_images(
    pipe,
    refiner,
    dataset,
    output_dir: Path,
    num_images_per_prompt: int = 8,
    num_batches_per_prompt: int = 1,
    num_inference_steps: int = 30,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    manual_seed: int = 0,
) -> Iterator[tuple[dict, Path]]:
    generator = torch.Generator(get_device()).manual_seed(manual_seed)

    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True]
    )

    for i, scene in enumerate(dataset):
        scene_dir = get_scene_dir(output_dir, scene)
        if (scene_dir / f"scene_{i:03}_metadata.json").exists():
            print(f"Scene images {scene_dir} exists. Skipping.")
            continue

        conditioning, pooled = compel(scene["prompt"])
        negative_conditioning, negative_pooled = compel(negative_prompt)

        conditioning, negative_conditioning = compel.pad_conditioning_tensors_to_same_length(
            [conditioning, negative_conditioning]
        )
        conditioning, pooled = compel(scene["prompt"])

        images = []
        for _ in range(num_batches_per_prompt):
            latents = pipe(
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=negative_conditioning,
                negative_pooled_prompt_embeds=negative_pooled,
                output_type="latent",
                generator=generator,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
            ).images

            for latent in latents:
                image = refiner(
                    prompt=scene["prompt"],
                    image=latent,
                    generator=generator,
                ).images[0]
                images.append(image)
        scene["images"] = images
        yield scene, scene_dir


def main(
    lora_model_id: str,
    refiner_model_id: str,
    dataset_id: str,
    output_dir: Path,
    num_images_per_prompt: int,
    num_batches_per_prompt: int,
    num_inference_steps: int,
    negative_prompt: str,
    enable_xformers_memory_efficient_attention: bool = False,
    debug: bool = False,
):
    dataset = load_scene_dataset(dataset_id)
    pipe = load_pipeline(lora_model_id)
    refiner = load_refiner(refiner_model_id)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, "
                    "please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            pipe.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if debug:
        for scene in dataset:
            print(f"prompt: {scene}")
        return

    for i, (scene, scene_dir) in enumerate(
        generate_scene_images(
            pipe,
            refiner,
            dataset,
            output_dir,
            num_images_per_prompt,
            num_batches_per_prompt,
            num_inference_steps,
            negative_prompt,
        )
    ):
        scene_dir.mkdir(exist_ok=True, parents=True)
        images = scene.pop("images")

        with (scene_dir / f"scene_{i:03}_metadata.json").open("w") as f:
            json.dump(scene, f, indent=4)

        for image_num, image in enumerate(images):
            image.save(scene_dir / f"scene_{i:03}_image_{image_num:02}.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--lora_model_id",
        type=str,
        default=DEFAULT_LORA_MODEL_ID,
        help="Model ID of LoRA model.",
    )
    parser.add_argument(
        "--refiner_model_id",
        type=str,
        default=DEFAULT_REFINER_MODEL,
        help="Model ID of the Refiner model.",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default=DEFAULT_DATASET_ID,
        help="Huggingface dataset ID.",
    )
    parser.add_argument("--output_dir", type=Path, help="Path to save images.")
    parser.add_argument("--num_images_per_prompt", type=int, default=8)
    parser.add_argument("--num_batches_per_prompt", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(
        args.lora_model_id,
        args.refiner_model_id,
        args.dataset_id,
        Path(args.output_dir),
        args.num_images_per_prompt,
        args.num_batches_per_prompt,
        args.num_inference_steps,
        args.negative_prompt,
        args.enable_xformers_memory_efficient_attention,
        args.debug,
    )
