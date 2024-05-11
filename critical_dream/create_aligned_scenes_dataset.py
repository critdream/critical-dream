"""Create a huggingface dataset of scenes that are aligned with speakers.

This dataset is used when rendering the final product, where composed scenes
are matched up with the speaker based on the raw captions. This allows the
images to better-match who is speaking in the scene.
"""

import json
import pandas as pd

from argparse import ArgumentParser
from pathlib import Path

from datasets import Dataset

from critical_dream.compose_scenes import Caption, parse_captions


SPEAKER_MAP = {
    "fjord": "TRAVIS",
    "beau": "MARISHA",
    "jester": "LAURA",
    "caduceus": "TALIESIN",
    "mollymauk": "TALIESIN",
    "yasha": "ASHLEY",
    "caleb": "LIAM",
    "veth": "SAM",
    "nott": "SAM",
    "matt": "MATT",
    "environment": "MATT",
    "npc": "MATT",
}


def read_captions(caption_file: Path) -> pd.DataFrame:
    with caption_file.open() as f:
        data = [Caption(**x) for x in json.load(f)]

    turns = parse_captions(data)
    return pd.DataFrame([t.model_dump() for t in turns])


def read_scenes(scene_file: Path) -> pd.DataFrame:
    with scene_file.open() as f:
        episode = json.load(f)

    scenes = []
    for scene in episode["scenes"]:
        episode_name, youtube_id = episode["name"].split("_", 1)
        scene.pop("turns")
        scenes.append({
            "episode_name": episode_name,
            "youtube_id": youtube_id,
            **scene,
        })

    return pd.DataFrame(scenes)


def align_scenes_with_speakers(scenes: pd.DataFrame, captions: pd.DataFrame) -> pd.DataFrame:
    """Aligns the speaker in the raw transcripts with a generated scene.

    This function should return a dataframe where each caption is potentially
    aligned with a scene.

    The caption is matched based on speaker and whether the start - end interval
    of the scene is within some tolerance threshold of the caption's timestamp.
    """
    scenes = scenes.reset_index().rename(columns={"index": "scene_id"})
    
    for row in captions.itertuples():
        ...
    import ipdb; ipdb.set_trace()
    ...


def main(caption_dir: Path, scene_dir: Path, dataset_id: str):

    aligned_scenes = []
    for caption_file in sorted(caption_dir.glob("*.json")):
        print(f"processing {caption_file}")

        scene_file = scene_dir /f"{caption_file.stem}_scenes.json"
        if not scene_file.exists():
            break

        captions = read_captions(caption_file)
        scenes = read_scenes(scene_file)
        aligned_scenes.append(align_scenes_with_speakers(scenes, captions))

    dataset = Dataset.from_list(pd.concat(aligned_scenes))
    print(f"pushing to huggingface hub: {dataset_id}")
    dataset.push_to_hub(dataset_id)
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--captions_dir",
        type=str,
        help="Directory containing caption JSON files.",
    )
    parser.add_argument(
        "--scene_dir",
        type=str,
        help="Directory containing scene JSON files.",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        help="Huggingface dataset ID to push the scenes to.",
    )
    args = parser.parse_args()
    main(Path(args.captions_dir), Path(args.scene_dir), args.dataset_id)
