"""Create a huggingface dataset from the composed scenes."""

import json

from argparse import ArgumentParser
from pathlib import Path

from datasets import Dataset


def main(input_dir: Path, dataset_id: str):
    scenes = []
    for file in sorted(input_dir.glob("*.json")):
        with file.open() as f:
            episode = json.load(f)
        for i, scene in enumerate(episode["scenes"]):
            episode_name, youtube_id = episode["name"].split("_", 1)
            scenes.append({
                "episode_name": episode_name,
                "youtube_id": youtube_id,
                "scene_id": i,
                **scene,
            })

    dataset = Dataset.from_list(scenes)
    print(f"pushing to huggingface hub: {dataset_id}")
    dataset.push_to_hub(dataset_id)
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the input JSON file containing the scenes.",
    )
    parser.add_argument(
        "dataset_id",
        type=str,
        help="Huggingface dataset ID to push the scenes to.",
    )
    args = parser.parse_args()
    main(Path(args.input_dir), args.dataset_id)
