"""Create a huggingface dataset of scenes that are aligned with speakers.

This dataset is used when rendering the final product, where composed scenes
are matched up with the speaker based on the raw captions. This allows the
images to better-match who is speaking in the scene.

TODO: save aligned scenes as separate CSV files, one per episode.
"""

import json
import pandas as pd
from io import BytesIO
from tempfile import TemporaryDirectory

from argparse import ArgumentParser
from pathlib import Path

from datasets import Dataset

from critical_dream.compose_scenes import Caption, parse_captions
from huggingface_hub import upload_file, upload_folder


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
    for i, scene in enumerate(episode["scenes"]):
        episode_name, youtube_id = episode["name"].split("_", 1)
        scene.pop("turns")
        scenes.append({
            "episode_name": episode_name,
            "youtube_id": youtube_id,
            "scene_id": i,
            **scene,
        })

    return pd.DataFrame(scenes)


def align_scenes_with_speakers(
    scenes: pd.DataFrame,
    captions: pd.DataFrame,
    tolerance: float = 60.0,
) -> pd.DataFrame:
    """Aligns the speaker in the raw transcripts with a generated scene.

    This function should return a dataframe where each caption is potentially
    aligned with a scene.

    The caption is matched based on speaker and whether the start time
    of the scene is within some tolerance threshold of the caption's timestamp.
    """
    aligned_scene_ids = []
    for row in captions.itertuples():
        time_selector = (row.start + tolerance) >= scenes.start_time
        speaker_selector = row.speaker.lower() == scenes.speaker.str.lower()
        relevant_scenes = scenes[time_selector & speaker_selector]
        if relevant_scenes.empty:
            # unaligned scenes get a scene_id of -1
            aligned_scene_ids.append(-1)
        elif len(relevant_scenes) == 1:
            aligned_scene_ids.append(relevant_scenes.scene_id.item())
        else:
            # to resolve multiple relevant scenes, pick the closest one that
            # occured before the caption timestamp
            selected_scene = relevant_scenes.loc[
                (relevant_scenes.start_time - row.start).abs().idxmin()
            ]
            aligned_scene_ids.append(selected_scene.scene_id)

    captions["scene_id"] = aligned_scene_ids
    captions = captions.merge(
        scenes[["scene_id", "character", "in_game"]],
        on=["scene_id"],
        how="left",
    )
    captions["episode_name"] = scenes["episode_name"].iloc[0]
    captions["youtube_id"] = scenes["youtube_id"].iloc[0]

    # remove unaligned scenes
    captions = captions[captions.scene_id != -1]
    return captions


def main(caption_dir: Path, scene_dir: Path, dataset_id: str):

    aligned_output = {}
    for caption_file in sorted(caption_dir.glob("*.json")):
        print(f"processing {caption_file}")
        episode_name = caption_file.stem.split("_")[0]

        scene_file = scene_dir /f"{caption_file.stem}_scenes.json"
        if not scene_file.exists():
            break

        captions = read_captions(caption_file)
        scenes = read_scenes(scene_file)
        if scenes.empty:
            print("no scenes found for this episode, skipping.")
            continue
        aligned = align_scenes_with_speakers(scenes, captions)
        aligned_output[episode_name] = aligned

    all_episodes_df = pd.concat(aligned_output.values())
    aligned_output["all"] = all_episodes_df

    print(f"pushing to huggingface hub: {dataset_id}")
    all_episodes_dataset = Dataset.from_pandas(all_episodes_df)
    all_episodes_dataset.push_to_hub(dataset_id)

    with TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir)
        for episode_name, df in aligned_output.items():
            df.to_csv(out_path / f"aligned_scenes_{episode_name}.csv", index=False)

        # maps episode names to video id
        video_id_map = all_episodes_df.groupby("episode_name").youtube_id.first()
        video_id_map.to_csv(out_path / "video_id_map.csv", header=True)

        upload_folder(
            repo_id=dataset_id,
            repo_type="dataset",
            folder_path=out_path,
            path_in_repo=".",
        )


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
