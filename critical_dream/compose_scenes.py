"""Compose scenes for each episode based on transcript."""

import argparse
import json
import outlines
import re
import time

from pathlib import Path
from pydantic import BaseModel


DEFAULT_INITIAL_SPEAKER = "MATT"


class Caption(BaseModel):
    text: str
    start: float
    duration: float


class Turn(BaseModel):
    speaker: str
    text: str
    start: float
    end: float


class Scene(BaseModel):
    character: str
    start_time: float
    end_time: float
    scene_description: str


class Episode(BaseModel):
    name: str
    scenes: list[Scene]


model = outlines.models.openai("gpt-4-turbo-preview")
generator = outlines.generate.text(model)


def new_speaker(text: str) -> bool:
    return re.match("^[A-Z]+:", text)


def parse_captions(
    data: list[Caption],
    default_initial_speaker: str = DEFAULT_INITIAL_SPEAKER,
) -> list[Turn]:

    turns: list[Turn] = []
    turn_batch: list[str] = []
    speaker = None
    start_time = None
    end_time = None

    for i, caption in enumerate(data):
        if i == 0 or new_speaker(caption.text):

            if turn_batch:
                # collect turn for current speaker before starting a new turn
                # batch for the new speaker
                end_time = start_time + caption.duration
                text = " ".join(turn_batch).strip().replace("\n", " ")
                turns.append(
                    Turn(
                        speaker=speaker,
                        text=text,
                        start=start_time,
                        end=end_time,
                    )
                )

                # refresh turn_batch
                turn_batch = []

            start_time = caption.start
            text_split = caption.text.split(":", 1)
            if len(text_split) == 1:
                speaker = default_initial_speaker
                turn_batch.append(text_split[0])
            else:
                speaker, text = text_split
                turn_batch.append(text)

        else:
            turn_batch.append(caption.text)

    return turns


@outlines.prompt
def compose_scene(turns: list[Turn]) -> str:
    """Write highly descriptive scenes using the captions from raw dialogue.

    The job of this function is to take an excerpt of a Dungeons and Dragons session,
    which is a transcript of the dialogue that occurred in a given time frame
    of the session. Based on the information and timestamp metadata of the captions,
    this function will output a list of highly descriptive scenes of the session.

    Each scene should only have one feature character. If the scene is describing
    the environment, the "character" should be "environment"

    The description should include the following:
    - character: the main subject of the scene 
    - start_time: the timestamp of when the scene starts
    - end_time: the timestamp of when the scene ends
    - scene_description: a highly descriptive paragraph of the scene, optimized
      so that AI image generation models like Stable Diffusion can create high
      quality images from the description.

    The selection of content for the scene description should focus on the
    environment that the characters are in, the actions being taken by the players,
    and actions being taken by the non-player characters in the scene.

    The output of this function should be in json format:
    
    [
        {
            "character": "environment",
            "start_time": 0,
            "end_time": 10,
            "scene_description": "this is a description of the environment."
        },
        {
            "character": "fjord",
            "start_time": 10,
            "end_time": 40,
            "scene_description": "this is a description of what fjord is doing."
        },
    ]

    Extract as many scenes as possible.

    Caption Dialogue
    ----------------

    {% for turn in turns %}
    start_time: {{ turn.start }}
    end_time: {{ turn.end }}
    player: {{ turn.speaker }}
    text: {{ turn.text }}

    {% endfor %}

    Json Output
    -----------

    """


def postprocess(output: str) -> str:
    splits = output.split("\n")
    if output.startswith("```json"):
        splits = splits[1:]
    if output.endswith("```"):
        splits = splits[:-1]
    try:
        if splits[-2].strip() == "...":
            splits.pop(-2)
    except IndexError:
        ...
    return "\n".join(splits)


def process_raw_scene(scene: dict) -> Scene:
    if "end_title" in scene:
        val = scene.pop("end_title")
        scene["end_time"] = val
    return Scene(**scene)


def compose_scenes(
    episode_name: str,
    turns: list[Turn],
    chunk_size: int = 1000,
) -> Episode:
    
    scenes = []
    print(f"Composing scenes for episode: {episode_name}")
    for i in range(0, len(turns), chunk_size):
        print(f"Processing chunk {i}")
        chunk = turns[i: i + chunk_size]
        output = generator(compose_scene(chunk))
        time.sleep(2)
        output = postprocess(output)
        try:
            json_output = json.loads(output)
        except:
            import ipdb; ipdb.set_trace()
            ...
        for scene in json_output:
            print(json.dumps(scene, indent=2))
            scenes.append(process_raw_scene(scene))
    return Episode(name=episode_name, scenes=scenes)


def main():
    parser = argparse.ArgumentParser(
        description='Compose scenes for each episode based on transcript'
    )
    parser.add_argument('data_path', help='Path to the caption directory')
    parser.add_argument('output_path', help='Path to the scenes text')
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for file in sorted(data_path.glob("*.json")):

        episode_name = file.stem
        episode_fp = output_path / f"{episode_name}_scenes.json"
        if episode_fp.exists():
            print(f"Episode {episode_name} already processed. Skipping.")
            continue

        with file.open() as f:
            data = [Caption(**x) for x in json.load(f)]

        turns = parse_captions(data)
        episode = compose_scenes(episode_name, turns)

        with episode_fp.open("w") as f:
            json.dump(episode.model_dump(), f, indent=2)


if __name__ == '__main__':
    main()
