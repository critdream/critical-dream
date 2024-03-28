"""Compose scenes for each episode based on transcript."""

import argparse
import json
import outlines
import re
import time
from openai import OpenAI
from typing import Iterator

from pathlib import Path
from pydantic import BaseModel


DEFAULT_INITIAL_SPEAKER = "MATT"
REQUEST_INTERVAL = 1.5
MAX_RETRIES = 10


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


MODEL = "gpt-4-turbo-preview"
MODEL_SEED = 41

client = OpenAI()


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

    Take an excerpt of a Dungeons and Dragons session, which is a transcript of
    the dialogue that occurred in a given time frame of the session. Based on
    the speaker, text and timestamp metadata of the captions, output a list of
    highly descriptive scenes.

    Each scene should only have one featured character. If the scene is describing
    the environment, the "character" should be "environment"

    A scene should include the following:
    - character: the main subject of the scene 
    - start_time: the timestamp of when the scene starts
    - end_time: the timestamp of when the scene ends
    - scene_description: a highly descriptive paragraph of the scene, optimized
      so that AI image generation models like Stable Diffusion can create high
      quality images from the description. Keep the description as short as
      possible, but long enough to be highly descriptive.

    The selection of content for the scene description should focus on the
    environment that the characters are in, the actions being taken by the
    player characters (PCs), and actions being taken by the
    non-player characters (NPCs) in the scene.

    The output of this function should be in json format:
    
    {
        "scenes": [
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
    }

    BE SURE TO DO THE FOLLOWING:
    - Don't create scenes for dialogue that appear to be advertisements or meta
      conversations outside of the actual game world.
    - Extract as many scenes as possible.
    - For the character of each scene, use the name of the PC and not the voice
      actor. The following are the names of the voice actors and the PCs they
      play:
      - LAURA: jester
      - TRAVIS: fjord
      - SAM: nott or veth, depending on the context
      - MARISHA: beau
      - TALIESIN: mollymauk or caduceus, depending on the context
      - LIAM: caleb
      - ASHLEY: yasha
      - MATT: plays all of the NPCs in the campaign. Try to avoid using MATT as
        the character name for the scene and instead use the name of the NPC that
        he is voicing.

    Caption Dialogue
    ----------------

    {% for turn in turns %}
    speaker: {{ turn.speaker }}
    text: {{ turn.text }}
    start_time: {{ turn.start }}
    end_time: {{ turn.end }}

    {% endfor %}

    Json Output
    -----------

    """


def postprocess(output: str) -> str:
    """Custom formatting of raw output string to be json-readable."""

    # add tab between new-line and double quotes
    if re.search("\n[a-z_]+\"?:", output):
        output = re.sub("\n([a-z_]+)\"?:", "\n\t\"\\1\":", output)

    # no new-lines before double quotes
    if re.search("\n\"", output):
        output = re.sub("\n\"", "\"", output)

    # remove double quotes in scene_description
    scene_desc_regex = "(\"scene_description\": \".+)\"(.+)\"(.+\"\n)"
    if re.search(scene_desc_regex, output):
        output = re.sub(scene_desc_regex, "\\1'\\2'\\3", output)

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
    if "end_image" in scene:
        val = scene.pop("end_image")
        scene["end_time"] = val
    if "environment" in scene:
        scene.pop("environment")
        scene["character"] = "environment"
    try:
        return Scene(**scene)
    except:
        import ipdb; ipdb.set_trace()
        ...


def iter_turn_batches(
    turns: list[Turn],
    max_text_length: int,
) -> Iterator[list[Turn]]:
    batch = []
    curr_text_length = 0
    for turn in turns:

        if curr_text_length + len(turn.text) > max_text_length:
            yield batch
            batch = []
            curr_text_length = 0

        curr_text_length += len(turn.text)
        batch.append(turn)


def generate_scene_descriptions(turns: list[Turn]) -> str:
    prompt = compose_scene(turns)
    response = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert at converting dialogue into "
                    "highly descriptive scenes."
                )
            },
            {"role": "user", "content": prompt},
        ],
        seed=MODEL_SEED,
    )
    return response.choices[0].message.content


def compose_scenes(
    episode_name: str,
    turns: list[Turn],
    max_text_length: int,
) -> Episode:
    
    scenes = []
    print(f"Composing scenes for episode: {episode_name}")
    for i, turn_batch in enumerate(iter_turn_batches(turns, max_text_length)):
        print(f"Processing batch {i}")
        output = generate_scene_descriptions(turn_batch)
        time.sleep(REQUEST_INTERVAL)

        for i in range(MAX_RETRIES):
            try:
                if i > 0:
                    print(f"Retry {i}")
                json_output = json.loads(output)
                break
            except:
                time.sleep(REQUEST_INTERVAL)

        print(f"Output: {len(json_output['scenes'])} scenes")
        if len(json_output["scenes"]) == 0:
            print("No scenes generated for this batch.")
            continue
        print(json_output["scenes"][0])
        for scene in json_output["scenes"]:
            scenes.append(process_raw_scene(scene))
    return Episode(name=episode_name, scenes=scenes)


def main():
    parser = argparse.ArgumentParser(
        description='Compose scenes for each episode based on transcript'
    )
    parser.add_argument('data_path', help='Path to the caption directory')
    parser.add_argument('output_path', help='Path to the scenes text')
    parser.add_argument(
        '--max-text-length',
        type=int,
        default=10000,
        help='Maximum length of text for each scene',
    )
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
        episode = compose_scenes(episode_name, turns, args.max_text_length)

        with episode_fp.open("w") as f:
            json.dump(episode.model_dump(), f, indent=2)


if __name__ == '__main__':
    main()
