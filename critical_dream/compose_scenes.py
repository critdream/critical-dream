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
    background: str
    action: str
    object: str
    poses: str
    start_time: float
    end_time: float
    scene_description: str
    turns: list[Turn]


class Episode(BaseModel):
    name: str
    scenes: list[Scene]


SCENE_COMPOSITION_MODEL = "gpt-4-turbo"
JSON_FIX_MODEL = "gpt-3.5-turbo-0125"
TIMESTAMP_FIX_MODEL = "gpt-3.5-turbo-0125"
MODEL_SEED = 42

client = OpenAI()


def new_speaker(text: str) -> bool:
    return re.match("^[A-Z]+:", text)


def parse_captions(
    data: list[Caption],
    default_initial_speaker: str = DEFAULT_INITIAL_SPEAKER,
) -> list[Turn]:

    turns: list[Turn] = []
    speaker = None

    for i, caption in enumerate(data):
        if i == 0 or new_speaker(caption.text):
            text_split = caption.text.split(":", 1)
            if len(text_split) == 1:
                speaker, text = default_initial_speaker, text_split[0]
            else:
                speaker, text = text_split

        else:
            text = caption.text

        turns.append(
            Turn(
                speaker=speaker,
                text=text,
                start=caption.start,
                end=caption.start + caption.duration,
            )
        )

    return turns


@outlines.prompt
def compose_scene(turns: list[Turn]) -> str:
    """Write highly descriptive scenes using the captions from raw dialogue.

    Take an excerpt of a Dungeons and Dragons session transcript and write
    scenes of what's happening in a given window of time. Based on the speaker,
    text and timestamp metadata of the captions, output a list of highly
    descriptive scenes.

    Each scene should only have one featured character. If the scene is
    describing the environment, the "character" should be "environment".

    A scene consists of the following:
    - character: the main subject of the scene. If the scene is describing something
      happening in-game, DO NOT USE the voice actor's name, use the name of
      the player character instead.
    - background: a few word description of the scene background
    - action: the action that the character is doing
    - object: the object or other character that the main subject character is
      interacting with, if applicable
    - poses: comma-separated list of suggested poses for the character in the scene
    - start_time: the timestamp of when the scene starts
    - end_time: the timestamp of when the scene ends
    - scene_description: a highly descriptive paragraph of the scene, optimized
      so that AI image generation models can create high quality images from
      the description.

    The selection of dialogue for the scene description should focus on the
    environment that the characters are in, the actions being taken by the
    player characters (PCs), and actions being taken by the
    non-player characters (NPCs) in the scene.

    The output of this function should be in json format:
    
    {
        "scenes": [
            {
                "character": "environment",
                "background": "a short description of the physical environment.",
                "action": "none",
                "object": "none",
                "poses": "none",
                "start_time": 0,
                "end_time": 10,
                "scene_description": "this is a description of the environment."
            },
            {
                "character": "fjord",
                "background": "a short description of the scene background",
                "action": "what fjord is doing",
                "object": "the object or other character fjord is interacting with",
                "poses": "pose suggestion 1, pose suggestion 2",
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
    - Keep the scene descriptions as short as possible but still highly descriptive.
    - For the character of each scene, use the name of the player character and
      not the voice actor. The following are the names of the voice actors and
      the player characters they play. Below are the names of the voice actors
      and the characters they play.
    - MAKE SURE THE start_time AND end_time TIMESTAMPS ARE BASED ON THE DIALOGUE
      THAT THE SCENE DESCRIPTION IS REFERRING TO.

    VOICE ACTOR: player character
    -----------------------------
    LAURA: jester
    TRAVIS: fjord
    SAM: nott or veth, depending on the context
    MARISHA: beau
    TALIESIN: mollymauk or caduceus, depending on the context
    LIAM: caleb
    ASHLEY: yasha
    MATT: plays all of the NPCs in the campaign. Avoid using MATT as
        the character for the scene and instead use the name of the NPC
        that he is voicing.

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


@outlines.prompt
def fix_json(json_str: str) -> str:
    """Fix the json output from the scene composition prompt.

    The format of this json output is not valid. Fix the json output so that it
    follows the json format:

    {
        "scenes": [
            {
                "character": "environment",
                "background": "a short description of the physical environment.",
                "action": "none",
                "object": "none",
                "poses": "",
                "start_time": 0,
                "end_time": 10,
                "scene_description": "this is a description of the environment."
            },
            {
                "character": "fjord",
                "background": "a short description of the physical environment.",
                "action": "what fjord is doing",
                "object": "the object or other character fjord is interacting with",
                "poses": "pose suggestion 1, pose suggestion 2",
                "start_time": 10,
                "end_time": 40,
                "scene_description": "this is a description of what fjord is doing."
            },
        ]
    }

    Invalid Json
    ------------
    {{ json_str }}

    Json Output
    -----------
    """


@outlines.prompt
def fix_timestamp(scene: str, turns: list[Turn], prev_timestamp: float) -> str:
    """Fix the timestamp associated with a scene description.

    The timestamp data associated with the scene description is potentially
    incorrect. Fix the "start_time" and "end_time" values so that they line
    up with the actual timestamps of the dialogue that the scene description is
    referring to.

    When the scene description is referring to a player character, e.g. Fjord,
    use the VOICE ACTOR: player character information below to determine what
    timestamp associated with the scene. When the scene description is referring
    to an NPC or the environment, the timestamp is usually associated with MATT,
    the dungeon master.

    The output format should be valid json:

    {
        "character": "character",
        "background": "a short description of the character.",
        "action": "none",
        "object": "none",
        "poses": "",
        "start_time": <START_TIME>
        "end_time": <END_TIME>,
        "scene_description": "this is the scene description."
    },

    VOICE ACTOR: player character
    -----------------------------
    LAURA: jester
    TRAVIS: fjord
    SAM: nott or veth, depending on the context
    MARISHA: beau
    TALIESIN: mollymauk or caduceus, depending on the context
    LIAM: caleb
    ASHLEY: yasha
    MATT: plays all of the NPCs in the campaign. Avoid using MATT as
        the character for the scene and instead use the name of the NPC
        that he is voicing.

    MAKE SURE YOU DO THE FOLLOWING:
    - THE start_time AND end_time TIMESTAMPS ARE BASED ON THE DIALOGUE THAT THE
      SCENE DESCRIPTION IS REFERRING TO.
    - THE start_time of the scene must be after the end_time of the previous
      scene: {{ prev_timestamp }}

    Caption Dialogue
    ----------------

    {% for turn in turns %}
    speaker: {{ turn.speaker }}
    text: {{ turn.text }}
    start_time: {{ turn.start }}
    end_time: {{ turn.end }}

    {% endfor %}

    Scene Description
    -----------------

    {{ scene }}

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


def process_raw_scene(scene: dict) -> dict:
    for key in scene:
        if key.startswith("end") and key != "end_time":
            val = scene.pop(key)
            scene["end_time"] = val

    if "environment" in scene:
        scene.pop("environment")
        scene["character"] = "environment"
    return scene


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
        model=SCENE_COMPOSITION_MODEL,
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


def fix_scene_description(json_str: str) -> str:
    prompt = fix_json(json_str)
    response = client.chat.completions.create(
        model=JSON_FIX_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are an expert at fixing invalid json.",
            },
            {"role": "user", "content": prompt},
        ],
        seed=MODEL_SEED,
    )
    return response.choices[0].message.content


def fix_timestamp_data(scene: dict, turns: list[Turn], prev_timestamp: float) -> str:
    scene["start_time"] = "..."
    scene["end_time"] = "..."
    prompt = fix_timestamp(scene, turns, prev_timestamp)
    response = client.chat.completions.create(
        model=TIMESTAMP_FIX_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert at matching timestamp data with the "
                    "correct content."
                ),
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
    prev_timestamp = 0

    for i, turn_batch in enumerate(iter_turn_batches(turns, max_text_length)):
        print(f"Processing batch {i}")
        output = generate_scene_descriptions(turn_batch)
        time.sleep(REQUEST_INTERVAL)

        for i in range(MAX_RETRIES):
            try:
                if i > 0:
                    print(f"Retry {i}")
                json_output = json.loads(output)
                print(f"Output: {len(json_output['scenes'])} scenes")
                if len(json_output["scenes"]) == 0:
                    print("No scenes generated for this batch.")
                    continue
                for scene in json_output["scenes"]:
                    print(scene)

                for scene in json_output["scenes"]:
                    # scene = json.loads(fix_timestamp_data(scene, turn_batch, prev_timestamp))
                    try:
                        scene = Scene(**process_raw_scene(scene), turns=turn_batch)
                        scenes.append(scene)
                        prev_timestamp = scene.end_time
                    except Exception:
                        print(f"Error processing scene: {scene}")
                break
            except:
                output = fix_scene_description(output)
                time.sleep(REQUEST_INTERVAL)

    return Episode(name=episode_name, scenes=scenes)


def main():
    parser = argparse.ArgumentParser(
        description='Compose scenes for each episode based on transcript'
    )
    parser.add_argument('data_path', help='Path to the caption directory')
    parser.add_argument('output_path', help='Path to the scenes text')
    parser.add_argument("--n_episodes", type=int, default=None)
    parser.add_argument(
        '--max_text_length',
        type=int,
        default=5000,
        help='Maximum length of text for each scene',
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, file in enumerate(sorted(data_path.glob("*.json")), start=1):

        if args.n_episodes is not None and i > args.n_episodes:
            break

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
