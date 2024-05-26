"""Pyscript app script."""

import pandas as pd
import js
import random
from pyweb import pydom
from pyodide.http import open_url
from pyscript import window, document, display, ffi
from js import console


data_url = (
    "https://huggingface.co/datasets/cosmicBboy/"
    "critical-dream-aligned-scenes-mighty-nein-v1/raw/main/aligned_scenes.csv"
)
image_url_template = (
    "https://huggingface.co/datasets/cosmicBboy/"
    "critical-dream-scene-images-mighty-nein-v1/resolve/main/"
    "{episode_name}/{scene_name}_image_{image_num}.png"
)

NUM_IMAGE_VARIATIONS = 12
SPEAKER_INTERVAL = 500
UPDATE_INTERVAL = 15_000

ABOUT_CONTENTS = """
<div>
    <p>
    👋 Welcome! I'm Niels Bantilan, and I built this project as a big fan of
    Critical Role who happens to be a machine learning engineer.
    <p/>
    
    <p>
    If you're here, there's a chance that you're a fan of Critical Role
    too.
    <p/>
    
    <p>
    The primary goal of Critical Dream is to give you just a little more amusement
    and immersion as you watch the Critical Role cast spin their epic tales over the
    table.
    </p>

    <p>
    The Critical Dream image generation model does its best to render what's
    happening in the episodes as they happen, but you'll notice weird
    things like extra fingers, floating horns, and pointy-earred Caleb and Beau.
    </p>

    <p>
    There's a lot to improve, but honestly I kind of like the fact that the model
    sometimes produces epic looking scenes, but most of the time it's
    cursed, djanky-looking portraits of the characters 🫠.
    </p>

    <p>
    I've rendered the first few episodes of Campaign Two: The Might Nein, with
    more coming soon.
    </p>

    <br>

    <h2>The message of this medium</h2>

    <p>
    This project is possible because of the amazing and talented artists who
    brought Critical Role's cast of characters to life. To create the Critical
    Dream image generation model, I fined-tuned
    <a href="https://huggingface.co/papers/2307.01952" target="_blank">Stable Diffusion XL</a>
    on this art, and I do not take this act lightly because
    
    <strong><i>
    the data is the foundation of the model, and I believe that those creating
    the data have a right to the monetary gains resulting from the model.
    </i></strong>

    </p>
    
    <p>
    According to this belief, here are the design decisions that I laid out
    to guide this project:
    </p>

    <ul>
        <li>
        🚫 I have not monetized this website, nor do I have any plans on monetizing
        it without some kind of profit-sharing agreement in place with the
        credited artists.
        </li>

        <li>
        🏞️ For full transparency, the fine-tuning data for Critical Role-specific
        characters are documented and credited
        <a href="https://github.com/cosmicBboy/critical-dream" target="_blank">here</a>.
        </li>

        <li>
        🤖 The model and training code will not be open sourced until the license
        and usage parameters of the model are agreed upon by myself and the
        credited artists.
        </li>

        <li>
        🎲 Images are displayed alongside the original YouTube videos managed by
        the Critical Role team. I will not upload my own YouTube videos to
        divert views from their channel.
        </li>
    </ul>

    <br>

    <h2>
    To the credited artists
    </h2>

    <p>
    Beyond the entertainment value of Critical Dream, I built this project
    because I also want to spur a different kind of discourse at the intersection
    of Art and AI by engaging with you and the creative community more broadly.
    </p>

    <p>
    For a long time I've thought that many of today's AI companies take valuable data
    published on the internet and use the "AI" label to re-package them behind a
    walled garden where you pay for software services built on top of that data. The
    creators of that data get nothing in return, and this is exactly where the
    creative value chain is broken ⛓️.
    </p>

    <p>
    At the same time, other organizations publish their models as open source
    or open weights. While this is great for AI research, this also breaks the
    creative value chain because it commoditizes the work pain-stakingly produced
    by people. The question I ask myself as I develop this project is:
    </p>

    <blockquote>
    Is there a healthier way to organize creative and technical ecosystems on the
    internet so that we can all create cool things while appropriately crediting
    and compensating all of the stakeholders in the ecosystem, from creators to
    engineers?
    </blockquote>

    <p>
    If you want to discuss topics like this and get involved with this project,
    please feel free to join the <a href="#">discord channel</a>.
    </p>

</div>
"""

EPISODE_STARTS = {
    "c2e001": 854,
    "c2e002": 504,
    "c2e003": 420,
}

EPISODE_BREAKS = {
    "c2e001": (5529, 6547),
    "c2e002": (7583, 8470),
    "c2e003": (7992, 8921),
}

SCENE_DURATION = 5


speaker_update_interval_id = None
image_update_interval_id = None

speaker = None
character = None
scene_id = None
last_scene_time = 0



def load_data():

    def scene_name(df):
        return "scene_" + df.scene_id.astype(str).str.pad(3, fillchar="0")

    def midpoint(df):
        mid = (df["end_time"] - df["start_time"]) / 2
        return df["start_time"] + mid

    return (
        pd.read_csv(open_url(data_url))
        .rename(columns={"start": "start_time", "end": "end_time"})
        .assign(
            scene_name=scene_name,
            mid_point=midpoint,
        )
    )


def log(message):
    print(message)  # log to python dev console
    console.log(message)  # log to JS console


def set_episode_dropdown(df):
    select = pydom["select#episode"][0]
    episodes = df.episode_name.unique()

    for episode_name in episodes:
        num = episode_name.split("e")[1]
        content = f"Campaign 2 Episode {num}"
        option = pydom.create("option", html=content)
        option.value = episode_name
        select.append(option)


def set_current_episode(event):
    global player, video_id_map

    episode_name = document.getElementById("episode").value
    video_id = video_id_map[episode_name]
    console.log(f"video id: {video_id}")
    # set video on the youtube player
    player.cueVideoById(video_id)


def find_closest_scene(
    df: pd.DataFrame,
    current_time: float,
    environment: bool = False,
) -> pd.Series:
    if environment:
        df = df.query("character == 'environment'")
    distance = abs(df["mid_point"] - current_time)
    closest_scene = df.loc[distance.idxmin()]
    return closest_scene


def find_scene(
    episode_name: str,
    df: pd.DataFrame,
    current_time: float,
    speaker: str | None = None,
    character: str | None = None,
) -> pd.Series:
    df = df.query(f"episode_name == '{episode_name}'")

    if speaker:
        df = df.query(f"speaker == '{speaker}'")

    if character:
        df = df.query(f"character == '{character}'")

    current_time = min(current_time, df["end_time"].max())
    current_time = max(current_time, df["start_time"].min())

    break_start, break_end = EPISODE_BREAKS[episode_name]
    if current_time <= EPISODE_STARTS[episode_name]:
        return find_closest_scene(df, current_time, environment=True)
    elif break_start <= current_time <= break_end:
        # during the mid-episode break, show an environment image from the intro
        return find_closest_scene(df, 0, environment=True)

    result = df.loc[
        (df["start_time"] <= current_time)
        & (current_time <= df["end_time"])
    ]

    # if found, return result
    if not result.empty:
        assert result.shape[0] == 1
        return result.iloc[0]

    # otherwise find the closest environment scene to the timestamp
    return find_closest_scene(df, current_time)


@ffi.create_proxy
def update_image():
    global df, player, speaker, character

    current_time = float(player.getCurrentTime())
    episode_name = document.getElementById("episode").value

    scene_name = find_scene(
        episode_name,
        df,
        current_time,
        speaker=speaker,
        character=character,
    )["scene_name"]

    image_num = str(random.randint(0, 11)).zfill(2)
    image_url = image_url_template.format(
        episode_name=episode_name, scene_name=scene_name, image_num=image_num
    )
    console.log(f"updating image, current time: {current_time}")

    current_image = document.querySelector("img#current-image")
    current_image.classList.remove("show")

    @ffi.create_proxy
    def set_new_image():
        current_image.setAttribute("src", image_url)

    @ffi.create_proxy
    def show_new_image():
        current_image.classList.add("show")

    js.setTimeout(set_new_image, 50)
    js.setTimeout(show_new_image, 100)


@ffi.create_proxy
def update_speaker():
    global df, player, speaker, character, scene_id, last_scene_time

    current_time = float(player.getCurrentTime())
    episode_name = document.getElementById("episode").value
    scene = find_scene(episode_name, df, current_time)

    new_speaker = scene["speaker"]
    new_character = scene["character"]
    new_scene_id = scene["scene_id"]
    console.log(f"current speaker: {speaker}, character: {character}, new_scene_id: {new_scene_id}")

    update_scene = False
    if (current_time - last_scene_time) > SCENE_DURATION:
        update_scene = True
        last_scene_time = current_time
    elif current_time == 0:
        update_scene = True

    if update_scene and (
        character != new_character
        or scene_id != new_scene_id
    ):
        console.log(f"update image | speaker: {speaker}, character: {character} new_scene_id: {new_scene_id}")
        speaker = new_speaker
        character = new_character
        scene_id = new_scene_id
        update_image()


@ffi.create_proxy
def on_youtube_frame_api_ready():
    global player

    console.log("on_youtube_frame_api_ready")
    player = window.YT.Player.new(
        "player",
        videoId="byva0hOj8CU",
        playerVars=ffi.to_js({
            "cc_load_policy": 1,
        })
    )
    player.addEventListener("onReady", on_ready)
    player.addEventListener("onStateChange", on_state_change)


@ffi.create_proxy
def close_modal():
    loading = document.getElementById('loading')
    loading.close()


@ffi.create_proxy
def on_ready(event):
    global image_update_interval_id, speaker_update_interval_id

    console.log("[pyscript] youtube iframe ready")

    if speaker_update_interval_id is None: 
        speaker_update_interval_id = js.setInterval(update_speaker, SPEAKER_INTERVAL)
        console.log(f"set speaker interval id: {speaker_update_interval_id}")

    if image_update_interval_id is None:
        # image_update_interval_id = js.setInterval(update_image, UPDATE_INTERVAL)
        console.log(f"set image interval id: {image_update_interval_id}")

    resize_iframe(event)
    js.setTimeout(close_modal, 1500)


@ffi.create_proxy
def on_state_change(event):
    global last_scene_time

    console.log(f"[pyscript] youtube player state change {event.data}")
    if int(event.data) in (-1, 1):
        # update speaker and image when new episode is selected (-1) or the
        # user jumps to different part of the video
        update_speaker()
        last_scene_time = 0


@ffi.create_proxy
def resize_iframe(event):
    # log("resizing iframe")
    container = document.getElementById("image")
    image = document.getElementById("current-image")
    iframe = document.getElementById("player")
    # set to current width
    iframe.height = container.clientWidth
    container.height = container.clientWidth
    image.height = container.clientWidth


def create_youtube_player():
    window.onYouTubeIframeAPIReady = on_youtube_frame_api_ready

    # insert iframe_api script
    tag = document.createElement("script")
    div = document.getElementById('youtube-player');
    tag.src = "https://www.youtube.com/iframe_api"
    div.appendChild(tag)

    # make sure iframe is the same size as the image
    window.addEventListener("resize", resize_iframe)


def show_about(event):
    about_model = document.getElementById("about")
    about_model.showModal()


def hide_about(event):
    about_modal = document.getElementById("about")
    about_modal.close()


def main():
    console.log("Starting up app...")
    global df, video_id_map

    about = document.getElementById("about-contents")
    about.innerHTML = ABOUT_CONTENTS

    df = load_data()
    video_id_map = df.groupby("episode_name").youtube_id.first()
    log(f"data {df.head()}")
    log(f"video id map {video_id_map}")

    # set dropdown values and set current episode onchange function
    set_episode_dropdown(df)
    episode_selector = document.getElementById("episode")
    episode_selector.onchange = set_current_episode

    # create youtube player
    create_youtube_player()
    console.log(window)


main()
