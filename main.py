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
UPDATE_INTERVAL = 15_000


def load_data():

    def scene_name(df):
        return "scene_" + df.scene_id.astype(str).str.pad(3, fillchar="0")

    def midpoint(df):
        return (df["end_time"] - df["start_time"]) / 2

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
        content = f"Episode {num}"
        option = pydom.create("option", html=content)
        option.value = episode_name
        select.append(option)


def set_current_episode(event):
    global player
    global video_id_map

    episode_name = document.getElementById("episode").value
    video_id = video_id_map[episode_name]
    console.log(f"video id: {video_id}")
    # set video on the youtube player
    player.cueVideoById(video_id)


def find_scene_name(df: pd.DataFrame, current_time: float) -> str:
    current_time = min(current_time, df["end_time"].max())
    current_time = max(current_time, df["start_time"].min())

    result = df.loc[
        (df["start_time"] <= current_time)
        & (current_time <= df["end_time"])
    ]

    # if found, return result
    if not result.empty:
        log(result)
        assert result.shape[0] == 1
        return result.iloc[0]["scene_name"]

    # otherwise find the closest environment scene to the timestamp
    environment_scene = df.query("speaker == 'MATT'")
    distance = abs(environment_scene["mid_point"] - current_time)
    closest_scene = environment_scene.loc[distance.idxmin()]["scene_name"]
    return closest_scene


@ffi.create_proxy
def update_image():
    global df
    global player

    current_time = float(player.getCurrentTime())
    episode_name = document.getElementById("episode").value
    scene_name = find_scene_name(df.query(f"episode_name == '{episode_name}'"), current_time)
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

    js.setTimeout(set_new_image, 1000)
    js.setTimeout(show_new_image, 1200)



@ffi.create_proxy
def on_youtube_frame_api_ready():
    global player

    console.log("on_youtube_frame_api_ready")
    player = window.YT.Player.new(
        "player",
        videoId="byva0hOj8CU",
    )
    player.addEventListener("onReady", on_ready)
    player.addEventListener("onStateChange", on_state_change)


@ffi.create_proxy
def close_modal():
    loading = document.getElementById('loading')
    loading.close()


@ffi.create_proxy
def on_ready(event):
    global interval_id

    console.log("[pyscript] youtube iframe ready")
    js.setTimeout(update_image, 1000)
    js.setTimeout(close_modal, 2000)
    resize_iframe(event)


@ffi.create_proxy
def on_state_change(event):
    console.log("[pyscript] youtube player state change")
    interval_id = js.setInterval(update_image, UPDATE_INTERVAL)
    console.log(f"set interval id: {interval_id}")
    update_image()


@ffi.create_proxy
def resize_iframe(event):
    # log("resizing iframe")
    container = document.getElementById("image")
    iframe = document.getElementById("player")
    # set to current width
    iframe.height = container.clientWidth


def create_youtube_player():
    window.onYouTubeIframeAPIReady = on_youtube_frame_api_ready

    # insert iframe_api script
    tag = document.createElement("script")
    div = document.getElementById('youtube-player');
    tag.src = "https://www.youtube.com/iframe_api"
    div.appendChild(tag)

    # make sure iframe is the same size as the image
    window.addEventListener("resize", resize_iframe)


def main():
    console.log("Starting up app...")
    global df
    global video_id_map

    df = load_data()
    video_id_map = df.groupby("episode_name").youtube_id.first()
    log(f"data {df.head()}")
    log(f"video id map {video_id_map}")
    display(df.head(), target="pandas-output-inner", append="False")

    # set dropdown values and set current episode onchange function
    set_episode_dropdown(df)
    episode_selector = document.getElementById("episode")
    episode_selector.onchange = set_current_episode

    # create youtube player
    create_youtube_player()
    console.log(window)


main()
