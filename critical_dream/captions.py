"""Download youtube captions."""

from typing import Iterator

import json
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import Playlist


CRITROLE_CAMPAIGN_TWO = "https://www.youtube.com/playlist?list=PL1tiwbzkOjQxD0jjAE7PsWoaCrs0EkBH2"


def get_video_ids(playlist_link: str) -> Iterator[str]:
    playlist = Playlist(playlist_link)
    for video_id in playlist.video_urls:
        yield video_id.split("v=")[-1]


def main(output_path: Path, playlist_link: str):

    print(f"Downloading transcripts for playlist: {playlist_link}")

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    for i, video_id in enumerate(get_video_ids(playlist_link), start=1):
        print(f"Downloading transcript for video {i}: {video_id}")
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_manually_created_transcript(["en-US", "en"]).fetch()
        transcript_fp = output_path / f"c2e{i:03}_{video_id}.json"
        print(f"Writing to {transcript_fp}")
        with transcript_fp.open("w") as f:
            json.dump(transcript, f, indent=2)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("output_path", type=str)
    parser.add_argument(
        "--playlist_link",
        type=str,
        default=CRITROLE_CAMPAIGN_TWO,
    )
    args = parser.parse_args()

    main(Path(args.output_path), args.playlist_link)
