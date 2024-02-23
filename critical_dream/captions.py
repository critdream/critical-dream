from youtube_transcript_api import YouTubeTranscriptApi


video_ids = [
    "eRFetHZDSg4",
]


def main():
    transcripts = []
    for video_id in video_ids:
        transcripts.append(YouTubeTranscriptApi.get_transcript(video_id))
    return transcripts
