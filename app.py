import os
from flask import Flask, request, jsonify
from googleapiclient.discovery import build
from transformers import pipeline
import pandas as pd

app = Flask(__name__)


def get_youtube_comments(video_url, api_key):
    # Extract video ID from the URL
    video_id = video_url.split("v=")[1]

    # Create a YouTube API client
    youtube = build("youtube", "v3", developerKey=api_key)

    # Get video details
    video_response = youtube.videos().list(part="snippet", id=video_id).execute()

    video_title = video_response["items"][0]["snippet"]["title"]

    # Get comments
    comments = []
    nextPageToken = None

    while True:
        comment_response = (
            youtube.commentThreads()
            .list(
                part="snippet",
                videoId=video_id,
                maxResults=100,  # Adjust as needed
                pageToken=nextPageToken,
            )
            .execute()
        )

        for item in comment_response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        nextPageToken = comment_response.get("nextPageToken")

        if not nextPageToken:
            break

    return video_title, comments


classifier = pipeline(
    task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None
)


def process_model_output(model_output):
    if not model_output:  # Check if the list is not empty
        return None

    # Find the dictionary with the highest score
    max_label = max(model_output, key=lambda x: x["score"])

    # Extract the label from the dictionary
    label = max_label["label"]

    return label


@app.route("/")
def analyze_news():
    return "hello world"


@app.route("/analyze_sentiment", methods=["POST"])
def analyze_sentiment():
    # Extract video URL from the request
    video_url = request.json.get("video_url")

    # Replace 'YOUR_API_KEY' with your actual YouTube API key
    api_key = "AIzaSyDqsVQaYHzJ7Xjr3JDOZdv1lJwmShVMuNk"

    # Perform sentiment analysis
    video_title, comments = get_youtube_comments(video_url, api_key)
    sentiment_results = []

    for comment in comments:
        model_output = classifier(comment)
        sentiment_result = process_model_output(model_output)
        sentiment_results.append(sentiment_result)

    return jsonify({"video_title": video_title, "sentiment_results": sentiment_results})


if __name__ == "__main__":
    app.run(debug=True)
