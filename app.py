import os
from flask import Flask, request, jsonify
from googleapiclient.discovery import build
from transformers import pipeline
import pandas as pd
from collections import Counter

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

    # Initialize variables to store the maximum score and corresponding label
    max_score = float("-inf")
    max_label = None

    # Iterate over each element in the model_output list
    for sublist in model_output:
        # If the element is not a list, skip it
        if not isinstance(sublist, list):
            continue

        # Iterate over each dictionary in the sublist
        for output in sublist:
            # If the element is not a dictionary, skip it
            if not isinstance(output, dict):
                continue

            # Extract the score and label from the current dictionary
            score = output.get("score", 0.0)
            label = output.get("label", "")

            # Update max_score and max_label if the current score is greater
            if score > max_score:
                max_score = score
                max_label = label

    return max_label


@app.route("/")
def analyze_news():
    return "hello world"


@app.route("/analyze_sentiment", methods=["POST"])
def analyze_sentiment():
    # Extract video URL from the request
    video_url = request.json.get("video_url")

    api_key = "asdf"

    # Perform sentiment analysis
    video_title, comments = get_youtube_comments(video_url, api_key)
    sentiment_results = []

    for comment in comments:
        model_output = classifier(comment)
        sentiment_result = process_model_output(model_output)
        sentiment_results.append(
            (comment, sentiment_result)
        )  # Include comment with sentiment result

    # Calculate sentiment summary
    sentiment_counts = Counter(
        sentiment_result for _, sentiment_result in sentiment_results
    )
    total_comments = len(sentiment_results)
    sentiment_summary = {
        "positive": sentiment_counts.get("positive", 0),
        "negative": sentiment_counts.get("negative", 0),
        "neutral": sentiment_counts.get("neutral", 0),
        "total_comments": total_comments,
    }

    # Calculate overall sentiment score
    positive_comments = sentiment_counts.get("positive", 0)
    negative_comments = sentiment_counts.get("negative", 0)
    overall_sentiment_score = (positive_comments - negative_comments) / total_comments

    # Construct the response
    response = {
        "video_title": video_title,
        "sentiment_summary": sentiment_summary,
        "overall_sentiment_score": overall_sentiment_score,
        "sentiment_results": sentiment_results,  # Include comments with sentiment results
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
