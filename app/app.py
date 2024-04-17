from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from transformers import pipeline, TFPegasusForConditionalGeneration, AutoTokenizer
import pandas as pd
import os
import re
from googleapiclient.discovery import build


app = Flask(__name__)
CORS(app)

classifier = pipeline(
    task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None
)

# Initialize the summarization model and tokenizer
tokenizer = None
model = None
cached_model = None
cached_tokenizer = None


def load_transformer_models():

    global tokenizer, model, cached_tokenizer, cached_model

    if cached_tokenizer and cached_model:
        tokenizer = cached_tokenizer
        model = cached_model
        return

    # Check if model and tokenizer are saved, if not, load from pretrained
    if os.path.exists("saved_models/my_tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(
            "saved_models/my_tokenizer", local_files_only=True
        )
        model = TFPegasusForConditionalGeneration.from_pretrained(
            "saved_models/my_model", local_files_only=True
        )

    cached_tokenizer = tokenizer
    cached_model = model


# Load transformer models during application startup
# load_transformer_models()


def get_max_score_label(data_list):
    if not data_list:  # Check if the list is not empty
        return None

    max_label = max(data_list, key=lambda x: x["score"])
    return max_label["label"]


def preprocess_comments(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = text.encode("ascii", "ignore").decode("ascii")  # Remove non-ASCII characters
    return text


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
                maxResults=100,
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


@app.route("/analyze_comments", methods=["POST"])
def analyze_comments():

    video_url = request.json.get("video_url")

    api_key = "AIzaSyDOOVneXYot-RYzGBZRAc3IufC7d5kwz-A"

    video_title, comments = get_youtube_comments(video_url, api_key)
    sentiment_results = []

    df = pd.DataFrame(columns=["SerialNo", "Label", "Comment"])

    # Lazy Loading
    if not tokenizer or not model:
        load_transformer_models()

    batch_results = []

    # Batch Processing
    batch_size = 10
    num_batches = len(comments) // batch_size + 1

    for batch_num in range(num_batches):
        batch_comments = comments[batch_num * batch_size : (batch_num + 1) * batch_size]

        for i, comment in enumerate(batch_comments, start=batch_num * batch_size + 1):
            model_output = classifier(comment)
            max_label = get_max_score_label(model_output[0])
            batch_results.append(
                {"SerialNo": i, "Label": max_label, "Comment": comment}
            )

    df = pd.DataFrame(batch_results)

    print(df.info())
    print(df)

    # Convert sentiment counts to a list of dictionaries
    sentiment_counts = df["Label"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    sentiment_counts_list = sentiment_counts.to_dict(orient="records")

    # Sort the DataFrame by label
    df_sorted = df.sort_values(by="Label")
    sorted_comments_list = df_sorted.to_dict(orient="records")

    # Summarize comments by sentiment
    comments_by_sentiment = df.groupby("Label")["Comment"].apply(" ".join).to_dict()
    summary_by_sentiment = {}

    for sentiment, comments in comments_by_sentiment.items():
        clean_comments = preprocess_comments(comments)
        inputs = tokenizer(
            clean_comments, max_length=1024, truncation=True, return_tensors="tf"
        )
        summary_ids = model.generate(inputs["input_ids"])
        summary_text = tokenizer.decode(
            summary_ids[0].numpy(), skip_special_tokens=True
        )
        summary_by_sentiment[sentiment] = summary_text

    # Return the results as JSON
    return jsonify(
        {
            "video_title": video_title,
            "sentiment_counts": sentiment_counts_list,
            "sorted_comments": sorted_comments_list,
            "summary_by_sentiment": summary_by_sentiment,
        }
    )


@app.route("/download_csv", methods=["POST"])
def download_csv():
    data = request.json
    df = pd.DataFrame(data["sorted_comments"])
    csv = df.to_csv(index=False)
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=comments_analysis.csv"},
    )


@app.route("/")
def home():
    return "hello"


if __name__ == "__main__":
    app.run()
