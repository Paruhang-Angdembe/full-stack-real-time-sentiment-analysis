document
  .getElementById("analyze-button")
  .addEventListener("click", function () {
    var videoUrl = document.getElementById("video-url").value;
    // Log the YouTube URL to the console
    console.log("YouTube URL:", videoUrl);
    var formData = new FormData();
    formData.append("video_url", videoUrl);

    // fetch("http://127.0.0.1:5000/analyze_comments", {
    //   method: "POST",
    //   body: formData,
    // })
    //   .then((response) => response.json())
    //   .then((data) => {
    //     document.getElementById("result").innerHTML = JSON.stringify(
    //       data,
    //       null,
    //       2
    //     );
    //   })
    //   .catch((error) => console.error("Error:", error));
  });

function displayResults(data) {
  const resultsDiv = document.getElementById("results");
  resultsDiv.innerHTML = `
        <h2>Video Title: ${data.video_title}</h2>
        <p>Unique Commenters Count: ${data.unique_commenters_count}</p>
        <div id="sentimentCounts"></div>
        <div id="sortedComments"></div>
        <div id="summaries"></div>
        <button onclick="downloadCSV()">Download CSV</button>
    `;

  // Populate Sentiment Counts
  const sentimentCountsDiv = document.getElementById("sentimentCounts");
  sentimentCountsDiv.innerHTML =
    "<h3>Sentiment Counts:</h3>" +
    data.sentiment_counts
      .map((count) => `<p>${count.Sentiment}: ${count.Count}</p>`)
      .join("");

  // Populate Sorted Comments
  const sortedCommentsDiv = document.getElementById("sortedComments");
  sortedCommentsDiv.innerHTML =
    "<h3>Sorted Comments:</h3>" +
    data.sorted_comments
      .map(
        (comment) =>
          `<p>${comment.SerialNo}. ${comment.Label}: ${comment.Comment}</p>`
      )
      .join("");

  // Populate Summaries
  const summariesDiv = document.getElementById("summaries");
  summariesDiv.innerHTML =
    "<h3>Summaries:</h3>" +
    Object.entries(data.summary_by_sentiment)
      .map(
        ([sentiment, summary]) => `<p>Summary for ${sentiment}: ${summary}</p>`
      )
      .join("");
}

function downloadCSV() {
  // Since we're using POST, the server expects the sorted_comments data
  const sortedComments = document.getElementById("sortedComments").innerText;
  fetch("http://127.0.0.1:5000/download_csv", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ sorted_comments: sortedComments }),
  })
    .then((response) => response.blob())
    .then((blob) => {
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "comments_analysis.csv");
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
      window.URL.revokeObjectURL(url);
    })
    .catch((error) => console.error("Error:", error));
}
