async function analyzeComments() {
  // Get the video URL from the input field
  const videoUrl = document.getElementById("video-url").value;

  // Check if the URL is empty
  if (!videoUrl) {
    alert("Please enter a YouTube video URL");
    return;
  }

  // Show the loader (replace this with your preferred loader implementation)
  const loader = document.getElementById("loader");
  loader.style.display = "block";

  // Add blur class to blurred container
  const blurredContainer = document.querySelector(".blurred-container");
  blurredContainer.classList.add("blurred");

  // Send a POST request to your server-side endpoint (assuming it's at /analyze_comments)
  const response = await fetch("/analyze_comments", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ video_url: videoUrl }),
  });

  loader.style.display = "none";

  // After processing response (optional):
  blurredContainer.classList.remove("blurred");

  // Check for errors in the response
  if (!response.ok) {
    alert("Error analyzing comments: " + response.statusText);
    return;
  }

  // Parse the JSON response
  const data = await response.json();

  // Display the analysis results (replace this with your own logic for displaying data)
  //   console.log("Video Title:", data.video_title);
  //   console.log("Sentiment Counts:", data.sentiment_counts);
  //   console.log("Sorted Comments:", data.sorted_comments);
  //   console.log("Summary by Sentiment:", data.summary_by_sentiment);
  console.log(data);

  // You can update the HTML content to display the results here
  // For example, by creating elements or populating existing elements
}

// Add an event listener to the button
document
  .getElementById("analyze-button")
  .addEventListener("click", analyzeComments);
