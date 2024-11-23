const video = document.getElementById("video");
const statusDiv = document.getElementById("status");
const questionDiv = document.getElementById("question");

// Load Face API models
Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
    faceapi.nets.faceExpressionNet.loadFromUri('/models')
]).then(startVideo);

// Start video feed
function startVideo() {
    navigator.mediaDevices.getUserMedia({ video: {} })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(err => console.error("Error accessing webcam:", err));
}

// Analyzing the face every few seconds
video.addEventListener("play", () => {
    const canvas = faceapi.createCanvasFromMedia(video);
    document.body.append(canvas);
    const displaySize = { width: video.width, height: video.height };
    faceapi.matchDimensions(canvas, displaySize);

    setInterval(async () => {
        const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceExpressions();
        const resizedDetections = faceapi.resizeResults(detections, displaySize);
        
        canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
        faceapi.draw.drawDetections(canvas, resizedDetections);
        faceapi.draw.drawFaceExpressions(canvas, resizedDetections);

        // Check if face is detected
        if (detections.length === 0) {
            statusDiv.innerText = "Face out of frame!";
            askFocusQuestion();
        } else {
            const expressions = detections[0].expressions;
            statusDiv.innerText = "Face detected. Analyzing mood...";

            const mood = analyzeMood(expressions);
            if (mood) {
                askMoodQuestion(mood);
            }
        }
    }, 2000);
});

// Function to analyze mood based on expressions
function analyzeMood(expressions) {
    const { happy, sad, angry, surprised } = expressions;
    if (happy > 0.6) return "happy";
    if (sad > 0.6) return "sad";
    if (angry > 0.6) return "angry";
    if (surprised > 0.6) return "surprised";
    return null;
}

// Ask a focus question if the user moves out of the frame
function askFocusQuestion() {
    questionDiv.innerText = "We noticed you're not focused. What caused the distraction?";
}

// Ask a mood-related question
function askMoodQuestion(mood) {
    let question = "";
    switch (mood) {
        case "happy":
            question = "You seem happy! What made you smile?";
            break;
        case "sad":
            question = "You seem a bit down. Want to share what’s on your mind?";
            break;
        case "angry":
            question = "Feeling frustrated? Anything you’d like to talk about?";
            break;
        case "surprised":
            question = "You look surprised! Did something unexpected happen?";
            break;
    }
    questionDiv.innerText = question;
}
