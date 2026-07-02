import os
import cv2
import base64
import numpy as np
from flask import Flask, render_template_string, request, jsonify
from features import extract_all_features
from utils import load_model

app = Flask(__name__)

# Load the model once when the server starts
if os.path.exists("model.pkl"):
    print("Loading AI Model for Web Server...")
    model, scaler, threshold = load_model()
else:
    print("ERROR: model.pkl not found! Run train.py first.")
    model, scaler, threshold = None, None, 0.5

# The HTML, CSS, and JavaScript for the frontend UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Recapture Detection</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            text-align: center; 
            background: #1e1e2f; 
            color: #ffffff;
            margin: 0;
            padding: 20px;
        }
        h2 { color: #00d2ff; }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-top: 20px;
        }
        video { 
            border: 4px solid #333; 
            border-radius: 12px; 
            box-shadow: 0 8px 16px rgba(0,0,0,0.5);
            background: #000;
        }
        #result-box { 
            margin-top: 25px; 
            padding: 15px 40px;
            background: #2a2a40;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            font-size: 28px; 
            font-weight: bold; 
        }
        .FAKE { color: #ff4757; }
        .REAL { color: #2ed573; }
        .score { font-size: 16px; font-weight: normal; color: #a4b0be; margin-top: 5px; }
    </style>
</head>
<body>
    <h2>Live Anti-Spoofing AI Demo</h2>
    <p>Hold a phone screen or printed photo up to the camera to test.</p>
    
    <div class="container">
        <!-- Video Feed -->
        <video id="video" width="500" height="375" autoplay playsinline></video>
        
        <!-- Hidden Canvas for processing frames -->
        <canvas id="canvas" width="500" height="375" style="display:none;"></canvas>
        
        <!-- Result Display -->
        <div id="result-box">Waiting for camera...</div>
    </div>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const resultBox = document.getElementById('result-box');

        // Request Webcam Access
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => { video.srcObject = stream; })
            .catch(err => { resultBox.innerHTML = "<span class='FAKE'>Camera access denied!</span>"; });

        // Send a frame to the Python backend every 1 second
        setInterval(async () => {
            if (video.readyState === video.HAVE_ENOUGH_DATA) {
                // Draw current video frame to canvas
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                // Convert to Base64 JPEG
                const dataURL = canvas.toDataURL('image/jpeg', 0.8);

                try {
                    // Send to Flask server
                    const response = await fetch('/predict_live', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: dataURL })
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        console.error(data.error);
                    } else {
                        // Update UI
                        const colorClass = data.prediction.includes("FAKE") ? "FAKE" : "REAL";
                        resultBox.innerHTML = `
                            <div class="${colorClass}">${data.prediction}</div>
                            <div class="score">Fake Probability: ${(data.score * 100).toFixed(1)}%</div>
                        `;
                    }
                } catch (e) {
                    console.error("Network Error", e);
                }
            }
        }, 1000); // 1000 ms = 1 frame per second
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serves the main HTML webpage."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict_live', methods=['POST'])
def predict_live():
    """Receives base64 image from webcam, runs the model, returns JSON."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.json
        image_data = data['image'].split(',')[1] # Remove the "data:image/jpeg;base64," prefix
        
        # Save temp file so we don't have to rewrite your features.py script
        temp_filename = "temp_webcam.jpg"
        with open(temp_filename, "wb") as fh:
            fh.write(base64.b64decode(image_data))
            
        # Extract features and predict
        features = extract_all_features(temp_filename).reshape(1, -1)
        features_scaled = scaler.transform(features)
        score = model.predict_proba(features_scaled)[0, 1]
        
        prediction = "FAKE / SCREEN" if score >= threshold else "REAL PHOTO"
        
        # Cleanup temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        return jsonify({
            'prediction': prediction,
            'score': float(score)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n--- Starting Live Web Demo ---")
    print("Open your browser and go to: http://127.0.0.1:5001\n")
    # Changed host to 127.0.0.1 and port to 5001 to avoid Apple AirPlay conflicts
    app.run(host='127.0.0.1', port=5001, debug=False)