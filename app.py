from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import numpy as np
import os
import threading
import pytesseract  # For OCR
import socket  # For checking internet connectivity

app = Flask(__name__)

# Variables to track progress
progress_status = {"percentage": 0}

# Load model for object detection
model_path = "models/"
net = cv2.dnn.readNetFromCaffe(
    os.path.join(model_path, "deploy.prototxt"),
    os.path.join(model_path, "mobilenet_iter_73000.caffemodel")
)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Create uploads directory
os.makedirs('uploads', exist_ok=True)


def check_internet():
    """Check if there is an active internet connection."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        file = request.files['image']
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        # Reset progress and start processing in a background thread
        progress_status["percentage"] = 0
        threading.Thread(target=process_image, args=(filepath, file.filename)).start()

        return jsonify({"status": "Processing started", "filename": file.filename})
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500


@app.route('/progress', methods=['GET'])
def get_progress():
    return jsonify(progress_status)


def process_image(filepath, filename):
    global progress_status

    try:
        progress_status["percentage"] = 10  # Start at 10%

        if not check_internet():
            # No internet, extract text instead of processing
            progress_status["percentage"] = 50
            extracted_text = extract_text(filepath)

            progress_status.update({
                "percentage": 100,
                "error": "No internet connection. Displaying extracted text.",
                "extracted_text": extracted_text,
            })
            return

        # Internet is available, process the image
        image = cv2.imread(filepath)
        (h, w) = image.shape[:2]

        # Preprocess image and run detection
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        progress_status["percentage"] = 30  # Progress to 30%

        detections = net.forward()
        progress_status["percentage"] = 50  # Progress to 50%

        # Draw bounding boxes
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the processed image as WebP
        output_path = os.path.join('uploads', f"processed_{os.path.splitext(filename)[0]}.webp")
        compression_quality = 90
        while True:
            cv2.imwrite(output_path, image, [cv2.IMWRITE_WEBP_QUALITY, compression_quality])
            if os.path.getsize(output_path) < 800 * 1024 or compression_quality <= 10:
                break
            compression_quality -= 5

        progress_status["percentage"] = 90  # Progress to 90%

        # Calculate sizes
        original_size_kb = os.path.getsize(filepath) / 1024
        processed_size_kb = os.path.getsize(output_path) / 1024

        progress_status.update({
            "percentage": 100,  # Complete
            "processed_image": f"/uploads/processed_{os.path.splitext(filename)[0]}.webp",
            "processed_image_size": round(processed_size_kb, 2),
            "original_image": f"/uploads/{filename}",  # Provide the original image path
            "original_image_size": round(original_size_kb, 2),
            "filename": filename,
        })

    except Exception as e:
        progress_status.update({"percentage": -1, "error": str(e)})


def extract_text(filepath):
    """Extract text from an image using Tesseract OCR."""
    try:
        image = cv2.imread(filepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)

        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)


if __name__ == '__main__':
    app.run(debug=True, port=8003)
