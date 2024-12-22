from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import tempfile
import os

app = Flask(__name__)
CORS(app)

# Load YOLO model
model = YOLO('yolov8l-world.pt')
model.set_classes(["helmet", "gloves", "glasses", "masks", "goggles"])

# Mapping YOLO labels to desired count labels
LABEL_MAPPING = {
    "glasses": "goggles",
    "helmet": "helmet",
    "gloves": "gloves",
    "masks": "masks",
}

def process_image(image):
    # Perform YOLO inference
    results = model.predict(image, device='cpu')[0]
    counts = {cls: 0 for cls in ["helmet", "gloves", "masks", "goggles"]}

    for det in results.boxes:
        label = model.names[int(det.cls[0])]  # Get the detected class label
        mapped_label = LABEL_MAPPING.get(label)  # Map YOLO label to counts label
        if mapped_label in counts:
            counts[mapped_label] += 1

    # Annotate image with bounding boxes
    annotated_image = results.plot()
    _, buffer = cv2.imencode('.jpg', annotated_image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')  # Encode image for frontend display

    return counts, encoded_image

def process_video(video_path):
    # Initialize counters
    counts = {cls: 0 for cls in ["helmet", "gloves", "masks", "goggles"]}
    annotated_frames = []

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video_path = os.path.join(tempfile.gettempdir(), "output_video.mp4")
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform YOLO inference on the frame
        results = model.predict(frame, device='cpu')[0]

        for det in results.boxes:
            label = model.names[int(det.cls[0])]  # Get the detected class label
            mapped_label = LABEL_MAPPING.get(label)  # Map YOLO label to counts label
            if mapped_label in counts:
                counts[mapped_label] += 1

        # Annotate the frame
        annotated_frame = results.plot()
        if out is None:
            # Initialize video writer with the first frame's dimensions
            height, width, _ = annotated_frame.shape
            out = cv2.VideoWriter(temp_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

        # Write the annotated frame to the output video
        out.write(annotated_frame)

    # Release video resources
    cap.release()
    if out:
        out.release()

    # Read and encode the processed video
    with open(temp_video_path, "rb") as video_file:
        encoded_video = base64.b64encode(video_file.read()).decode('utf-8')

    return counts, encoded_video

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'success': False, 'message': 'No file uploaded.'}), 400

    # Save the file temporarily
    temp_file_path = os.path.join(tempfile.gettempdir(), file.filename)
    file.save(temp_file_path)

    # Check if the file is an image or a video
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
        # Process as an image
        np_image = cv2.imdecode(np.fromfile(temp_file_path, np.uint8), cv2.IMREAD_COLOR)
        if np_image is None:
            return jsonify({'success': False, 'message': 'Invalid image file.'}), 400

        counts, encoded_image = process_image(np_image)
        os.remove(temp_file_path)

        return jsonify({
            'success': True,
            'type': 'image',
            'helmets': counts['helmet'],
            'gloves': counts['gloves'],
            'masks': counts['masks'],
            'goggles': counts['goggles'],
            'image': encoded_image
        })
    elif file_extension in ['mp4', 'avi', 'mov', 'mkv']:
        # Process as a video
        counts, encoded_video = process_video(temp_file_path)
        os.remove(temp_file_path)

        return jsonify({
            'success': True,
            'type': 'video',
            'helmets': counts['helmet'],
            'gloves': counts['gloves'],
            'masks': counts['masks'],
            'goggles': counts['goggles'],
            'video': encoded_video
        })
    else:
        return jsonify({'success': False, 'message': 'Unsupported file type.'}), 400

if __name__ == '__main__':
    app.run(debug=True)