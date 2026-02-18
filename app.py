import os
import uuid
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, render_template, request, send_file
from moviepy import VideoFileClip, concatenate_videoclips
import csv

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
CLIP_PADDING = 4
ML_CONFIDENCE = 0.22

HIGHLIGHT_SOUNDS = [
    'Gunshot, gunfire', 'Explosion', 'Machine gun', 'Boom',
    'Laughter', 'Chuckle', 'Shout', 'Yell', 'Screaming', 'Clapping', 'Cheering'
]

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YAMNet model once
print("Loading YAMNet model...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = yamnet_model.class_map_path().numpy()

class_names = []
with open(class_map_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        class_names.append(row['display_name'])

print("Model Loaded Successfully!")

# -------------- ML FUNCTION ----------------
def generate_stream_highlights(video_path):
    with VideoFileClip(video_path) as video:

        found_moments = []

        for start_t in range(0, int(video.duration), 1):
            end_t = min(start_t + 1.5, video.duration)
            chunk = video.subclipped(start_t, end_t)

            if chunk.audio is None:
                continue

            audio_data = chunk.audio.to_soundarray(fps=16000)

            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            scores, _, _ = yamnet_model(audio_data.astype(np.float32))
            mean_scores = np.mean(scores, axis=0)

            top_idx = np.argmax(mean_scores)
            label = class_names[top_idx]
            conf = mean_scores[top_idx]

            if label in HIGHLIGHT_SOUNDS and conf > ML_CONFIDENCE:
                found_moments.append([
                    max(0, start_t - CLIP_PADDING),
                    min(video.duration, end_t + CLIP_PADDING)
                ])

        if not found_moments:
            return None

        # Merge overlaps
        merged = []
        curr_s, curr_e = found_moments[0]

        for next_s, next_e in found_moments[1:]:
            if next_s < curr_e:
                curr_e = max(curr_e, next_e)
            else:
                merged.append((curr_s, curr_e))
                curr_s, curr_e = next_s, next_e

        merged.append((curr_s, curr_e))

        final_clips = [video.subclipped(s, e) for s, e in merged]
        final_video = concatenate_videoclips(final_clips)

        unique_name = f"highlight_{uuid.uuid4().hex}.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, unique_name)

        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=30
        )

        return output_path

# -------------- FLASK APP ----------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return "No file uploaded"

    file = request.files["video"]

    if file.filename == "":
        return "No selected file"

    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(filepath)

    output_path = generate_stream_highlights(filepath)

    if output_path is None:
        return "No highlights detected. Try lowering ML_CONFIDENCE."

    return send_file(output_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

