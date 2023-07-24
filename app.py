from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from roboflow import Roboflow

app = Flask(__name__)

# Replace with your Roboflow API key
API_KEY = "pZdTdiT33baKyejAq2Db"

# Initialize Roboflow with the API key
rf = Roboflow(api_key=API_KEY)

# Specify the project and model you want to use
project_name = "skin-cancer-detection-wfldq"
model_version = 3

# Get the project and model
project = rf.workspace().project(project_name)
model = project.version(model_version).model

def classify_image_local(image_path):
    # Infer on a local image
    prediction_response = model.predict(image_path)
    return prediction_response.json()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/', methods=['GET', 'POST'])
def upload_and_classify():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            file.save(filepath)

            result = classify_image_local(filepath)
            os.remove(filepath)  # Remove the uploaded file after classification

            return jsonify(result)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
