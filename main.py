import os
from flask import Flask, render_template, request
import base64
from main_calculation import Models

import json

app = Flask(__name__)
model = Models()
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route("/", methods=["POST", "GET", 'OPTIONS'])
def index_page():
	return render_template('index.html')

@app.route("/models")
def models():
	return render_template('models.html')

@app.route('/hook2', methods = ["GET", "POST", 'OPTIONS'])
def main_predict():
	"""
	Decodes image and uses it to make prediction.
	"""
	if request.method == 'POST':
		message = 'lets predict'
		image_b64 = request.values['imageBase64'].split(',')[1]
		image = base64.decodebytes(image_b64.encode('utf-8'))
		pred_label = model.predict_label(image)

	return json.dumps(pred_label) ##str(pred_label)


if __name__ == '__main__':
    app.run()