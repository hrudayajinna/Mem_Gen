from http.client import HTTPException
import os
from PIL import Image
from generator import MemeGenerator
from deephumor.models import (
    CaptioningLSTM, 
    CaptioningLSTMWithLabels, 
    CaptioningTransformerBase,
    CaptioningTransformer
)
from flask import Flask, request, redirect, url_for, flash, render_template, send_from_directory

app = Flask(__name__)
app.secret_key = 'thisisasecretkey'
UPLOAD_FOLDER = 'uploads'
MEMES_PREDICTIONS = 'memes_predictions'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MEMES_PREDICTIONS'] = MEMES_PREDICTIONS

FILE_TO_CLASS = {
    'LSTMDecoderWords.best.pth': CaptioningLSTM,
    'TransformerDecoderBaseWords.best.pth': CaptioningTransformerBase,
    'TransformerDecoderChars.best.pth': CaptioningTransformer
}

@app.route("/")
def main():
    models = list(FILE_TO_CLASS.keys())
    return render_template("index.html", models=models)

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files: 
        flash('No file part')
        return redirect(url_for('main'))

    file = request.files["file"]
    caption = request.form.get('caption', None)
    model_name = request.form.get('model', None)
    model = FILE_TO_CLASS[model_name].from_pretrained(f"./models/{model_name}")
    model.eval()

    if file.filename == "":
        flash("No selected file")
        return redirect(url_for('main'))
    
    fp = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(fp)

    generator = MemeGenerator(model=model, mode='word')
    pred = generator.generate(img_path=fp, caption=caption, T=1.3, beam_size=10, top_k=100)
    pred.save(os.path.join(app.config['MEMES_PREDICTIONS'], file.filename))
    return render_template("result.html", fp=url_for('serve_meme', filename=file.filename))

@app.route('/uploads/<filename>')
def serve_meme(filename):
    return send_from_directory(app.config['MEMES_PREDICTIONS'], filename)

@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException): return e
    print("An error occured: ", e)
    
    flash("An error occured, please retry")
    return redirect(url_for('main'))