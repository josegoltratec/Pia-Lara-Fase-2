
from flask import Flask, render_template, request

app = Flask(__name__)



@app.route("/process-audio", methods=['POST'])
def process_audio():
    return 'jose'


@app.route("/")
def index():
    return render_template('index.html')