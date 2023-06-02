from flask import Flask, render_template, request
import os
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa
import uuid
from pydub import AudioSegment

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-audio', methods=['POST'])
def process_audio():
    audio_file = request.files['file']
    model_type = request.form.get('model')
    
    if not os.path.isdir('audio_files'):
        os.makedirs('audio_files')

    MODEL_PATH = "model-a"

    if model_type == 'model-b':
        MODEL_PATH = 'model-b'
        
    
    ruta_audio = "audio_files/"
    name_audio = str(uuid.uuid4()) + '.wav'
    audio_path = ruta_audio + name_audio

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_PATH)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="spanish", task="transcribe")

    audio_file.save(os.path.join('audio_files', name_audio))
    song = AudioSegment.from_file(audio_path)
    song.export(audio_path, format="wav")
    data, s = librosa.load(audio_path, sr=16000) 
    input_features = processor(data, sampling_rate=16000, return_tensors="pt").input_features 
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    os.remove(audio_path)

    return transcription

if __name__ == '__main__':
    app.run(debug=True)
