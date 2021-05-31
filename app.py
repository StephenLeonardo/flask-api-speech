from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World!'



from google.cloud import speech
import speech_recognition as sr
client = speech.SpeechClient()
r = sr.Recognizer()

import json
import io
import os
import pathlib
from urllib.request import urlopen
from data_tools import scaled_in, inv_scaled_ou
from data_tools import audio_files_to_numpy, numpy_audio_to_matrix_spectrogram, matrix_spectrogram_to_numpy_audio, export_audio_file

import librosa
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.models import model_from_json



physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Sample rate chosen to read audio
sample_rate = 8000
# Minimum duration of audio files to consider
min_duration = 1.0
# Our training data will be frame of slightly above 1 second
frame_length = 8064
# hop length for clean voice files separation (no overlap)
hop_length_frame = 8064
# hop length for noise files (we split noise into several windows)
hop_length_frame_noise = 5000

loaded_model = tf.keras.models.load_model('my_model3')


def callGoogleAPI():
    # gcs_uri = "gs://cloud-samples-data/speech/brooklyn_bridge.raw"
    cwd = os.getcwd()
    filepath = os.path.join(cwd, 'exported', 'test.wav')

    harvard = sr.AudioFile(filepath)
    with harvard as source:
       audio = r.record(source)

    transcript = r.recognize_google(audio, language='id-ID')

    # with io.open(filepath, "rb") as audio_file:
    #     content = audio_file.read()
    #
    # audio = speech.RecognitionAudio(content=content)
    #
    #
    # config = speech.RecognitionConfig(
    #     sample_rate_hertz=8000,
    #     language_code="id-ID",
    # )
    #
    # response = client.recognize(config=config, audio=audio)
    #
    # transcript = ''
    #
    # for result in response.results:
    #     transcript += str(result.alternatives[0].transcript) + ' '
        # The first alternative is the most likely one for this portion.
        # print(u"Transcript: {}".format(result.alternatives[0].transcript))

    return transcript


@app.route('/test-speech', methods=['GET'])
def testSpeech():
    print("heehehehe")
    cwd = os.getcwd()
    filepath = os.path.join(cwd, 'exported', 'Recording.wav')


    harvard = sr.AudioFile(filepath)
    with harvard as source:
       audio = r.record(source)

    transcript = r.recognize_google(audio, language='id-ID')
    # content = None
    # with io.open(filepath, "rb") as audio_file:
    #     content = audio_file.read()
    #
    # audio = speech.RecognitionAudio(content=content)
    #
    #
    # config = speech.RecognitionConfig(
    #     encoding='AMR',
    #     sample_rate_hertz=8000,
    #     language_code="id-ID",
    # )
    #
    # response = client.recognize(config=config, audio=audio)

    # transcript = ''

    # for result in response.results:
    #     transcript += str(result.alternatives[0].transcript) + ' '
        # The first alternative is the most likely one for this portion.
        # print(u"Transcript: {}".format(result.alternatives[0].transcript))

    return transcript



@app.route('/predict', methods=['GET'])
def predict():

    # request_json = request.json
    # print("data: {}".format(request_json))
    # print("type: {}".format(type(request_json)))
    # user = request.args.get('user')
    # sudah termasuk link public untuk filenya
    filename = request.args.get('filename', None)
    file_url =  request.args.get('file_url', None)

    print(filename)
    print(file_url)

    print('woohoo')
    # Url untuk cloud storage
    # dirname = 'https://storage.googleapis.com/ezu-storage/'
    #
    #
    # full_name = dirname + filename

    #
    print("oke")
    z = io.BytesIO(urlopen(file_url).read())


    print("yep")

    filepath_to_save = './assets/' + filename
    pathlib.Path((filepath_to_save)).write_bytes(z.getbuffer())

    print("yess")

    # X, sample_rate = librosa.load(filepath)


    audio = audio_files_to_numpy(filepath_to_save, sample_rate,
                             frame_length, hop_length_frame, min_duration)

    n_fft = 255
    hop_length_fft = 63
    dim_square_spec = int(n_fft / 2) + 1

    m_amp_db_audio,  m_pha_audio = numpy_audio_to_matrix_spectrogram(
    audio, dim_square_spec, n_fft, hop_length_fft)




    #global scaling to have distribution -1/1
    X_in = scaled_in(m_amp_db_audio)
    #Reshape for prediction
    X_in = X_in.reshape(X_in.shape[0],X_in.shape[1],X_in.shape[2],1)
    #Prediction using loaded network
    X_pred = loaded_model.predict(X_in)
    #Rescale back the noise model
    inv_sca_X_pred = inv_scaled_ou(X_pred)
    #Remove noise model from noisy speech
    X_denoise = m_amp_db_audio - inv_sca_X_pred[:,:,:,0]
    #Reconstruct audio from denoised spectrogram and phase
    audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha_audio, frame_length, hop_length_fft)




    output_file = (audio_denoise_recons.flatten() * 100)
    export_filename = 'test.wav'
    # url untuk export audio yg udah dijernihin ke cloud storage
    export_path = './exported/' + export_filename
    sf.write(export_path, output_file, sample_rate)

    translation = 'selamat pagi'

    transcript = callGoogleAPI()

    if transcript is not None and transcript != '':
        translation = transcript

    #
    response_json = {
        "filename" : request.args.get('filename', None),
        "file_url" : request.args.get("file_url", None),
        "translation" : str(translation)
    }



    return json.dumps(response_json)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
