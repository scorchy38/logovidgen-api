import io
import os
import json
import requests
import firebase_admin
from datetime import timedelta
from PIL import Image
import moviepy.editor as mpy
from firebase_admin import credentials, storage
from flask import Flask, request, jsonify
from gradio_client import Client

cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'logo-video-generator.appspot.com'
})


def create_app():
    app = Flask(__name__)

    @app.route('/lineart', methods=['POST'])
    def handle_lineart_request():
        data = request.json

        urls = []
        logo_url = data['imageUrl']

        audio_url = data['audioUrl']
        text_clip_content = data['content']
        audio_data = requests.get(audio_url).content
        with open('background_audio.mp3', 'wb') as audio_file:
            audio_file.write(audio_data)

        client = Client("https://edden-app-controlnet-v1-1.hf.space/",
                        hf_token="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")

        for prompt in data['prompts']:
            dir_path = client.predict(
                data['imageUrl'],
                prompt,
                data['additionalPrompt'],
                data['negativePrompt'],
                data['numOfImages'],
                data['imageRes'],
                data['imageRes'],
                data['numOfSteps'],
                data['guidanceScale'],
                data['seed'],
                data['preprocessor'],
                api_name="/lineart"
            )

            json_path = os.path.join(dir_path, 'captions.json')

            if os.path.exists(json_path):
                with open(json_path, 'r') as json_file:
                    captions = json.load(json_file)

                image_path = list(captions.keys())[-1]

                if os.path.exists(image_path):
                    with open(image_path, 'rb') as image_file:
                        image_data = image_file.read()

                    blob = storage.bucket().blob('generated/' + prompt)
                    blob.upload_from_string(
                        image_data, content_type='image/png')

                    download_url = blob.generate_signed_url(
                        timedelta(seconds=300), method='GET')

                    urls.append(download_url)

        clips = []
        for url in urls:
            img_data = requests.get(url).content
            img = Image.open(io.BytesIO(img_data))
            img.save('temp.png')

            clip = mpy.ImageClip('temp.png').set_duration(1)
            clip = clip.crossfadein(0.1).crossfadeout(0.1)
            clips.append(clip)

        txt_clip = mpy.TextClip(text_clip_content, fontsize=24, color='white')
        txt_clip = txt_clip.set_duration(2).crossfadein(0.5)
        clips.append(txt_clip)

        video = mpy.concatenate_videoclips(clips, method='compose')

        audio = mpy.AudioFileClip('background_audio.mp3')

        video = video.set_audio(audio)
        video.write_videofile('output.mp4', fps=24)

        with open('output.mp4', 'rb') as video_file:
            video_data = video_file.read()

        blob = storage.bucket().blob('videos/output.mp4')
        blob.upload_from_string(video_data, content_type='video/mp4')

        video_url = blob.generate_signed_url(
            timedelta(seconds=3000), method='GET')

        return jsonify({'video_url': video_url})

    return app
