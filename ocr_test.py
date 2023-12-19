from flask import Flask, render_template, request, jsonify,Response
from flask_socketio import SocketIO, emit
import io
import os
from google.cloud import vision
import cv2
import base64
from google.cloud.vision_v1 import types
from pykakasi import kakasi

app = Flask(__name__)
socketio = SocketIO(app)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\三好航馬\\Desktop\\チーム制作testfile\\teamwork-project-406308-6e3455ceaad5.json"
cap = cv2.VideoCapture(0)
client = vision.ImageAnnotatorClient()

# フォルダが存在しない場合に作成する
image_folder = 'images'
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

kakasi_instance = kakasi()
kakasi_instance.setMode('J', 'H')  # 漢字をひらがなに変換する設定
conv = kakasi_instance.getConverter()

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = base64.b64encode(buffer)
            frame_str = frame_bytes.decode('utf-8')
            
            socketio.emit('video_frame',{'frame':frame_str},namespace='/video')
            socketio.sleep(0.1)
            
def capture_image():
    ret, frame = cap.read()

    # 新しいファイル名を生成（例: captured_image.png）
    file_name = os.path.join(image_folder, 'captured_image.png')

    # 画像を保存
    cv2.imwrite(file_name, frame)

    return file_name

def base64_encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read())
    return encoded_image.decode('utf-8')

#文字認識を行い得られた文字をひらがなに変換している
def detect_and_convert_text(image_path):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        detected_text = texts[0].description
        
        original_text = detected_text

        # 検出された文字列をひらがなに変換
        converted_text = conv.do(detected_text)
        
        return converted_text
    else:
        return 'No text found.'
    
@app.route('/')
def title():
    return render_template('title.html')


@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture')
def capture():
    # Capture image from camera and save it
    image_path = capture_image()
    
    # Encode image to base64
    encoded_image = base64_encode_image(image_path)

    return jsonify({'image': encoded_image, 'file_path': image_path})

@app.route('/result')
def result():
    return render_template('result.html')
    

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('image_request')
def handle_image_request():
    while True:
        ret, frame = cap.read()
        _, buffer = cv2.imencode('.jpg', frame)
        image_str = base64.b64encode(buffer)
        image_data = image_str.decode('utf-8')
        emit('image_response', {'image': image_data})
        socketio.sleep(0.1)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file_path' not in request.form:
        return jsonify({'error': 'No image file path'})

    # Get the file path from the request
    image_path = request.form['file_path']

    # Perform text detection on the image and convert to hiragana
    text_result = detect_and_convert_text(image_path)

    return jsonify({'result': text_result})

@socketio.on('connect', namespace='/video')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect', namespace='/video')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True)
