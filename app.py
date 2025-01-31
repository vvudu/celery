import os
import uuid
import io
from flask import Flask, request, jsonify, send_file
from celery import Celery
import cv2
from cv2 import dnn_superres
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "processed_images"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Настройка Celery
app.config['CELERY_BROKER_URL'] = 'redis://redis:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://redis:6379/0'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Глобальная загрузка модели
scaler = dnn_superres.DnnSuperResImpl_create()
scaler.readModel("upscale/EDSR_x2.pb")
scaler.setModel("edsr", 2)

@celery.task(bind=True)
def upscale_task(self, image_bytes):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    result = scaler.upsample(image)
    
    # Сохраняем результат в память
    _, img_encoded = cv2.imencode('.png', result)
    return img_encoded.tobytes()

@app.route('/upscale', methods=['POST'])
def upscale_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    image_bytes = file.read()
    task = upscale_task.apply_async(args=[image_bytes])
    return jsonify({'task_id': task.id}), 202

@app.route('/tasks/<task_id>', methods=['GET'])
def get_task_status(task_id):
    task = upscale_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        return jsonify({'task_id': task_id, 'status': 'pending'}), 202
    elif task.state == 'SUCCESS':
        return jsonify({'task_id': task_id, 'status': 'completed', 'download_url': f'/processed/{task_id}.png'}), 200
    else:
        return jsonify({'task_id': task_id, 'status': task.state}), 500

@app.route('/processed/<task_id>.png', methods=['GET'])
def get_processed_image(task_id):
    task = upscale_task.AsyncResult(task_id)
    if task.state == 'SUCCESS':
        return send_file(io.BytesIO(task.result), mimetype='image/png')
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
