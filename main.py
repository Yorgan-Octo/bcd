import sys

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import io
import torch
import os
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))
app = FastAPI()

# Загрузка вашей обученной модели
model_path = 'yolov5/runs/train/bcd/weights/best.pt'  # Путь к вашей обученной модели
device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    from yolov5.models.common import DetectMultiBackend  # Импорт из YOLOv5
    model = DetectMultiBackend(model_path, device=device)
except Exception as e:
    print(f"Error loading model: {e}")
    raise HTTPException(status_code=500, detail="Error loading model")


#try:
    # Загрузка модели YOLOv5
 #   model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, device=device)
 #   print("Model loaded successfully")
#except Exception as e:
#    print(f"Error loading model: {e}")
 #   raise HTTPException(status_code=500, detail="Error loading model")

# Папка для сохранения изображений
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Чтение изображения из загруженного файла
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Генерация уникального имени для файла
        image_filename = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join(output_dir, image_filename)

        # Сохранение изображения в папку
        cv2.imwrite(image_path, image)

        # Выполнение предсказания модели на сохраненном изображении
        results = model(image_path)

        # Печать результатов (для отладки)
        results.print()

        # Отрисовка bounding box'ов на изображении
        results.render()  # Это обновляет изображение с отрисованными bounding box'ами

        # Получение обработанного изображения с отрисованными результатами
        rendered_image = results.ims[0]  # Извлечение обработанного изображения как NumPy массив

        # Преобразование NumPy массива в байтовый поток
        _, buffer = cv2.imencode('.png', rendered_image)  # Кодирование в PNG формат
        output_buffer = io.BytesIO(buffer)  # Создание буфера
        output_buffer.seek(0)  # Перемещаем указатель в начало буфера

        # Возвращаем изображение как поток
        return StreamingResponse(output_buffer, media_type="image/png")

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4000)
