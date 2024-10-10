import torch

# Загрузка вашей обученной модели
model_path = 'yolov5/runs/train/bcd/weights/best.pt'  # Путь к вашей обученной модели
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Загрузка модели
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, device=device)

# Инференс на изображении
img = '12.jpg'  # Укажите путь к изображению для инференса
results = model(img)

# Вывод результатов
results.print()  # Печать результатов
results.show()   # Отобразить результаты
results.save()   # Сохранить результаты в файл