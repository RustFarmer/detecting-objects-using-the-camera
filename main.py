from tkinter import *
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import urllib.request
import sys
from collections import defaultdict

# Класс для загрузки файлов YOLO
class YOLOFileDownloader:
    BASE_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/"
    
    @staticmethod
    def download_file(url, filename):
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print("Download completed!")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                return False
        return True

    @staticmethod
    def ensure_yolo_files():
        files = {
            "yolov4-tiny.cfg": YOLOFileDownloader.BASE_URL + "cfg/yolov4-tiny.cfg",
            "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
            "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
        }
        
        for filename, url in files.items():
            if not os.path.exists(filename):
                if not YOLOFileDownloader.download_file(url, filename):
                    return False
        return True

# Класс для обнаружения объектов
class ObjectDetector:
    def __init__(self):
        # Проверяем наличие файлов
        if not YOLOFileDownloader.ensure_yolo_files():
            raise FileNotFoundError("Required YOLO files are missing")
            
        # Пути к файлам модели YOLO
        self.config_path = "yolov4-tiny.cfg"
        self.weights_path = "yolov4-tiny.weights"
        self.classes_path = "coco.names"
        
        # Загрузка модели YOLO
        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Загрузка названий классов
        with open(self.classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Получение выходных слоев сети
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Для отслеживания уникальных объектов
        self.last_detected = set()

    def detect_objects(self, frame, conf_threshold=0.5, nms_threshold=0.4):
        height, width = frame.shape[:2]
        current_detections = set()
        
        # Подготовка изображения для нейросети
        blob = cv2.dnn.blobFromImage(
            frame, 
            1/255.0, 
            (416, 416), 
            swapRB=True, 
            crop=False
        )
        
        # Подача изображения в сеть
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layers)
        
        # Обработка результатов
        boxes = []
        confidences = []
        class_ids = []
        
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > conf_threshold:
                    # Масштабирование координат bounding box
                    box = detection[0:4] * np.array([width, height, width, height])
                    (center_x, center_y, w, h) = box.astype("int")
                    
                    # Расчет координат углов прямоугольника
                    x = int(center_x - (w / 2))
                    y = int(center_y - (h / 2))
                    
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    
                    # Добавляем название класса в текущие обнаружения
                    current_detections.add(self.classes[class_id])
        
        # Применение Non-Maxima Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        
        # Рисование bounding boxes и подписей
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y, w, h) = boxes[i]
                class_name = self.classes[class_ids[i]]
                confidence = confidences[i]
                
                label = f"{class_name}: {confidence:.2f}"
                color = (0, 255, 0)  # Зеленый цвет
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Выводим названия предметов при изменении
        self.print_detected_objects(current_detections)
        
        return frame

    def print_detected_objects(self, current_detections):
        # Проверяем, изменился ли набор обнаруженных объектов
        if current_detections != self.last_detected:
            self.last_detected = current_detections
            
            # Формируем список для вывода
            if current_detections:
                items = ", ".join(sorted(current_detections))
                print(f"Обнаружены: {items}")
            else:
                print("Объекты не обнаружены")

# Основное приложение
def main():
    try:
        # Попытка инициализации детектора
        detector = ObjectDetector()
    except Exception as e:
        print(f"Initialization error: {e}")
        # Создаем простое GUI с сообщением об ошибке
        error_app = Tk()
        error_app.title("Error")
        Label(error_app, text=f"Failed to initialize object detector:\n{str(e)}", 
              fg="red", padx=20, pady=20).pack()
        Button(error_app, text="Exit", command=error_app.destroy).pack(pady=10)
        error_app.mainloop()
        return
    
    # Создание объекта видеозахвата
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        print("Error: Could not open camera")
        return
        
    width, height = 800, 600
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Создание GUI
    app = Tk()
    app.title("Object Detection Camera")
    app.bind('<Escape>', lambda e: app.quit())

    label_widget = Label(app)
    label_widget.pack()

    # Для отслеживания FPS
    frame_count = 0
    start_time = cv2.getTickCount()
    
    def open_camera():
        nonlocal frame_count, start_time
        
        ret, frame = vid.read()
        if not ret:
            print("Failed to capture frame")
            app.after(10, open_camera)
            return
            
        # Обнаружение объектов на кадре
        try:
            processed_frame = detector.detect_objects(frame)
        except Exception as e:
            print(f"Detection error: {e}")
            processed_frame = frame
        
        # Расчет FPS
        frame_count += 1
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        fps = frame_count / elapsed_time
        
        # Отображение FPS на кадре
        cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Конвертация для отображения в Tkinter
        opencv_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGBA)
        captured_image = Image.fromarray(opencv_image)
        photo_image = ImageTk.PhotoImage(image=captured_image)
        
        # Обновление изображения в интерфейсе
        label_widget.photo_image = photo_image
        label_widget.configure(image=photo_image)
        app.after(10, open_camera)

    # Кнопки управления
    button_frame = Frame(app)
    button_frame.pack(fill=X)

    Button(button_frame, text="Open Camera", command=open_camera).pack(side=LEFT, padx=10, pady=5)
    Button(button_frame, text="Exit", command=app.destroy).pack(side=RIGHT, padx=10, pady=5)

    app.mainloop()

    # Освобождение ресурсов при закрытии
    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
