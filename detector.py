import cv2
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5):
        """Инициализация детектора YOLOv8."""
        self.model = YOLO(model_path)          # загрузка модели
        self.conf_threshold = conf_threshold
        self.class_names = (self.model.names if hasattr(self.model, 'names') else None)

    def detect(self, frame):
        """
        Выполнить детекцию объектов на кадре. 
        Возвращает список кортежей (x, y, w, h, class_id, conf).
        """
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.predict(img, conf=self.conf_threshold, verbose=False)
        detections = []
        if results and len(results) > 0:
            for det in results[0].boxes.data.tolist():  # извлекаем результаты из тензора
                x1, y1, x2, y2, conf, cls_id = det
                if conf < self.conf_threshold:
                    continue
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                detections.append((x, y, w, h, int(cls_id), float(conf)))
        return detections

    def draw_detections(self, frame, detections):
        """Нарисовать боксы и метки классов на изображении для визуализации."""
        for (x, y, w, h, class_id, conf) in detections:
            class_name = self.class_names[class_id] if self.class_names else str(class_id)
            color = (0, 255, 0)  # цвет рамки (зеленый)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"{class_name}: {conf:.2f}"
            # фон подписи
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - text_h - 2), (x + text_w, y), color, cv2.FILLED)
            # текст метки
            cv2.putText(frame, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return frame
