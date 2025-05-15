import cv2
import numpy as np

class DepthEstimator:
    def __init__(self, model_path="model-small.onnx"):
        """Инициализация модели оценки глубины (MiDaS)."""
        self.net = cv2.dnn.readNet(model_path)
        try:
            # Пытаемся использовать GPU (если OpenCV собран с CUDA)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except:
            # Используем CPU по умолчанию
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        # Настройки для small-версии MiDaS
        self.input_size = (256, 256)                          # размер входа модели
        self.mean = (123.675, 116.28, 103.53)                 # среднее для нормализации
        self.scale = 1/255.0

    def estimate_depth(self, frame):
        """
        Вычислить карту глубины для входного кадра.
        Возвращает 2D-массив глубин (нормированный 0..1 относительный).
        """
        blob = cv2.dnn.blobFromImage(frame, self.scale, self.input_size, self.mean, swapRB=True, crop=False)
        self.net.setInput(blob)
        depth_map = self.net.forward()                       # инференс модели MiDaS
        depth_map = depth_map[0, :, :]                       # убираем размерность батча
        depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
        depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return depth_map

    def estimate_distance(self, bbox, depth_map):
        """
        Оценить расстояние до объекта по его bbox и карте глубины.
        Берется значение в центре bbox.
        """
        x, y, w, h = bbox
        cx = int(x + w/2); cy = int(y + h/2)
        depth_val = float(depth_map[cy, cx])
        # При наличии калибровки depth_val можно перевести в метры
        return depth_val
