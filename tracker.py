import numpy as np

class KalmanFilter2D:
    def __init__(self, dt=1.0, accel_std=1.0, meas_std=1.0):
        # Инициализация параметров фильтра Калмана (4x4 матрицы)
        self.x = np.zeros((4, 1))                      # вектор состояния [X, Y, Vx, Vy]
        self.P = np.eye(4) * 1.0                       # ковариация состояния
        self.F = np.array([[1, 0, dt, 0],              # матрица перехода F
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0],               # матрица измерения H
                           [0, 1, 0, 0]], dtype=float)
        q, r = accel_std**2, meas_std**2
        self.Q = np.eye(4) * q                         # ковариация процесса
        self.R = np.array([[r, 0], [0, r]], dtype=float)  # ковариация измерения
        self.I = np.eye(4)

    def init_state(self, px, py):
        self.x = np.array([[px], [py], [0], [0]], dtype=float)
        self.P = np.eye(4) * 1.0

    def predict(self):
        # Шаг прогнозирования
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q

    def update(self, px, py):
        # Коррекция по новому измерению (px, py)
        z = np.array([[px], [py]], dtype=float)
        y = z - self.H.dot(self.x)                                # инновация
        S = self.H.dot(self.P).dot(self.H.T) + self.R             # ковариация инновации
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))            # выигрыш Калмана
        self.x = self.x + K.dot(y)                                # обновление состояния
        self.P = (self.I - K.dot(self.H)).dot(self.P)             # обновление ковариации

class Track:
    def __init__(self, track_id, bbox):
        # Инициализация нового трека по детекции bbox = (x, y, w, h)
        self.id = track_id
        x, y, w, h = bbox
        cx = x + w/2.0
        cy = y + h/2.0
        self.kf = KalmanFilter2D()
        self.kf.init_state(cx, cy)
        self.bbox = bbox
        self.time_since_update = 0

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        # Получаем предсказанное положение центра
        px = float(self.kf.x[0]); py = float(self.kf.x[1])
        w, h = self.bbox[2], self.bbox[3]              # сохраняем прежние размеры
        x = px - w/2.0; y = py - h/2.0
        return (int(x), int(y), int(w), int(h))

    def update(self, bbox):
        # Обновление трека по новой детекции
        x, y, w, h = bbox
        cx = x + w/2.0; cy = y + h/2.0
        self.kf.update(cx, cy)
        self.bbox = bbox
        self.time_since_update = 0

class Tracker:
    def __init__(self, max_missing=5):
        self.tracks = []
        self.next_id = 1
        self.max_missing = max_missing

    def update(self, detections):
        """
        Обновить треки по новым детекциям.
        detections: список (x, y, w, h, class_id, conf).
        Возвращает список актуальных треков (словарей с id и bbox).
        """
        preds = [track.predict() for track in self.tracks]   # прогноз всех треков
        track_assigned = [False] * len(self.tracks)
        det_assigned = [False] * len(detections)
        # Сопоставление детекций с треками
        for i, track in enumerate(self.tracks):
            best_match, best_dist = None, float('inf')
            track_cx = preds[i][0] + preds[i][2]/2.0
            track_cy = preds[i][1] + preds[i][3]/2.0
            for j, det in enumerate(detections):
                if det_assigned[j]:
                    continue
                x, y, w, h, cls_id, conf = det
                det_cx = x + w/2.0; det_cy = y + h/2.0
                dist = (track_cx - det_cx)**2 + (track_cy - det_cy)**2
                if dist < best_dist:
                    best_dist = dist; best_match = j
            if best_match is not None and best_dist < (50.0**2):
                # Назначаем детекцию j треку i
                j = best_match
                self.tracks[i].update(detections[j][:4])
                track_assigned[i] = True
                det_assigned[j] = True
        # Инициализация новых треков из неназначенных детекций
        for j, det in enumerate(detections):
            if det_assigned[j]:
                continue
            bbox = det[:4]
            new_track = Track(self.next_id, bbox)
            self.tracks.append(new_track)
            self.next_id += 1
        # Удаление старых треков, потерянных более max_missing кадров
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_missing]
        # Подготовка результата: список активных треков
        output_tracks = []
        for track in self.tracks:
            x, y, w, h = track.bbox
            output_tracks.append({"id": track.id, "bbox": (int(x), int(y), int(w), int(h))})
        return output_tracks
