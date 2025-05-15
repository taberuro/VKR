import cv2
# Импорт всех модулей системы
from detector import Detector
from tracker import Tracker
from depth_estimator import DepthEstimator
from planner import Planner
from visualizer import Visualizer

def main(video_source=0):
    # Инициализация модулей
    detector = Detector("yolov8n.pt", conf_threshold=0.5)
    tracker = Tracker(max_missing=5)
    depth_estimator = DepthEstimator("model-small.onnx")
    # Параметры камеры (фокусные px и центр) – заданы по калибровке или приближенно
    fx = fy = 500.0  # пример значения фокусного расстояния
    cx = 320.0; cy = 240.0  # центр изображения для 640x480
    planner = Planner(fx, fy, cx, cy)
    visualizer = Visualizer()

    # Подключение к автопилоту (условный код)
    # autopilot = connect_to_px4()

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Шаг 1: Детекция объектов
        detections = detector.detect(frame)
        # Шаг 2: Обновление треков
        tracks = tracker.update(detections)
        # Шаг 3: Оценка глубины текущего кадра
        depth_map = depth_estimator.estimate_depth(frame)
        for track in tracks:
            x, y, w, h = track['bbox']
            cx = int(x + w/2); cy = int(y + h/2)
            depth_val = float(depth_map[cy, cx])
            track['depth'] = depth_val  # добавить оценку расстояния в структуру трека
        # Шаг 4: Выбор цели и расчет вектора
        target_track = planner.select_target(tracks)
        if target_track:
            vector = planner.compute_vector(drone_position=None, drone_orientation=None, track=target_track)
            # Отправка команды в PX4 (псевдокод)
            # autopilot.send_velocity_vector(vector)
        # Шаг 5: Визуализация
        output_frame = visualizer.draw_tracks(frame, tracks, depth_map)
        cv2.imshow("AutoNav System", output_frame)
        # Выход из цикла по нажатию ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
