import cv2

class Visualizer:
    def __init__(self):
        self.colors = {}  # словарь цветов для каждого track_id
    
    def get_color(self, track_id):
        # Генерировать постоянный уникальный цвет для данного track_id
        if track_id not in self.colors:
            import random
            random.seed(track_id)
            self.colors[track_id] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        return self.colors[track_id]

    def draw_tracks(self, frame, tracks, depth_map=None):
        """
        Нарисовать на изображении рамки треков, их ID, 3D-боксы и расстояние (если доступно).
        """
        for track in tracks:
            tid = track['id']
            x, y, w, h = track['bbox']
            color = self.get_color(tid)
            # 2D-рамка и ID
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"ID {tid}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Построение "3D-бокса"
            dx = int(w * 0.3)   # горизонтальное смещение задней грани
            dy = int(h * 0.1)   # вертикальное смещение задней грани
            # Координаты углов передней грани (совпадает с 2D-боксом)
            front_tl = (x, y)
            front_tr = (x+w, y)
            front_bl = (x, y+h)
            front_br = (x+w, y+h)
            # Задняя грань (смещенная)
            back_tl = (x + dx, y - dy)
            back_tr = (x + w + dx, y - dy)
            back_bl = (x + dx, y + h - dy)
            back_br = (x + w + dx, y + h - dy)
            # Рисуем переднюю и заднюю прямоугольники
            cv2.rectangle(frame, front_tl, front_br, color, 1)
            cv2.rectangle(frame, back_tl, back_br, color, 1)
            # Соединяем соответствующие углы линиями
            cv2.line(frame, front_tl, back_tl, color, 1)
            cv2.line(frame, front_tr, back_tr, color, 1)
            cv2.line(frame, front_bl, back_bl, color, 1)
            cv2.line(frame, front_br, back_br, color, 1)
            # Если известна глубина – вывести численное значение
            if depth_map is not None:
                cx = x + w//2; cy = y + h//2
                depth_val = depth_map[cy, cx]
                cv2.putText(frame, f"{depth_val:.2f}", (x, y+h+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
