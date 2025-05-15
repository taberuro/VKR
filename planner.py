import math

class Planner:
    def __init__(self, fx, fy, cx, cy):
        """
        fx, fy – фокусные расстояния камеры (px); cx, cy – координаты центра изображения.
        """
        self.fx = fx; self.fy = fy
        self.cx = cx; self.cy = cy

    def compute_vector(self, drone_position, drone_orientation, track):
        """
        Вычислить вектор сближения в системе дрона.
        drone_position: текущие координаты БПЛА (не используется в базовом методе).
        drone_orientation: ориентация БПЛА (например, угол yaw) – может использоваться для преобразования в глобальную систему.
        track: словарь с данными цели (bbox и depth).
        Возвращает кортеж (Vx, Vy, Vz) – направляющие косинусы или относительные скорости.
        """
        x, y, w, h = track['bbox']
        depth = track.get('depth', 1.0)  # глубина (если не задана – берем 1.0 как относительную единицу)
        # Центр объекта в координатах изображения
        cx_det = x + w/2.0
        cy_det = y + h/2.0
        # Рассчитываем координаты вектора в камере
        X_cam = (cx_det - self.cx) / self.fx * depth
        Y_cam = (cy_det - self.cy) / self.fy * depth
        Z_cam = depth
        vec_cam = (X_cam, Y_cam, Z_cam)
        # Примечание: здесь мы работаем в системе координат дрона (камеры), 
        # поэтому для наведения можно напрямую использовать vec_cam как направление.
        # При необходимости можно добавить преобразование в глобальную систему с учетом drone_orientation.
        return vec_cam

    def select_target(self, tracks):
        """
        Выбрать приоритетную цель из списка треков.
        Сейчас: просто первый трек (можно улучшить, учитывая классы или расстояние).
        """
        if not tracks:
            return None
        return tracks[0]
