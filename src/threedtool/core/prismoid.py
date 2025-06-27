import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Dict, Union
from threedtool.annotations import Array3, Array3x3
from threedtool.core.basefigure import Figure
from threedtool.fmath.fmath import normalization, rot_v, project


class Prismoid(Figure):
    """
    Класс призмоида (усеченной пирамиды)
    """

    def __init__(
            self,
            base_center: Array3,
            base_size: Tuple[float, float],  # (length, width)
            top_size: Tuple[float, float],  # (length, width)
            height: float,
            rotation: Array3x3 = np.eye(3),
            color: str = "purple"
    ):
        """
        Конструктор призмоида

        :param base_center: Центр нижнего основания
        :param base_size: Размеры нижнего основания (длина, ширина)
        :param top_size: Размеры верхнего основания (длина, ширина)
        :param height: Высота призмоида
        :param rotation: Матрица поворота
        """
        self.base_center = np.array(base_center)
        self.base_size = np.array(base_size)
        self.top_size = np.array(top_size)
        self.height = height
        self.rotation = rotation.copy()
        self.color = color

        # Вычисляем центр верхнего основания
        self.top_center = self.base_center + self.rotation[:, 2] * height

        # Вычисляем вектор направления
        self.axis = normalization(self.rotation[:, 2])

    def get_vertices(self) -> NDArray[np.float64]:
        """
        Возвращает 8 вершин призмоида
        """
        half_base = self.base_size / 2.0
        half_top = self.top_size / 2.0

        # Локальные оси для основания
        axis_x = self.rotation[:, 0]
        axis_y = self.rotation[:, 1]

        # Вершины нижнего основания
        base_vertices = [
            self.base_center - axis_x * half_base[0] - axis_y * half_base[1],
            self.base_center - axis_x * half_base[0] + axis_y * half_base[1],
            self.base_center + axis_x * half_base[0] + axis_y * half_base[1],
            self.base_center + axis_x * half_base[0] - axis_y * half_base[1],
        ]

        # Вершины верхнего основания
        top_vertices = [
            self.top_center - axis_x * half_top[0] - axis_y * half_top[1],
            self.top_center - axis_x * half_top[0] + axis_y * half_top[1],
            self.top_center + axis_x * half_top[0] + axis_y * half_top[1],
            self.top_center + axis_x * half_top[0] - axis_y * half_top[1],
        ]

        return np.array(base_vertices + top_vertices)

    def get_bounding_box(self) -> Tuple[Array3, Array3]:
        """
        Возвращает минимальную и максимальную точки ограничивающего параллелепипеда
        """
        vertices = self.get_vertices()
        min_vals = vertices.min(axis=0)
        max_vals = vertices.max(axis=0)
        return min_vals, max_vals


    def show(self, ax):
        """
        Отображает призмоид на графике
        """
        vertices = self.get_vertices()
        base_vertices = vertices[:4]
        top_vertices = vertices[4:]

        # Рисуем нижнее основание
        for i in range(4):
            start = base_vertices[i]
            end = base_vertices[(i + 1) % 4]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=self.color)

        # Рисуем верхнее основание
        for i in range(4):
            start = top_vertices[i]
            end = top_vertices[(i + 1) % 4]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=self.color)

        # Рисуем боковые ребра
        for i in range(4):
            start = base_vertices[i]
            end = top_vertices[i]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=self.color)

    def rotate_x(self, *args, **kwargs):
        pass

    def rotate_y(self, *args, **kwargs):
        pass

    def rotate_z(self, *args, **kwargs):
        pass

    def rotate_euler(self, *args, **kwargs):
        pass

def generate_random_prismoid(
        min_size: float = 1.0,
        max_size: float = 5.0,
        min_height: float = 2.0,
        max_height: float = 10.0
) -> Prismoid:
    """
    Генерирует случайный призмоид

    :param min_size: Минимальный размер основания
    :param max_size: Максимальный размер основания
    :param min_height: Минимальная высота
    :param max_height: Максимальная высота
    :return: Случайный призмоид
    """
    # Случайный центр основания
    base_center = np.random.uniform(-10, 10, size=3)

    # Случайные размеры оснований
    base_size = np.random.uniform(min_size, max_size, size=2)
    top_size = np.random.uniform(min_size, max_size, size=2)

    # Случайная высота
    height = np.random.uniform(min_height, max_height)

    # Случайный поворот
    angle = np.random.uniform(0, 2 * np.pi)
    axis = normalization(np.random.uniform(-1, 1, size=3))
    rotation = rot_v(angle, axis)

    return Prismoid(
        base_center=base_center,
        base_size=base_size,
        top_size=top_size,
        height=height,
        rotation=rotation
    )