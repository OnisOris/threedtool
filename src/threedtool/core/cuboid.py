from abc import ABC
from typing import List

import numpy as np
from numpy.typing import NDArray

from threedtool.annotations import Array3, Array3x3
from threedtool.core.basefigure import Figure
from threedtool.fmath.fmath import (
    rot_v,
    rot_x,
    rot_y,
    rot_z,
)


class Cuboid(Figure, ABC):
    """
    Класс кубоида
    """

    def __init__(
        self,
        center: Array3 = np.zeros((3,)),
        length_width_height: Array3 = np.ones((3,)),
        rotation: Array3x3 = np.eye(3),
        color: str = "red",
    ):
        """
        Конструктор кубоида

        :param center: центральная точка кубоида
        :param length_width_height: длина, ширина, высота в ndarray
        :param rotation: оси кубоида, повернутые в пространстве
        :note: rotation размера 3x3, [[i],
                                      [j],
                                      [k]],
                                      i, j, k - орты
        """
        self.center: Array3 = center.copy()
        self.length_width_height: Array3 = length_width_height.copy()
        self.rotation: Array3x3 = rotation.copy()
        self.color: str = color

    @property
    def vertices_base(self):
        return self.get_vertices()[:4]

    @property
    def height_vector(self):
        return self.get_axes()[2] * self.length_width_height[2]

    @property
    def length(self):
        return self.length_width_height[0]

    @length.setter
    def length(self, value):
        self.length_width_height[0] = value

    @property
    def width(self):
        return self.length_width_height[1]

    @width.setter
    def width(self, value):
        self.length_width_height[1] = value

    @property
    def height(self):
        return self.length_width_height[2]

    @height.setter
    def height(self, value):
        self.length_width_height[2] = value

    def get_vertices(self) -> NDArray[np.float64]:
        """
        Возвращает 8 вершин кубоида в мировых координатах
        """
        half_sizes = self.length_width_height / 2.0
        # 8 комбинаций углов [-1,1] по каждой оси
        offsets = np.array(
            [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        )
        local_vertices = offsets * half_sizes
        world_vertices = (self.rotation @ local_vertices.T).T + self.center
        return world_vertices

    def get_axes(self):
        # Осевые векторы объекта
        return self.rotation.T

    def rotate_x(self, angle: float) -> None:
        """
        Функция вращения по оси x
        """
        self.rotation = self.rotation @ rot_x(angle=angle)

    def rotate_y(self, angle: float) -> None:
        """
        Функция вращения по оси y
        """
        self.rotation = self.rotation @ rot_y(angle=angle)

    def rotate_z(self, angle: float) -> None:
        """
        Функция вращения по оси z
        """
        self.rotation = self.rotation @ rot_z(angle=angle)

    def rotate_v(self, axis_vector: Array3, angle: float) -> None:
        """
        Функция вращения по заданной оси вектором v
        """
        self.rotation = self.rotation @ rot_v(angle=angle, axis=axis_vector)

    def rotate_euler(self, alpha: float, betta: float, gamma: float) -> None:
        """
        Вращение кубоида по углам Эйлера

        :param alpha: Угол прецессии
        :param betta: Угол нутации
        :param gamma: Угол собственного вращения
        """
        self.rotation = (
            rot_z(alpha) @ rot_x(betta) @ rot_z(gamma) @ self.rotation
        )

    def get_edges(self):
        """Возвращает список ребер кубоида в виде пар индексов вершин."""
        return [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        ]

    # def get_face_normals(self) -> List[Array3]:
    #     """Нормали граней: основание + боковые"""
    #     # нормаль основания
    #     e1 = self.vertices_base[1] - self.vertices_base[0]
    #     e2 = self.vertices_base[2] - self.vertices_base[0]
    #     base_normal = np.cross(e1, e2)
    #     base_normal /= np.linalg.norm(base_normal)
    #
    #     normals = [base_normal, -base_normal]
    #     # боковые грани
    #     N = len(self.vertices_base)
    #     for i in range(N):
    #         v0 = self.vertices_base[i]
    #         v1 = self.vertices_base[(i + 1) % N]
    #         edge = v1 - v0
    #         side_normal = np.cross(edge, self.height_vector)
    #         norm = np.linalg.norm(side_normal)
    #         if norm > 1e-8:
    #             normals.append(side_normal / norm)
    #     return normals

    def get_face_normals(self) -> List[Array3]:
        """Возвращает нормали для всех 6 граней кубоида"""
        axes = self.get_axes()
        normals = []
        for i in range(3):
            # Положительное и отрицательное направление для каждой оси
            normals.append(axes[i])
            normals.append(-axes[i])
        return normals

    def show(self, ax):
        """Отображает кубоид на графике."""
        vertices = self.get_vertices()
        edges = self.get_edges()

        # Отображаем вершины
        ax.scatter(
            vertices[:, 0], vertices[:, 1], vertices[:, 2], color="blue"
        )
        ax.quiver(*self.center, *self.rotation[0], color="red")
        ax.quiver(*self.center, *self.rotation[1], color="green")
        ax.quiver(*self.center, *self.rotation[2], color="blue")
        # Отображаем ребра
        for edge in edges:
            start, end = vertices[edge[0]], vertices[edge[1]]
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=self.color,
            )
