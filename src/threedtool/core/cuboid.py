from abc import ABC
from typing import Tuple, Union

import numpy as np
from icecream import ic
from numpy.typing import NDArray

from threedtool.core.annotations import Array3, Array3x3
from threedtool.core.transform import (
    rot_x,
    rot_y,
    rot_z,
    rot_v,
)

# from pyquaternion import Quaternion
from threedtool.core.basefigure import Figure, Point3, Vector3


def project(points: NDArray[np.float64], axis: Vector3) -> Tuple[float, float]:
    projections = points @ axis
    return projections.min(), projections.max()


class Cuboid(Figure, ABC):
    """
    Класс кубоида
    """

    def __init__(
        self,
        center: Array3 = np.zeros((3,)),
        length_width_height: Array3 = np.ones((3,)),
        rotation: Array3x3 = np.eye(3),
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
        self.center: Array3 = center
        self.length_width_height: Array3 = length_width_height
        self.rotation: Array3x3 = rotation

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
        ic(offsets)
        local_vertices = offsets * half_sizes
        world_vertices = (self.rotation @ local_vertices.T).T + self.center
        print(world_vertices)
        return world_vertices  # (8, 3)

    def get_axes(self):
        # Осевые векторы объекта
        return self.rotation

    def is_intersecting(self, other_cuboid) -> bool:
        """
        Проверка пересечения двух вращённых кубоидов с помощью SAT
        """

        vertices1 = self.get_vertices()
        vertices2 = other_cuboid.get_vertices()

        axes1 = self.get_axes()
        axes2 = other_cuboid.get_axes()

        # SAT: 15 осей (3 + 3 + 9 = 15)
        axes_to_test = (
            axes1
            + axes2
            + [
                np.cross(a, b)
                for a in axes1
                for b in axes2
                if np.linalg.norm(np.cross(a, b)) > 1e-8
            ]
        )

        for ax in axes_to_test:
            ax = ax / np.linalg.norm(ax)
            min1, max1 = project(vertices1, ax)
            min2, max2 = project(vertices2, ax)
            if max1 < min2 or max2 < min1:
                return False  # Есть разделяющая ось

        return True  # Нет разделяющих осей

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
        self.rotation = self.rotation @ rot_z(angle=angle)

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


if __name__ == "__main__":
    center = np.array([0, 0, 0])
    lwh = np.array([1, 2, 3])
    cb = Cuboid(center, lwh)
    ic(cb.get_vertices())

    cb.rotate_x(np.pi / 3)

    ic(cb.get_vertices())
