# from __future__ import annotations
# from math import sqrt
from abc import ABC, abstractmethod
from email.base64mime import header_length
from typing import Tuple, Union

import numpy as np
from docutils.nodes import header
from loguru import logger
from numpy.typing import NDArray

from threedtool.fmath.fmath import (
    normalization,
    rot_v,
    rot_x,
    rot_y,
    rot_z,
    rot_v,
)
from threedtool.core.basefigure import Figure, Point3
from threedtool.annotations import Array3


class LineSegment3(NDArray):
    """
    Класс отрезка, состоящий из двух точек
    """

    def __new__(cls, data: Union[list, tuple, NDArray]):
        arr = np.asarray(data, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(
                f"LineSegment must have shape (2,3), got {arr.shape}"
            )
        obj = NDArray.__new__(
            cls, shape=arr.shape, dtype=arr.dtype, buffer=arr
        )
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"

    def rotate_x(self):
        pass

    def rotate_y(self):
        pass

    def rotate_z(self):
        pass

    def rotate_euler(self):
        pass

    # def to_line3(self) -> Line3:
    #     return


class Line3(np.ndarray, Figure):
    """
    Класс строится на каноническом уравнении линии:

    (x-a)/p1 = (y-b)/p2 = (z-c)/p3
    Матрица Line3 выглядит следующим образом:
    [[a, b, c],
     [p1, p2, p3]]
    """

    def __new__(cls, data: Union[list, tuple, NDArray], *args, **kwargs):
        arr = np.asarray(data, dtype=np.float64)
        arr[1] = normalization(arr[1])
        if arr.shape != (2, 3):
            raise ValueError(f"Line3 must have shape (2,3), got {arr.shape}")
        obj = NDArray.__new__(
            cls, shape=arr.shape, dtype=arr.dtype, buffer=arr
        )
        return obj

    def __init__(
        self,
        data: Union[list, tuple, NDArray],
        length: float = 1.0,
        color: str = "red",
        *args,
        **kwargs,
    ):
        # arr = np.asarray(data, dtype=np.float64)
        # Figure.__init__(*args, **kwargs)
        self.length: float = length
        self.color = color

    @property
    def a(self):
        return self[0, 0]

    @property
    def b(self):
        return self[0, 1]

    @property
    def c(self):
        return self[0, 2]

    @property
    def p1(self):
        return self[1, 0]

    @property
    def p2(self):
        return self[1, 1]

    @property
    def p3(self):
        return self[1, 2]

    @property
    def abc(self):
        return self[0]

    @property
    def p(self):
        return self[1]

    @a.setter
    def a(self, a):
        self[0, 0] = a

    @b.setter
    def b(self, b):
        self[0, 1] = b

    @c.setter
    def c(self, c):
        self[0, 2] = c

    @p1.setter
    def p1(self, p1):
        self[1, 0] = p1

    @p2.setter
    def p2(self, p2):
        self[1, 1] = p2

    @p3.setter
    def p3(self, p3):
        self[1, 2] = p3

    def rotate_x(self, angle) -> None:
        self[1] = rot_x(angle) @ self[1]

    def rotate_y(self, angle) -> None:
        self[1] = rot_y(angle) @ self[1]

    def rotate_z(self, angle) -> None:
        self[1] = rot_z(angle) @ self[1]

    def rotate_euler(self, alpha: float, betta: float, gamma: float) -> None:
        self[1] = rot_z(alpha) @ rot_x(betta) @ rot_z(gamma) @ self[1]

    def offset_point(self, distance: float) -> Point3:
        """
        Точка отступа от центра линии

        Данная функция возвращает точку, которая отступается от центра линии [a, b, c] на расстояние distance
        в сторону вектора линии

        :param distance: дистанция отступа
        :type distance: float | int
        :return: Point3
        """
        vector_plus = normalization(self[1], distance)
        return_point = self.abc + vector_plus
        return Point3(return_point)

    def point_belongs_to_the_line(self, point: list | NDArray) -> bool:
        """
        Функция, определяющая, принадлежит ли точка прямой.

        Возвращает True, если принадлежит, False, если не принадлежит.
        :param point: список из координат [x, y, z]
        :type point: list or NDArray
        :return: bool
        """
        eq1 = np.round(
            self.p2 * self.p3 * (point[0] - self.a)
            - self.p1 * self.p3 * (point[1] - self.b),
            8,
        )
        eq2 = np.round(
            self.p1 * self.p3 * (point[1] - self.b)
            - self.p1 * self.p2 * (point[2] - self.c),
            8,
        )
        eq3 = np.round(
            self.p1 * self.p2 * (point[2] - self.c)
            - self.p2 * self.p3 * (point[0] - self.a),
            8,
        )
        if eq1 == 0 and eq2 == 0 and eq3 == 0:
            return True
        else:
            return False

    def equation_y(self):
        """
        Данная функция возвращает коэффициенты k_1, b, k_2, c из уравнения y = k_1*x + b + k_2*z + c

        :return: dict
        """
        return {
            "k_1": self.p2 / self.p1,
            "k_2": self.p2 / self.p3,
            "b": self.b - self.a * self.p2 / self.p1,
            "c": self.c * self.p2 / self.p3,
        }

    def show(self, ax) -> None:
        """
        Отображает линию
        """
        ax.scatter(self[0, 0], self[0, 1], self[0, 2], color="#FF00FF")
        vector = self[1]
        ax.quiver(*self[0], *vector, color="#FF00FF", length=self.length / 2)
        offset_point = self.offset_point(-self.length / 2)
        points = np.vstack([self[0], offset_point]).T
        ax.plot(*points, color="#FF00FF")

    # def intersects_with(self, other):
    #     from threedtool import Sphere, Cuboid
    #
    #     if isinstance(other, Cuboid):
    #         return self.is_intersecting_cuboid(other)
    #     elif isinstance(other, Sphere):
    #         return self.is_intersecting_sphere(other)
    #     elif isinstance(other, Line3):
    #         return self.is_intersecting_line(other)
    #     return False

    # def is_intersecting_sphere(self, sphere):
    #     return is_intersecting_line_sphere(sphere, self)

    def get_ABCD_of_plane(self) -> np.ndarray:
        """
        Возвращает ABCD коэффициенты плоскости.

        Через любую линию можно провести плоскость, через которую можно провести перпендикуляр к началу координат, либо
        нулевой вектор, в случае, если поскость проходит через начало координат
        """
        A = self.p2 * self.p3
        B = -2 * self.p1 * self.p3
        C = self.p1 * self.p2
        D = (
            -self.a * self.p2 * self.p3
            - 2 * self.b * self.p1 * self.p3
            - self.c * self.p1 * self.p2
        )
        return np.array([A, B, C, D])
