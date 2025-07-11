from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy import cos, sin
from numpy.typing import NDArray

from threedtool.annotations import Array3, Array3x3
from threedtool.core.basefigure import Vector3


def get_type_signature(obj):
    return obj.__class__.__module__ + "." + obj.__class__.__name__


def normalization(vector: NDArray, length: float = 1.0) -> NDArray:
    """
    Функция возвращает нормированный вектор заданной длины

    :param vector: Вектор любой размерности
    :type vector: NDArray
    :param length: Длина вектора
    :type length: float
    """
    return np.array(vector) / np.linalg.norm(vector) * length


def rot_v(angle: float, axis: Array3) -> Array3:
    """
    Функция фомирует матрицу поворота вокруг оси axis и на угол angle

    Для правой системы координат

    :param axis: Ось вращения
    :param angle: Угол вращения, заданный в радианах
    :return: Матрица, соответствующая углу поворота по axis
    """

    axis = normalization(axis, 1)
    x, y, z = axis
    c = cos(angle)
    s = sin(angle)
    t = 1 - c
    rotate = np.array(
        [
            [t * x**2 + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y**2 + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z**2 + c],
        ]
    )
    return rotate


def rot_x(angle: float) -> Array3x3:
    """
    Функция возвращает матрицу поворота для угла angle [рад] по оси x

    Для правой системы координат

    :param angle: Угол вращения, заданный в радианах
    :return: Матрица, соответствующая углу поворота по x
    """
    rotate_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )
    return rotate_x


def rot_y(angle: float) -> Array3x3:
    """
    Функция возвращает матрицу поворота для угла angle [рад] по оси y

    Для правой системы координат

    :param angle: Угол вращения, заданный в радианах
    :return: Матрица, соответствующая углу поворота по y
    """
    rotate_y = np.array(
        [
            [cos(angle), 0, sin(angle)],
            [0, 1, 0],
            [-sin(angle), 0, cos(angle)],
        ]
    )
    return rotate_y


def rot_z(angle: float) -> Array3x3:
    """
    Функция возвращает матрицу поворота для угла angle [рад] по оси z

    Для правой системы координат

    :param angle: Угол вращения, заданный в радианах
    :return: Матрица, соответствующая углу поворота по z
    """
    rotate_z = np.array(
        [[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]]
    )
    return rotate_z


def project(points: NDArray[np.float64], axis: Vector3) -> Tuple[float, float]:
    projections = points @ axis
    return projections.min(), projections.max()