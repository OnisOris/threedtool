from abc import ABC
from typing import Union, Tuple
import numpy as np
from numpy.typing import NDArray
from pyquaternion import Quaternion
from .basefigure import Figure, Point3, Vector3
from icecream import ic


def project(points: NDArray[np.float64],
            axis: Vector3) -> Tuple[float, float]:
    projections = points @ axis
    return projections.min(), projections.max()


class Cuboid(Figure, ABC):
    def __new__(cls, data: Union[list, tuple, np.ndarray]):
        arr = np.asarray(data, dtype=np.float64)
        if arr.shape != (5,3):
            raise ValueError(f"{__name__} must have shape (5,3), got {arr.shape}")
        obj = np.ndarray.__new__(cls, shape=arr.shape, dtype=arr.dtype, buffer=arr)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"

    @property
    def center(self):
        return self[0]
    @center.setter
    def center(self, center):
        self[0] = center
    @property
    def fsize(self):
        return self[1]
    @fsize.setter
    def fsize(self, value):
        self[1] = value
    @property
    def rotation(self):
        return self[2:5, :]
    @rotation.setter
    def rotation(self, value):
        self[2:5, :] = value

    def get_vertices(self) -> NDArray[np.float64]:
        """
        Возвращает 8 вершин кубоида в мировых координатах
        """
        half_sizes = self.fsize / 2.0
        # 8 комбинаций углов [-1,1] по каждой оси
        offsets = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        ic(offsets)
        local_vertices = offsets * half_sizes
        world_vertices = (self.rotation @ local_vertices.T).T + self.center
        print(world_vertices)
        return world_vertices  # (8, 3)

    def get_axes(self):
        # Осевые векторы объекта
        return [self.rotation[:, i] for i in range(3)]

    def is_intersecting(self, other_cuboid) -> bool:
        """
        Проверка пересечения двух вращённых кубоидов с помощью SAT
        """

        vertices1 = self.get_vertices()
        vertices2 = other_cuboid.get_vertices()

        axes1 = self.get_axes()
        axes2 = other_cuboid.get_axes()

        # SAT: 15 осей (3 + 3 + 9 = 15)
        axes_to_test = axes1 + axes2 + [np.cross(a, b) for a in axes1 for b in axes2 if np.linalg.norm(np.cross(a, b)) > 1e-8]

        for ax in axes_to_test:
            ax = ax / np.linalg.norm(ax)
            min1, max1 = project(vertices1, ax)
            min2, max2 = project(vertices2, ax)
            if max1 < min2 or max2 < min1:
                return False  # Есть разделяющая ось

        return True  # Нет разделяющих осей

    def rotate_x(self):
        pass

    def rotate_y(self):
        pass

    def rotate_z(self):
        pass

    def rotate_euler(self):
        pass

