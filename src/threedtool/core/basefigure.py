from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray


class Figure(ABC):
    """
    Base class of geometry figure
    """

    @abstractmethod
    def rotate_x(self, *args, **kwargs):
        pass

    @abstractmethod
    def rotate_y(self, *args, **kwargs):
        pass

    @abstractmethod
    def rotate_z(self, *args, **kwargs):
        pass

    @abstractmethod
    def rotate_euler(self, *args, **kwargs):
        pass

    @abstractmethod
    def show(self, *args, **kwargs):
        pass


class Point3(NDArray, Figure, ABC):
    """
    Класс точки [x, y, z]
    """

    def __new__(cls, data: Union[list, tuple, NDArray], *args, **kwargs):
        arr = np.asarray(data, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"Point3D must have shape (3,), got {arr.shape}")
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


class Vector3(Point3):
    def __new__(cls, data: Union[list, tuple, NDArray], *args, **kwargs):
        arr = np.asarray(data, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"Vector3D must have shape (3,), got {arr.shape}")
        obj = NDArray.__new__(
            cls, shape=arr.shape, dtype=arr.dtype, buffer=arr
        )
        return obj



AABB = Tuple[Point3, Point3]
Plane = Tuple[Vector3, float]
