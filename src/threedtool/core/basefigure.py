from abc import ABC, abstractmethod
from numpy.typing import NDArray
from threedtool.core.annotations import Array3
import numpy as np
from typing import Union, Tuple


class Figure(ABC):
    """
    Base class of geometry figure
    """

    @abstractmethod
    def rotate_x(self):
        pass

    @abstractmethod
    def rotate_y(self):
        pass

    @abstractmethod
    def rotate_z(self):
        pass

    @abstractmethod
    def rotate_euler(self):
        pass


class Point3(NDArray, Figure, ABC):
    """
    Класс точки [x, y, z]
    """

    def __new__(cls, data: Array3):
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


class Vector3(Point3):
    def __new__(cls, data: Array3):
        arr = np.asarray(data, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"Vector3D must have shape (3,), got {arr.shape}")
        obj = NDArray.__new__(
            cls, shape=arr.shape, dtype=arr.dtype, buffer=arr
        )
        return obj


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


class Line3(np.ndarray, Figure):
    """
    Класс строится на каноническом уравнении линии:

    (x-a)/p1 = (y-b)/p2 = (z-c)/p3
    Матрица Line3 выглядит следующим образом:
    [[a, b, c],
     [p1, p2, p3]]
    """

    def __new__(cls, data: Union[list, tuple, NDArray]):
        arr = np.asarray(data, dtype=np.float64)
        if arr.shape != (2, 3):
            raise ValueError(f"Line3 must have shape (2,3), got {arr.shape}")
        obj = NDArray.__new__(
            cls, shape=arr.shape, dtype=arr.dtype, buffer=arr
        )
        return obj

    def __init__(self, *args, **kwargs):
        pass

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


AABB = Tuple[Point3, Point3]  # (min_corner, max_corner)
Plane = Tuple[Vector3, float]
