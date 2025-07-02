from abc import ABC

import numpy as np
from numpy.typing import NDArray
from typing import Union
from threedtool.core.basefigure import Figure
from threedtool.fmath.fmath import (
    normalization,
    rot_v,
    rot_x,
    rot_y,
    rot_z,
    rot_v,
)

class Plane(np.ndarray, Figure, ABC):
    """
    Класс, реализующий уравнение плоскости типа:
    ax + by + cz + d = 0

    Наследуется от np.ndarray, всегда имеет размерность (4,), входные данные нормализуются
    """
    def __new__(cls, data: Union[list, tuple, NDArray]=[0, 0, 1, 0], *args, **kwargs):
        arr = np.asarray(data, dtype=np.float64)
        if arr.shape != (4,):
            raise ValueError(f"Plane must have shape (4,), got {arr.shape}")
        arr = arr / np.linalg.norm(arr[0:3])
        obj = NDArray.__new__(
            cls, shape=arr.shape, dtype=arr.dtype, buffer=arr
        )
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    @property
    def a(self):
        return self[0]

    @property
    def b(self):
        return self[1]

    @property
    def c(self):
        return self[2]

    @property
    def d(self):
        return self[3]

    @property
    def normal(self):
        """
        Геттер нормали плоскости
        """
        return self[0:3]

    @a.setter
    def a(self, a):
        self[0] = a

    @b.setter
    def b(self, b):
        self[1] = b

    @c.setter
    def c(self, c):
        self[2] = c

    @d.setter
    def d(self, d):
        self[3] = d

    def rotate_x(self):
        pass

    def rotate_y(self):
        pass

    def rotate_z(self):
        pass

    def rotate_euler(self):
        pass

    def show(self, ax, size: float = 10.0, alpha: float = 0.5) -> None:
        """
        Отображает плоскость в виде квадрата с заданным размером.

        Параметры:
            ax: Axes3D объект matplotlib для отрисовки.
            size: Размер квадрата (половина длины стороны). По умолчанию 10.0.
            alpha: Прозрачность плоскости (0-1). По умолчанию 0.5.
        """
        # Нормаль плоскости (a, b, c)
        normal = np.array([self.a, self.b, self.c])

        # Центр плоскости: проекция (0,0,0) на плоскость
        center = -self.d * normal

        # Поиск ортогональных векторов в плоскости
        # Первый вектор (выбираем любой непараллельный нормали)
        if not np.isclose(normal[0], 0) or not np.isclose(normal[1], 0):
            vec1 = np.array([normal[1], -normal[0], 0])
        else:
            vec1 = np.array([0, normal[2], -normal[1]])

        vec1 = vec1 / np.linalg.norm(vec1)

        # Второй вектор через векторное произведение
        vec2 = np.cross(normal, vec1)
        vec2 = vec2 / np.linalg.norm(vec2)

        # Генерация вершин квадрата
        points = []
        for i in (-1, 1):
            for j in (-1, 1):
                point = center + size * (i * vec1 + j * vec2)
                points.append(point)

        # Преобразуем в массив numpy
        points = np.array(points)

        # Создаем полигон через треугольники
        triangles = [[0, 1, 3], [0, 3, 2]]  # Два треугольника для квадрата

        # Отрисовываем треугольники
        for tri in triangles:
            x = points[tri, 0]
            y = points[tri, 1]
            z = points[tri, 2]
            ax.plot_trisurf(x, y, z, alpha=alpha, color="#FF00FF", linewidth=0)

        # Дополнительно рисуем нормаль
        ax.quiver(*center, *normal, length=size / 2, color="red", normalize=True)