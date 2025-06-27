import numpy as np

from threedtool.annotations import Array3, Array3x3
from threedtool.core.basefigure import Figure


class Sphere(Figure):
    """
    Класс сферы, задается center (3,) и radius - float
    """

    def __init__(
        self, center: Array3, radius: float, rotation: Array3x3 = np.eye(3)
    ):
        """
        Конструктор сферы

        :param center: Координата сферы
        :param radius: Радиус сферы
        """
        self.center: Array3 = center.copy()
        self.radius: float = radius
        self.rotation: Array3x3 = rotation.copy()

    def show(self, ax):
        """
        Отображает сферу на переданном 3D-объекте matplotlib.

        :param ax: объект Axes3D
        """
        # Создаем сетку сферических координат
        u = np.linspace(0, 2 * np.pi, 15)
        v = np.linspace(0, np.pi, 15)
        x = self.radius * np.outer(np.cos(u), np.sin(v)) + self.center[0]
        y = self.radius * np.outer(np.sin(u), np.sin(v)) + self.center[1]
        z = self.radius * np.outer(np.ones_like(u), np.cos(v)) + self.center[2]
        ax.quiver(*self.center, *self.rotation[0], color="red")
        ax.quiver(*self.center, *self.rotation[1], color="green")
        ax.quiver(*self.center, *self.rotation[2], color="blue")
        ax.plot_surface(x, y, z, color="cyan", alpha=0.1, edgecolor="gray")

        # Отметим центр
        ax.scatter(*self.center, color="blue")

    # def intersects_with(self, other):
    #     if isinstance(other, Sphere):
    #         return self.is_intersecting_sphere(other)
    #     from threedtool.core.cuboid import Cuboid
    #
    #     if isinstance(other, Cuboid):
    #         return other.is_intersecting_sphere(self)
    #     from threedtool.core.line import Line3
    #
    #     if isinstance(other, Line3):
    #         return self.is_intersecting_line(other)
    #     return False

    # def is_intersecting_line(self, line: "Line3") -> bool:
    #     """
    #     Проверяет пересечение бесконечной прямой и сферы.
    #
    #     Расстояние от центра сферы до прямой:
    #         dist = || (C - A) × v ||
    #     где A — точка на прямой, v — единичный направляющий вектор прямой,
    #           C — центр сферы.
    #
    #     :param line: объект Line3
    #     :return: True, если пересекаются, иначе False
    #     """
    #     return is_intersecting_line_sphere(self, line)

    def rotate_x(self):
        pass

    def rotate_y(self):
        pass

    def rotate_z(self):
        pass

    def rotate_euler(self):
        pass
