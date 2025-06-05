import numpy as np

from threedtool.core.annotations import Array3, Array3x3
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
        self.center: Array3 = center
        self.radius: float = radius
        self.rotation: Array3x3 = rotation

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

    def rotate_x(self):
        pass

    def rotate_y(self):
        pass

    def rotate_z(self):
        pass

    def rotate_euler(self):
        pass
