# /// script
# dependencies = [
#   "matplotlib",
#   "pyqt5",
# ]
# ///
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from threedtool import Sphere, Cuboid
from threedtool.core.transform import rot_x, rot_z
from threedtool import Origin
from icecream import ic


class Dspl:
    def __init__(self, input_array, qt=False):
        if qt:
            mpl.use("Qt5Agg")
        self.input_array = input_array
        self.fig = None
        self.ax = None
        self.create_subplot3D()

    def create_subplot3D(self) -> None:
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("X Label")
        self.ax.set_ylabel("Y Label")
        self.ax.set_zlabel("Z Label")

    def show(self) -> None:
        self.ax.set_box_aspect([1, 1, 1])  # убираем искажение пространства
        for obj in self.input_array:
            if hasattr(obj, "show") and callable(obj.show):
                # Если объект имеет свой метод show, используем его
                obj.show(self.ax)
            else:
                # Для объектов без метода show пытаемся получить вершины и ребра
                try:
                    vertices = obj.get_vertices()
                    edges = obj.get_edges()
                    # Отображаем вершины
                    self.ax.scatter(
                        vertices[:, 0],
                        vertices[:, 1],
                        vertices[:, 2],
                        color="b",
                    )
                    # Отображаем ребра
                    for edge in edges:
                        start, end = vertices[edge[0]], vertices[edge[1]]
                        self.ax.plot(
                            [start[0], end[0]],
                            [start[1], end[1]],
                            [start[2], end[2]],
                            color="r",
                        )
                except AttributeError:
                    print(f"Объект {type(obj)} не поддерживает отображение")
        plt.show()


if __name__ == "__main__":
    origin = Origin()
    # Создаем два кубоида
    cuboid0 = Cuboid(
        center=np.array([0, 4.5, 0]), length_width_height=np.array([1, 1, 1])
    )

    cuboid1 = Cuboid(
        center=np.array([0, 0, 0]), length_width_height=np.array([1, 1, 1])
    )

    cuboid2 = Cuboid(
        center=np.array([1.5, 0, 0]),
        length_width_height=np.array([1, 1, 1]),
        rotation=rot_x(np.pi / 4),
    )

    cuboid3 = Cuboid(
        center=np.array([4.5, 0, 0]),
        length_width_height=np.array([1, 1, 1]),
        color="green",
    )

    cuboid4 = Cuboid(
        center=np.array([0, 0, 4.5]), length_width_height=np.array([1, 1, 1])
    )

    cuboid5 = Cuboid(
        center=np.array([1.5, 0, 4.5]),
        length_width_height=np.array([1, 1, 1]),
        rotation=rot_x(np.pi / 4),
    )

    cuboid6 = Cuboid(
        center=np.array([4.5, 0, 4.5]),
        length_width_height=np.array([1, 1, 1]),
        color="green",
    )

    # cuboid3.rotate_x(np.pi / 3)
    cuboid6.rotate_z(np.pi / 4)
    cuboid6.rotate_x(np.pi / 4)
    cuboid6.color = "blue"

    ic(cuboid2.is_intersecting(cuboid3))
    sp = Sphere(center=np.array([0, 0, 0]), radius=2)
    # Создаем визуализатор и отображаем объекты
    visualizer = Dspl(
        [
            sp,
            cuboid0,
            cuboid1,
            cuboid2,
            cuboid3,
            origin,
            cuboid4,
            cuboid5,
            cuboid6,
        ]
    )
    # ic(cuboid2.get_precise_intersection_points(cuboid3))
    visualizer.show()
