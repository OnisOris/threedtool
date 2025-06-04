# /// script
# dependencies = [
#   "matplotlib",
#   "pyqt5",
# ]
# ///
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from threedtool.core.cuboid import Cuboid
from threedtool.core.transform import rot_x, rot_z
from threedtool import Origin
from icecream import ic

class Dspl:
    def __init__(self, input_array, qt=False):
        if qt:
            mpl.use('Qt5Agg')
        self.input_array = input_array
        self.fig = None
        self.ax = None
        self.create_subplot3D()

    def create_subplot3D(self) -> None:
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')

    def show(self) -> None:
        for obj in self.input_array:
            if hasattr(obj, 'show') and callable(obj.show):
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
                        color='b'
                    )
                    # Отображаем ребра
                    for edge in edges:
                        start, end = vertices[edge[0]], vertices[edge[1]]
                        self.ax.plot(
                            [start[0], end[0]],
                            [start[1], end[1]],
                            [start[2], end[2]],
                            color='r'
                        )
                except AttributeError:
                    print(f"Объект {type(obj)} не поддерживает отображение")
        self.ax.set_box_aspect([1, 1, 1]) # убираем искажение пространства
        plt.show()

if __name__ == "__main__":
    origin = Origin()
    # Создаем два кубоида
    cuboid1 = Cuboid(
        center=np.array([0, 0, 0]),
        length_width_height=np.array([1, 1, 1])
    )
    
    cuboid2 = Cuboid(
        center=np.array([1.5, 0, 0]),
        length_width_height=np.array([1, 2, 1]),
        rotation=rot_x(np.pi/4)
    )

    cuboid3 = Cuboid(
        center=np.array([3.5, 0, 0]),
        length_width_height=np.array([2, 3, 1]),
        rotation=rot_x(-np.pi/3) @ rot_z(np.pi/3)
    )
    ic(cuboid2.is_intersecting(cuboid3))
    # Создаем визуализатор и отображаем объекты
    visualizer = Dspl([cuboid1, cuboid2, cuboid3, origin])
    # ic(cuboid2.get_precise_intersection_points(cuboid3))
    visualizer.show()
