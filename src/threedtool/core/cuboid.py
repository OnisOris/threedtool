from abc import ABC
from idlelib.configdialog import is_int
# from typing import Tuple, Union, override

import numpy as np
from numpy.typing import NDArray


# import trimesh

from threedtool.annotations import Array3, Array3x3
from threedtool.core.basefigure import Figure
from threedtool.fmath.fmath import (
    rot_v,
    rot_x,
    rot_y,
    rot_z,
    project,
)


class Cuboid(Figure, ABC):
    """
    Класс кубоида
    """

    def __init__(
        self,
        center: Array3 = np.zeros((3,)),
        length_width_height: Array3 = np.ones((3,)),
        rotation: Array3x3 = np.eye(3),
        color: str = "red",
    ):
        """
        Конструктор кубоида

        :param center: центральная точка кубоида
        :param length_width_height: длина, ширина, высота в ndarray
        :param rotation: оси кубоида, повернутые в пространстве
        :note: rotation размера 3x3, [[i],
                                      [j],
                                      [k]],
                                      i, j, k - орты
        """
        self.center: Array3 = center
        self.length_width_height: Array3 = length_width_height
        self.rotation: Array3x3 = rotation
        self.color: str = color

    @property
    def length(self):
        return self.length_width_height[0]

    @length.setter
    def length(self, value):
        self.length_width_height[0] = value

    @property
    def width(self):
        return self.length_width_height[1]

    @width.setter
    def width(self, value):
        self.length_width_height[1] = value

    @property
    def height(self):
        return self.length_width_height[2]

    @height.setter
    def height(self, value):
        self.length_width_height[2] = value

    def get_vertices(self) -> NDArray[np.float64]:
        """
        Возвращает 8 вершин кубоида в мировых координатах
        """
        half_sizes = self.length_width_height / 2.0
        # 8 комбинаций углов [-1,1] по каждой оси
        offsets = np.array(
            [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        )
        local_vertices = offsets * half_sizes
        world_vertices = (self.rotation @ local_vertices.T).T + self.center
        return world_vertices

    def get_axes(self):
        # Осевые векторы объекта
        return self.rotation.T

    def intersects_with(self, other):
        from threedtool.core.sphere import Sphere, Line3  # локальный импорт, чтобы избежать циклов

        if isinstance(other, Cuboid):
            return self.is_intersecting_cuboid(other)
        elif isinstance(other, Sphere):
            return self.is_intersecting_sphere(other)
        elif isinstance(other, Line3):
            return self.is_intersecting_line(other)
        return False

    def is_intersecting_line(self, line: "Line3") -> bool:
        """
        Проверяет пересечение бесконечной прямой и ориентированного кубоида
        через метод «slab intersection» в локальных координатах.
        """
        # Переводим прямую в локальную систему кубоида
        R = self.rotation  # 3×3
        C = self.center  # 3,
        A = line.abc
        v = line.p  # 3, нормирован

        A_loc = R.T.dot(A - C)
        v_loc = R.T.dot(v)

        half = self.length_width_height / 2.0

        t_min, t_max = -np.inf, np.inf

        # для каждой локальной оси i
        for i in range(3):
            if abs(v_loc[i]) < 1e-8:
                # прямая параллельна граням: если не в промежутке, то нет пересечения
                if A_loc[i] < -half[i] or A_loc[i] > half[i]:
                    return False
            else:
                # находим пересечения с «плоскостями» x_i=±half[i]
                t1 = (-half[i] - A_loc[i]) / v_loc[i]
                t2 = (half[i] - A_loc[i]) / v_loc[i]
                t_near, t_far = min(t1, t2), max(t1, t2)

                t_min = max(t_min, t_near)
                t_max = min(t_max, t_far)
                if t_min > t_max:
                    return False

        return True  # существует хотя бы один t, где прямая внутри кубоида

    def is_intersecting_cuboid(self, other_cuboid):
        class_name = other_cuboid.__class__.__name__
        module_name = other_cuboid.__class__.__module__
        if class_name != "Cuboid" and module_name != "threedtool.core.cuboid.Cuboid":
            raise TypeError
        vertices1 = self.get_vertices()
        vertices2 = other_cuboid.get_vertices()

        axes1 = self.get_axes()
        axes2 = other_cuboid.get_axes()


        cross_products = np.array(
            [
                np.cross(a, b)
                for a in axes1
                for b in axes2
                if np.linalg.norm(np.cross(a, b)) > 1e-8
            ]
        )

        axes_to_test = [axes1, axes2]
        if cross_products.size > 0:
            axes_to_test.append(cross_products)
        axes_to_test = np.concatenate(axes_to_test, axis=0)

        for ax in axes_to_test:
            ax = ax / np.linalg.norm(ax)
            min1, max1 = project(vertices1, ax)
            min2, max2 = project(vertices2, ax)
            if max1 < min2 or max2 < min1:
                return False  # Есть разделяющая ось
        return True


    def is_intersecting_sphere(self, sphere) -> bool:
        """
        Проверяет пересечение кубоида и сферы

        :param sphere: Объект сферы
        :return: True если есть пересечение, иначе False
        """
        # Преобразование центра сферы в локальную систему кубоида
        center_local = self.rotation.T @ (sphere.center - self.center)

        # Размеры кубоида в локальной системе
        half_sizes = self.length_width_height / 2.0

        # Находим ближайшую точку в локальных координатах
        closest_local = np.clip(center_local, -half_sizes, half_sizes)

        # Вычисляем расстояние между точками
        distance_sq = np.sum((center_local - closest_local) ** 2)

        # Проверяем пересечение
        return distance_sq <= (sphere.radius**2)

    def rotate_x(self, angle: float) -> None:
        """
        Функция вращения по оси x
        """
        self.rotation = self.rotation @ rot_x(angle=angle)

    def rotate_y(self, angle: float) -> None:
        """
        Функция вращения по оси y
        """
        self.rotation = self.rotation @ rot_y(angle=angle)

    def rotate_z(self, angle: float) -> None:
        """
        Функция вращения по оси z
        """
        self.rotation = self.rotation @ rot_z(angle=angle)

    def rotate_v(self, axis_vector: Array3, angle: float) -> None:
        """
        Функция вращения по заданной оси вектором v
        """
        self.rotation = self.rotation @ rot_v(angle=angle, axis=axis_vector)

    def rotate_euler(self, alpha: float, betta: float, gamma: float) -> None:
        """
        Вращение кубоида по углам Эйлера

        :param alpha: Угол прецессии
        :param betta: Угол нутации
        :param gamma: Угол собственного вращения
        """
        self.rotation = (
            rot_z(alpha) @ rot_x(betta) @ rot_z(gamma) @ self.rotation
        )

    def get_edges(self):
        """Возвращает список ребер кубоида в виде пар индексов вершин."""
        return [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        ]

    def show(self, ax):
        """Отображает кубоид на графике."""
        vertices = self.get_vertices()
        edges = self.get_edges()

        # Отображаем вершины
        ax.scatter(
            vertices[:, 0], vertices[:, 1], vertices[:, 2], color="blue"
        )
        ax.quiver(*self.center, *self.rotation[0], color="red")
        ax.quiver(*self.center, *self.rotation[1], color="green")
        ax.quiver(*self.center, *self.rotation[2], color="blue")
        # Отображаем ребра
        for edge in edges:
            start, end = vertices[edge[0]], vertices[edge[1]]
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=self.color,
            )

    # def to_trimesh(self) -> trimesh.Trimesh:
    #     """
    #     Преобразует экземпляр Cuboid в корректный объект trimesh.Trimesh
    #     """
    #     # создаём unit box с центром в начале координат
    #     unit_box = trimesh.creation.box(extents=self.length_width_height)
    #
    #     # применяем поворот
    #     rotation_matrix = np.eye(4)
    #     rotation_matrix[:3, :3] = self.rotation
    #
    #     # применяем перенос
    #     translation_matrix = np.eye(4)
    #     translation_matrix[:3, 3] = self.center
    #
    #     # итоговая трансформация
    #     transform = translation_matrix @ rotation_matrix
    #     unit_box.apply_transform(transform)
    #
    #     return unit_box
    #
    # def get_precise_intersection_points(self, cuboid) -> NDArray[np.float64]:
    #     mesh1 = self.to_trimesh()
    #     mesh2 = cuboid.to_trimesh()
    #
    #
    #     if not mesh1.is_volume or not mesh2.is_volume:
    #         raise ValueError("Один из мешей не является объемом!")
    #
    #     intersection = mesh1.intersection(mesh2, engine='igl')
    #
    #     if intersection.is_empty:
    #         return np.empty((0, 3))
    #
    #     return intersection.vertices


if __name__ == "__main__":
    center = np.array([0, 0, 0])
    lwh = np.array([1, 2, 3])
    cb = Cuboid(center, lwh)

    cb.rotate_x(np.pi / 3)
