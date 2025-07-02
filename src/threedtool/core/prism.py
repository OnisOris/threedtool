from typing import List

import numpy as np
from numpy.typing import NDArray

from threedtool.annotations import Array3
from threedtool.core.basefigure import Figure


class Prism(Figure):
    """
    Правильная призма с произвольным *выпуклым* многоугольным основанием и вектором высоты.

    vertices_base: ndarray[N x 3]  -- вершины основания в порядке обхода;
    height_vec: Array3            -- вектор высоты (ориентирован вверх от основания);
    color: str
    """

    def __init__(
        self,
        vertices_base: NDArray[np.float64],
        height_vec: Array3,
        color: str = "magenta",
    ):
        # базовые данные
        self.vertices_base: NDArray[np.float64] = vertices_base.copy()
        self.height_vec: Array3 = height_vec.copy()
        self.color: str = color
        # предкомпилированные вершины (основание + верх)
        top = vertices_base + height_vec
        self._vertices = np.vstack([self.vertices_base, top])

    def get_vertices(self) -> NDArray[np.float64]:
        """Все вершины призмы: сначала основание (N), затем верх (N)"""
        return self._vertices

    def get_face_normals(self) -> List[Array3]:
        """Нормали граней: основание + боковые"""
        verts = self.vertices_base
        # нормаль основания
        e1 = verts[1] - verts[0]
        e2 = verts[2] - verts[0]
        base_normal = np.cross(e1, e2)
        base_normal /= np.linalg.norm(base_normal)

        normals = [base_normal, -base_normal]
        # боковые грани
        N = len(verts)
        for i in range(N):
            v0 = verts[i]
            v1 = verts[(i + 1) % N]
            edge = v1 - v0
            side_normal = np.cross(edge, self.height_vec)
            norm = np.linalg.norm(side_normal)
            if norm > 1e-8:
                normals.append(side_normal / norm)
        return normals

    # def intersects_with(self, other: object) -> bool:
    #     # Призма ↔ Призма
    #     if isinstance(other, Prism):
    #         return self.is_intersecting_prism(other)
    #     # Призма ↔ Кубоид (кубоид как призма)
    #     if isinstance(other, Cuboid):
    #         base = other.get_vertices()[:4]  # кубоид-основание
    #         height = other.height
    #         height_vec = other.get_axes()[2] * height
    #         prism_from_cuboid = Prism(base, height_vec)
    #         return self.is_intersecting_prism(prism_from_cuboid)
    #     # Призма ↔ Сфера (упрощённый тест)
    #     if isinstance(other, Sphere):
    #         V = self.get_vertices()
    #         dists = np.linalg.norm(V - other.center, axis=1)
    #         return np.any(dists <= other.radius)
    #     # Призма ↔ Прямая (упрощённый тест: проверяем пересечение рёбер)
    #     if isinstance(other, Line3):
    #         V = self.get_vertices()
    #         edges = [(V[i], V[(i + 1) % len(V)]) for i in range(len(V))]
    #         for start, end in edges:
    #             # параметризуем отрезок и проверяем на пересечение с прямой
    #             # TODO: точный алгоритм
    #             pass
    #         return False
    #     return False

    def rotate_x(self):
        pass

    def rotate_y(self):
        pass

    def rotate_z(self):
        pass

    def rotate_euler(self):
        pass

    def show(self, ax):
        """Отображает призму на matplotlib Axes3D"""
        verts = self.get_vertices()
        N = len(self.vertices_base)

        # Основание
        for i in range(N):
            start, end = verts[i], verts[(i + 1) % N]
            ax.plot(*zip(start, end), color=self.color)

        # Верх
        for i in range(N):
            start, end = verts[N + i], verts[N + (i + 1) % N]
            ax.plot(*zip(start, end), color=self.color)

        # Боковые рёбра
        for i in range(N):
            start, end = verts[i], verts[N + i]
            ax.plot(*zip(start, end), color=self.color)

        # Центр (среднее всех оснований)
        center = np.mean(self.vertices_base, axis=0)
        ax.scatter(*center, color="black")
