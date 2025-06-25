from __future__ import annotations

import numpy as np

from .fmath import project


def is_intersecting_line_line(a, b) -> bool:
    rank = np.linalg.matrix_rank(np.vstack([a, b]))
    return rank == 2


def is_intersecting_cuboid_line(a, b) -> bool:
    """
    Проверяет пересечение кубоида и бесконечной прямой
    """
    # Переводим прямую в локальную систему кубоида
    R = a.rotation  # 3×3
    C = a.center  # 3,
    A = b.abc
    v = b.p  # 3, нормирован

    A_loc = R.T.dot(A - C)
    v_loc = R.T.dot(v)

    half = a.length_width_height / 2.0

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


def is_intersecting_sphere_sphere(a, b):
    center_dist_sq = np.sum((a.center - b.center) ** 2)
    radius_sum = a.radius + b.radius
    return center_dist_sq <= radius_sum**2


def is_intersecting_cuboid_cuboid(a, b):
    vertices1 = a.get_vertices()
    vertices2 = b.get_vertices()

    axes1 = a.get_axes()
    axes2 = b.get_axes()

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
            return False
    return True


def is_intersecting_sphere_cuboid(a, b) -> bool:
    """
    Проверяет пересечение кубоида и сферы

    :param a: Объект сферы
    :type a: Sphere
    :param b: Объект кубоида
    :type b: Cuboid
    :return: True если есть пересечение, иначе False
    :rtype: bool
    """
    # Преобразование центра сферы в локальную систему кубоида
    center_local = a.rotation.T @ (b.center - a.center)

    # Размеры кубоида в локальной системе
    half_sizes = a.length_width_height / 2.0

    # Находим ближайшую точку в локальных координатах
    closest_local = np.clip(center_local, -half_sizes, half_sizes)

    # Вычисляем расстояние между точками
    distance_sq = np.sum((center_local - closest_local) ** 2)

    # Проверяем пересечение
    return distance_sq <= (b.radius**2)


def is_intersecting_line_sphere(sphere, line) -> bool:
    """
    Проверяет пересечение бесконечной прямой и сферы.

    Расстояние от центра сферы до прямой:
        dist = || (C - A) × v ||
    где A — точка на прямой, v — единичный направляющий вектор прямой,
          C — центр сферы.

    :param sphere: Объект сферы
    :type sphere: Sphere
    :param line: объект линии
    :type line: Line3
    :return: True, если пересекаются, иначе False
    """
    # точка A на прямой
    A = line.abc
    # направляющий вектор v (должен быть нормирован)
    v = line.p

    # вектор из A в центр сферы
    CA = sphere.center - A
    # векторное произведение и квадрат нормы
    cross = np.cross(CA, v)
    dist_sq = np.dot(cross, cross)  # ||cross||^2

    return dist_sq <= (sphere.radius**2)


def is_intersecting_prism_cuboid(prism, cuboid) -> bool:
    return is_intersecting_prism_prism(prism, cuboid)


def is_intersecting_prism_prism(prism1, prism2) -> bool:
    """SAT: тест разделяющих осей между двумя призмами"""
    V1 = prism1.get_vertices()
    V2 = prism2.get_vertices()
    axes = prism1.get_face_normals() + prism2.get_face_normals()

    # собрать рёбра каждой призмы
    # основание размер N, top так же
    N = len(prism1.vertices_base)
    edges1 = [V1[i + 1] - V1[i] for i in range(N - 1)] + [V1[0] - V1[N - 1]]
    edges2 = [V2[i + 1] - V2[i] for i in range(N - 1)] + [V2[0] - V2[N - 1]]
    for e1 in edges1:
        for e2 in edges2:
            cp = np.cross(e1, e2)
            if np.linalg.norm(cp) > 1e-8:
                axes.append(cp / np.linalg.norm(cp))

    for ax in axes:
        min1, max1 = project(V1, ax)
        min2, max2 = project(V2, ax)
        if max1 < min2 or max2 < min1:
            return False
    return True


