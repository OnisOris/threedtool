from threedtool.core.cuboid import Cuboid
from threedtool.core.line import Line3
from threedtool.core.prism import Prism
from threedtool.core.sphere import Sphere
from threedtool.fmath.dispatch import register_intersection
from threedtool.fmath.intersections import (
    is_intersecting_cuboid_cuboid,
    is_intersecting_cuboid_line,
    is_intersecting_line_line,
    is_intersecting_line_sphere,
    is_intersecting_sphere_cuboid,
    is_intersecting_sphere_sphere,
)


# T
@register_intersection(Cuboid, Cuboid)
def intersect_cuboid_cuboid(a: Cuboid, b: Cuboid) -> bool:
    return is_intersecting_cuboid_cuboid(a, b)


# T
@register_intersection(Cuboid, Sphere)
def intersect_cuboid_sphere(a: Sphere, b: Cuboid) -> bool:
    return is_intersecting_sphere_cuboid(a, b)


# T
@register_intersection(Sphere, Sphere)
def intersect_sphere_sphere(a: Sphere, b: Sphere) -> bool:
    return is_intersecting_sphere_sphere(a, b)


# T
@register_intersection(Line3, Sphere)
def intersect_line_sphere(ln: Line3, sp: Sphere) -> bool:
    return is_intersecting_line_sphere(sp, ln)


# T
@register_intersection(Cuboid, Line3)
def intersect_cuboid_line(a: Cuboid, b: Line3) -> bool:
    return is_intersecting_cuboid_line(a, b)


# T
@register_intersection(Line3, Line3)
def intersect_line_line(a: Line3, b: Line3) -> bool:
    return is_intersecting_line_line(a, b)


# @register_intersection(Prism, Prism)
# def intersect_prism_prism(a: Prism, b: Prism) -> bool:
#     return a.is_intersecting_prism(b)


# @register_intersection(Prism, Cuboid)
# def intersect_prism_cuboid(pr: Prism, cb: Cuboid) -> bool:
#     # Представляем кубоид как призму того же основания и высоты
#     base = cb.get_vertices()[:4]
#     height_vec = cb.get_axes()[2] * cb.height
#     return pr.is_intersecting_prism(Prism(base, height_vec))


# @register_intersection(Prism, Sphere)
# def intersect_prism_sphere(pr: Prism, sp: Sphere) -> bool:
#     # простой тест — любая вершина в сфере
#     V = pr.get_vertices()
#     dists = np.linalg.norm(V - sp.center, axis=1)
#     return np.any(dists <= sp.radius)


# @register_intersection(Prism, Line3)
# def intersect_prism_line(pr: Prism, ln: Line3) -> bool:
#     # TODO: точная проверка, пока грубо отрезки рёбер
#     V = pr.get_vertices()
#     N = len(pr.vertices_base)
#     edges = [(V[i], V[(i + 1) % N]) for i in range(N * 2)]
#     for A, B in edges:
#         # проверка пересечения отрезка AB и прямой ln
#         # ... ваш алгоритм ...
#         pass
#     return False
