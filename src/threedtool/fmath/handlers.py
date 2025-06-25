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
    is_intersecting_prism_prism,
    is_intersecting_prism_cuboid,
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


@register_intersection(Prism, Prism)
def intersect_prism_prism(a: Prism, b: Prism) -> bool:
    return is_intersecting_prism_prism(a, b)


@register_intersection(Prism, Cuboid)
def intersect_prism_cuboid(a: Prism, b: Cuboid) -> bool:
    return is_intersecting_prism_cuboid(a, b)

