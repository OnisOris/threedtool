import threedtool as tdt
from threedtool.core.prism import Prism
import numpy as np
from threedtool.fmath.dispatch import (
    _INTERSECT_HANDLERS,
    register_intersection,
)

print(_INTERSECT_HANDLERS)

line = tdt.Line3([[0, 1, 0], [-1, 1, 0]], length=2)
sp = tdt.Sphere(tdt.Point3([1, 0, 0]), radius=1)
# sp2 = tdt.Sphere(tdt.Point3([2, 0, 0]), radius=1)
# cb = tdt.Cuboid(rotation=tdt.rot_y(23) @ tdt.rot_z(24))
# base = np.array(
#     [
#         [0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0],
#         [0.5, 0.866, 0.0],
#     ]
# )
# height_vec = np.array([0.0, 0.0, 2.0])  # высота призмы 2 по Z

# Создаём призму
# pr = Prism(base, height_vec, color="green")
dspl = tdt.Dspl([line, sp])
# print(tdt.find_intersections([sp, cb, sp2, line, pr]))
print(tdt.intersect(sp, line))
dspl.show()
# line_2 = tdt.Line3([[-1, 0, 0], [1, 1, 0]], length=2)
