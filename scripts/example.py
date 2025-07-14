import threedtool as tdt
from threedtool.core.prism import Prism
import numpy as np
from threedtool.display import Dspl

line = tdt.Line3([[0, 1, 0], [-1, 1, 0]], length=2)
sp = tdt.Sphere(tdt.Point3([1, 0, 0]), radius=1)
cb = tdt.Cuboid(rotation=tdt.rot_y(23) @ tdt.rot_z(24))
base = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.5, -3, 0.0],
        [1.0, 0.0, 0.0],
        [0.6, 4, 0.0],
    ]
)
plane = tdt.Plane([1, 1, 1, 0])
height_vec = np.array([0.0, 0.0, 2.0])  # высота призмы 2 по Z

# Создаём призму
pr = Prism(base, height_vec, color="green")
dspl = Dspl([line, sp, pr, cb, plane])
print(tdt.intersect(sp, cb))
dspl.show()
