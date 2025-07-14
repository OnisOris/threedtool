import numpy as np

import threedtool as tdt
from threedtool.core.prism import Prism
from threedtool.display import Dspl

cb = tdt.Cuboid(rotation=tdt.rot_y(23) @ tdt.rot_z(24))
base = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
    ]
)
height_vec = np.array([0.0, 0.0, 2.0])  # высота призмы 2 по Z

# Создаём призму
pr = Prism(base, height_vec, color="green")
dspl = Dspl([cb, pr])
# print(tdt.intersect(cb, pr))
dspl.show()
