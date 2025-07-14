import numpy as np

import threedtool as tdt
from threedtool.display import Dspl

if __name__ == "__main__":
    o = tdt.Origin()
    center = np.array([0, 0, 0])
    lwh = np.array([1, 2, 3])
    cb = tdt.Cuboid(center, lwh)
    cb.rotate_x(np.pi / 3)
    dp = Dspl([cb, o])
    dp.show()
