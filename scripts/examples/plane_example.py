import threedtool as tdt
from threedtool.display import Dspl

if __name__ == "__main__":
    o = tdt.Origin()
    plane = tdt.Plane()
    dp = Dspl([plane, o])
    dp.show()