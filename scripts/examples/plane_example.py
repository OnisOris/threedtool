import threedtool as tdt

if __name__ == "__main__":
    o = tdt.Origin()
    plane = tdt.Plane()
    dp = tdt.Dspl([plane, o])
    dp.show()