import threedtool as tdt

line = tdt.Line3([[0, 0, 0], [1, 2, 1]], length=2)
sp = tdt.Sphere(tdt.Point3([0, 0, 0]), radius=2)
dspl = tdt.Dspl([line, sp])
print(line)
dspl.show()
