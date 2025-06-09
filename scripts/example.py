import threedtool as tdt
line = tdt.Line3([[0, 1, 0], [-1, 1, 0]], length=2)
sp = tdt.Sphere(tdt.Point3([1, 0, 0]), radius=1)
sp2 = tdt.Sphere(tdt.Point3([2, 0, 0]), radius=1)
cb = tdt.Cuboid(rotation=tdt.rot_y(23) @ tdt.rot_z(24))
print(tdt.find_intersections([sp, cb, sp2, line]))
dspl = tdt.Dspl([line, sp, cb, sp2])
dspl.show()
line_2 = tdt.Line3([[-1, 0, 0], [1, 1, 0]], length=2)
