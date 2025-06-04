import numpy as np
from numpy.typing import NDArray
from threedtool.core.basefigure import Point3, Vector3

class Origin:
    """
    Система координат
    """
    def __init__(self,
                 o: Point3 = Point3([0, 0, 0]),
                 i: Vector3 = Vector3([1, 0, 0]),
                 j: Vector3 = Vector3([0, 1, 0]),
                 k: Vector3 = Vector3([0, 0, 1])):
        self.o: Point3 = o
        self.i: Vector3 = i
        self.j: Vector3 = j
        self.k: Vector3 = k

    def show(self, ax):
        ax.quiver(*self.o, *self.i, color="red")
        ax.quiver(*self.o, *self.j, color="green")
        ax.quiver(*self.o, *self.k, color="blue")
