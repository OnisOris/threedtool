import numpy as np
from numpy.typing import NDArray
from threedtool.core.basefigure import Point3, Vector3
from mpl_toolkits.mplot3d import proj3d  # Добавляем необходимый импорт


class Origin:
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
        color_i = (0.8, 0, 0)
        color_j = (0, 0.6, 0)
        color_k = (0, 0, 0.8)

        # Отрисовка осей
        ax.quiver(*self.o, *self.i, color=color_i)
        ax.quiver(*self.o, *self.j, color=color_j)
        ax.quiver(*self.o, *self.k, color=color_k)

        # Преобразование 3D-координат в 2D-экранные координаты
        x2d, y2d, _ = proj3d.proj_transform(*self.o, ax.get_proj())

        # Создание аннотации в 2D-пространстве
        ax.annotate(
            'O',
            xy=(x2d, y2d),  # 2D-координаты на экране
            xytext=(-8, -8),  # Смещение в точках экрана
            textcoords='offset points',
            fontsize=12,
            ha='center',
            va='center',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                alpha=0.7,
                edgecolor='none'
            )
        )