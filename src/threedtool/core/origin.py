import numpy as np
from numpy.typing import NDArray
from threedtool.core.basefigure import Point3, Vector3


class Origin:
    """
    Система координат с адаптивным отображением
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
        # Приглушённые цвета осей [реализация по вашему запросу]
        color_i = (0.8, 0, 0)    # тёмно-красный
        color_j = (0, 0.6, 0)     # тёмно-зелёный
        color_k = (0, 0, 0.8)     # тёмно-синий

        # Отрисовка осей
        ax.quiver(*self.o, *self.i, color=color_i)
        ax.quiver(*self.o, *self.j, color=color_j)
        ax.quiver(*self.o, *self.k, color=color_k)

        # Адаптивная подпись "O" [реализация через аннотацию]
        ax.annotate('O',
                    xy=self.o,               # Точка привязки (начало координат)
                    xytext=(-8, -8),         # Смещение в точках экрана
                    textcoords='offset points',  # Система координат для смещения
                    fontsize=12,
                    ha='center',             # Центрирование текста
                    va='center',
                    bbox=dict(               # Фон для читаемости
                        boxstyle='round,pad=0.3',
                        facecolor='white',
                        alpha=0.7,
                        edgecolor='none'
                    ))
