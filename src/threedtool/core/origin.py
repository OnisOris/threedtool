import numpy as np
from numpy.typing import NDArray
from threedtool.core.basefigure import Point3, Vector3

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
        x2d, y2d, _ = proj_transform(*self.o, ax.get_proj())

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
def proj_transform(xs, ys, zs, M):
    """
    Transform the points by the projection matrix *M*.
    """
    vec = _vec_pad_ones(xs, ys, zs)
    return _proj_transform_vec(vec, M)

def _vec_pad_ones(xs, ys, zs):
    if np.ma.isMA(xs) or np.ma.isMA(ys) or np.ma.isMA(zs):
        return np.ma.array([xs, ys, zs, np.ones_like(xs)])
    else:
        return np.array([xs, ys, zs, np.ones_like(xs)])

def _proj_transform_vec(vec, M):
    vecw = np.dot(M, vec.data)
    w = vecw[3]
    txs, tys, tzs = vecw[0]/w, vecw[1]/w, vecw[2]/w
    if np.ma.isMA(vec[0]):  # we check each to protect for scalars
        txs = np.ma.array(txs, mask=vec[0].mask)
    if np.ma.isMA(vec[1]):
        tys = np.ma.array(tys, mask=vec[1].mask)
    if np.ma.isMA(vec[2]):
        tzs = np.ma.array(tzs, mask=vec[2].mask)
    return txs, tys, tzs