# import numpy as np
# from .threeDTool import normal_of_triangle
# from numpy import sqrt
#
#
# def full_vstack(vector: list | np.ndarray) -> np.ndarray:
#     """
#     Функция применяет vstack() метод к каждому элементу массива
#     :param vector: Массив
#     :type vector: list or np.ndarray
#     """
#     entry_point = vector[0]
#     for element in vector:
#         entry_point = np.vstack([entry_point, element])
#     return entry_point
#
#
# class Plane:
#     """
#     Класс, реализующий уравнение плоскости типа:
#     ax + by + cz + d = 0
#     """
#
#     def __init__(self, a: float = 0, b: float = 0, c: float = 1, d: float = 0):
#         """
#         :param a: Коэффициент a
#         :type a: float
#         :param b: Коэффициент b
#         :type b: float
#         :param c: Коэффициент c
#         :type c: float
#         :param d: Коэффициент d
#         :type d: float
#         """
#         self._a = a
#         self._b = b
#         self._c = c
#         self._d = d
#
#     #
#     def create_plane3(self, matrix: list | np.ndarray) -> np.ndarray:
#         """
#         Функция создает плоскость из трех точек
#         :param matrix: Матрица размеров 3x3
#         :type matrix: np.array
#         :return: np.array
#         """
#         G = np.array([[-1], [-1], [-1]])
#         abc = np.dot(np.linalg.inv(matrix), G)
#         coefficients = np.array([abc.T[0][0], abc.T[0][1], abc.T[0][2], 1])
#         self._a = coefficients[0]
#         self._b = coefficients[1]
#         self._c = coefficients[2]
#         self._d = coefficients[3]
#         return coefficients
#
#     @property
#     def a(self):
#         return self._a
#
#     @property
#     def b(self):
#         return self._b
#
#     @property
#     def c(self):
#         return self._c
#
#     @property
#     def d(self):
#         return self._d
#
#     @a.setter
#     def a(self, a):
#         self._a = a
#
#     @b.setter
#     def b(self, b):
#         self._b = b
#
#     @c.setter
#     def c(self, c):
#         self._c = c
#
#     @d.setter
#     def d(self, d):
#         self._d = d
#
#     def coeffs(self) -> np.ndarray:
#         """
#         Функция возвращает коэффициенты линии
#         :return: ndarray[float]
#         """
#         return np.array([self._a, self._b, self._c, self._d])
#
#     def get_N(self) -> np.ndarray:
#         """
#         Возвращает координаты вектора нормали плоскости np.array([a, b, c])
#         :return: np.ndarray
#         """
#         return np.array([self._a, self._b, self._c])
#
#     def create_plane_from_triangle(self, triangle: np.ndarray | list,
#                                    point: int = 1,
#                                    create_normal: bool = False) -> None:
#         """
#         Данная функция принимает массив 4x3. Строка 1 - координаты вектора нормали (пишутся координаты только второй
#         точки, первая исходит из нуля).
#         Строки 2, 3, 4 - координаты вершин треугольника формата [x, y, z]
#         На основе четырех точек создается плоскость и коэффициенты a, b, c, d записываются в поля объекта класса Plane
#         Согласно параграфу 123 "Уравнение плоскости", плоскость, проходящая через точку M_0 (x_0, y_0, z_0) и
#         перпендикулярному вектору N{a, b, c}, представляется уравнением:
#         A(x - x_0) + B(y - y_0) + C(z - z_0) = 0 или Ax + By + Cz + D = 0,
#         где D = -Ax_0 - By_o - Cz_0
#         Поэтому мы берем первую вершину треугольника (по умолчанию point=1) и вектор нормали и на основе
#         него создаем уравнение плоскости.
#         Если create_normal = True, то это значит, что на вход идет матрица 3x3 с вершинами треугольника, тогда вектор
#         нормали vector_N создается автоматически
#         :param triangle: Массив 4x3
#         :type triangle: np.ndarray | list
#         :param point: номер вершины
#         :type point: int
#         :param create_normal: Создавать ли нормаль у треугольника
#         :type create_normal: bool
#         :return: None
#         """
#
#
#         if create_normal:
#             vector_N = normal_of_triangle(triangle[0], triangle[1], triangle[2])
#         else:
#             vector_N = triangle[0]
#         mod = sqrt(vector_N[0] ** 2 + vector_N[1] ** 2 + vector_N[2] ** 2)
#
#         if mod != 1:
#             a, b, c = vector_N[0] / mod, vector_N[1] / mod, vector_N[2] / mod
#         else:
#             a, b, c = vector_N[0], vector_N[1], vector_N[2]
#         first_point = triangle[point]
#         self._a, self._b, self._c = a, b, c
#         #  Вычисление коэффициента D
#         self._d = - self._a * first_point[0] - self._b * first_point[1] - self._c * first_point[2]
#
#     ###################
#     #       0         # hight
#     ################### ->
#     # lenth          x
#     def show(self) -> None:
#         """
#         Данная функция отображает плсокость
#         :return: None
#         """
#         hight = 20  # Высота прямоугольника
#         lenth = 20  # Длина прямоугольника
#         x1 = -lenth / 2
#         y1 = hight / 2
#         z1 = self.projection_z(x1, y1)
#         if z1 == "Uncertainty z":
#             z1 = hight / 2
#         point1 = np.array([x1, y1, z1])
#         x2 = lenth / 2
#         y2 = hight / 2
#         z2 = self.projection_z(x2, y2)
#         if z2 == "Uncertainty z":
#             z2 = hight / 2
#         point2 = np.array([x2, y2, z2])
#         x3 = lenth / 2
#         y3 = -hight / 2
#         z3 = self.projection_z(x3, y3)
#         if z3 == "Uncertainty z":
#             z3 = -hight / 2
#         point3 = np.array([x3, y3, z3])
#         x4 = -lenth / 2
#         y4 = -hight / 2
#         z4 = self.projection_z(x4, y4)
#         if z4 == "Uncertainty z":
#             z4 = -hight / 2
#         point4 = np.array([x4, y4, z4])
#
#         matrix_x_y_z = full_vstack([point1, point2, point3, point4, point1]).T
#         points = np.array([[x1, y1, 0], [x2, y2, 0], [x3, y3, 0], [x4, y4, 0], [x1, y1, 0]])
#         matrix_points = full_vstack(points).T
#         fig = plt.figure()
#         ax = fig.add_subplot(projection='3d')
#         ax.set_xlabel("X", fontsize=15, color='red')
#         ax.set_ylabel("Y", fontsize=15, color='green')
#         ax.set_zlabel("Z", fontsize=15, color='blue')
#
#         ax.plot(matrix_x_y_z[0], matrix_x_y_z[1], matrix_x_y_z[2], c='r')
#         ax.plot(matrix_points[0], matrix_points[1], matrix_points[2], c='g')
#
#         points_proj = np.array([[points[0], point1],
#                                 [points[1], point2],
#                                 [points[2], point3],
#                                 [points[3], point4]])
#         figures = []
#         for i in range(4):
#             figures.append(ax.plot(points_proj[i].T[0], points_proj[i].T[1], points_proj[i].T[2], c='b'))
#         plt.show()
#
#     def projection_z(self, point_x, point_y):
#
#         """
#         Данная функция берет координаты плоскости x и y, и по ним ищет точку z на плоскости класса Plane.
#         Рассмотрено три случая:
#         1) Только c = 0, тогда точка z на плоскости может быть любой в координатах x, y.
#         Поэтому ее можно задать вручную.
#         2) b и a = 0, тогда плоскость параллельна осям y и x, а z = const, поэтому для нахождения z достаточно взять
#         точку O(0, 0, z), тогда z = -D/C
#         3) Нормальный вариант, когда ни один из коэффициентов не равен нулю.
#         Возвращает Point z or "Uncertainty"
#         :param point_x: Точка x
#         :param point_y: Точка y
#         :return: np.ndarray | str
#         """
#         if self._c == 0:
#             return "Uncertainty z"
#         elif self._b == 0 and self._a == 0:
#             return -self._d / self._c
#         else:
#             point_z = (-self._a * point_x - self._b * point_y - self._d) / self._c
#             return point_z
#
#     def projection_y(self, point_x, point_z):
#
#         """
#         Данная функция берет координаты плоскости x и z, и по ним ищет точку y на плоскости класса Plane.
#         Рассмотрено три случая:
#         1) Только b = 0, тогда точка y на плоскости может быть любой в координатах x, z.
#         Поэтому ее можно задать вручную.
#         2) c и a = 0, тогда плоскость параллельна осям z и x, а y = const, поэтому для нахождения y достаточно взять
#         точку O(0, y, 0), тогда y = -D/B
#         3) Нормальный вариант, когда ни один из коэффициентов не равен нулю.
#         Возвращает Point y or "Uncertainty"
#         :param point_x: Точка x
#         :param point_z: Точка z
#         :return: np.ndarray | str
#         """
#         if self._b == 0:
#             return "Uncertainty y"
#         elif self._a == 0 and self._c == 0:
#             return -self._d / self._b
#         else:
#             point_y = (-self._a * point_x - self._c * point_z - self._d) / self._b
#         return point_y
#
#     def projection_x(self, point_y, point_z):
#
#         """
#         Данная функция берет координаты плоскости x и z, и по ним ищет точку y на плоскости класса Plane.
#         Рассмотрено три случая:
#         1) Только a = 0, тогда точка y на плоскости может быть любой в координатах y, z.
#         Поэтому ее можно задать вручную.
#         2) a и c = 0, тогда плоскость параллельна осям x и z, а x = const, поэтому для нахождения x достаточно взять
#         точку O(x, 0, 0), тогда x = -D/B
#         3) Нормальный вариант, когда ни один из коэффициентов не равен нулю.
#         Возвращает Point x or "Uncertainty"
#         :param point_y: Точка y
#         :param point_z: Точка z
#         :return: np.ndarray | str
#         """
#         if self._a == 0:
#             return "Uncertainty x"
#         elif self._b == 0 and self._c == 0:
#             return -self._d / self._a
#         else:
#             point_x = (-self._b * point_y - self._c * point_z - self._d) / self._a
#             return point_x
#
#
#
