# /// script
# dependencies = [
#   "matplotlib",
#   "pyqt6",
# ]
# ///
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.use("qtagg")

    class Dspl:
        def __init__(self, input_array):
            self.input_array = input_array
            self.fig = None
            self.ax = None
            self.create_subplot3D()

        def create_subplot3D(self) -> None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection="3d")
            self.ax.set_xlabel("X Label")
            self.ax.set_ylabel("Y Label")
            self.ax.set_zlabel("Z Label")

        def show(self) -> None:
            self.ax.set_box_aspect([1, 1, 1])  # убираем искажение пространства
            for obj in self.input_array:
                if hasattr(obj, "show") and callable(obj.show):
                    # Если объект имеет свой метод show, используем его
                    obj.show(self.ax)
                else:
                    # Для объектов без метода show пытаемся получить вершины и ребра
                    try:
                        vertices = obj.get_vertices()
                        edges = obj.get_edges()
                        # Отображаем вершины
                        self.ax.scatter(
                            vertices[:, 0],
                            vertices[:, 1],
                            vertices[:, 2],
                            color="b",
                        )
                        # Отображаем ребра
                        for edge in edges:
                            start, end = vertices[edge[0]], vertices[edge[1]]
                            self.ax.plot(
                                [start[0], end[0]],
                                [start[1], end[1]],
                                [start[2], end[2]],
                                color="r",
                            )
                    except AttributeError:
                        print(
                            f"Объект {type(obj)} не поддерживает отображение"
                        )
            plt.show()
except ImportError as e:
    print(
        f"{e}\nIstall threedtool with option plotting:\npip install threedtool[plotting]"
    )
