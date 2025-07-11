from typing import Annotated, Any

from numpy.typing import NDArray

# Аннотации для векторов различных размеров
Array17 = Annotated[NDArray[Any], (17,)]
Array14 = Annotated[NDArray[Any], (14,)]
Array7 = Annotated[NDArray[Any], (7,)]
Array6 = Annotated[NDArray[Any], (6,)]
Array4 = Annotated[NDArray[Any], (4,)]
Array3 = Annotated[NDArray[Any], (3,)]
Array2 = Annotated[NDArray[Any], (2,)]

# Аннотации для матриц
Array2x2 = Annotated[NDArray[Any], (2, 2)]
Array3x3 = Annotated[NDArray[Any], (3, 3)]
Array4x4 = Annotated[NDArray[Any], (4, 4)]
