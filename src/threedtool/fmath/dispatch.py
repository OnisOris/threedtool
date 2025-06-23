# threedtool/fmath/dispatch.py
from typing import Callable, Dict, Tuple, Type, Any, List, Union

# Словарь: ключ — (класс A, класс B), значение — функция intersect(a, b) -> bool
_INTERSECT_HANDLERS: Dict[Tuple[Type, Type], Callable[[Any, Any], bool]] = {}


def register_intersection(type_a: Type, type_b: Type):
    """
    Декоратор для регистрации функции пересечения двух типов.

    Гарантирует, что при вызове intersect(a, b) или intersect(b, a)
    найдётся нужная функция.
    """

    def decorator(fn: Callable[[Any, Any], bool]):
        _INTERSECT_HANDLERS[(type_a, type_b)] = fn
        _INTERSECT_HANDLERS[(type_b, type_a)] = lambda b, a: fn(a, b)
        return fn

    return decorator


def intersect(a: Any, b: Any) -> bool:
    """
    Вызывает зарегистрированный обработчик пересечения для типов a и b.
    """
    key = (type(a), type(b))
    try:
        handler = _INTERSECT_HANDLERS[key]
    except KeyError:
        raise NotImplementedError(
            f"No intersection handler for {type(a).__name__} ↔ {type(b).__name__}"
        )
    return handler(a, b)


def find_intersections(
    objects: List[object],
) -> Dict[int, Union[List[int], str]]:
    intersections: Dict[int, Union[List[int], str]] = {}

    for i, obj1 in enumerate(objects):
        # if not hasattr(obj1, "intersects_with"):
        #     intersections[i] = "unknown"
        #     continue

        intersections[i] = []
        for j, obj2 in enumerate(objects):
            if i == j:
                continue
            try:
                if intersect(obj1, obj2):
                    intersections[i].append(j)
            except Exception:
                continue
    return intersections
