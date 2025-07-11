try:
    import matplotlib
except ImportError:
    raise ImportError(
        "Display module requires 'matplotlib' and 'PyQt6'. "
        "Install with: pip install threedtool[plotting]"
    )
from .dspl import Dspl
