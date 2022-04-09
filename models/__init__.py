from functools import partial
from typing import Any, Tuple
from enum import Enum

from .models import (
    CNNModel1,
    CNNModel2,
    CNNModel3,
    CNNModel4,
    CNNModel5
)

class CNNModelAbstract:
    """docstring for CNNModel."""
    num_classes: int
    input_shape: Tuple[int, int, int]
    model: Any


class Models(Enum):
    """Docstring for Models."""
    MODEL_1 = CNNModel1
    MODEL_2 = CNNModel2
    MODEL_3 = CNNModel3
    MODEL_4 = CNNModel4
    MODEL_5 = CNNModel5
    