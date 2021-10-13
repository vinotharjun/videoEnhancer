from .basic_imports import *
from .torch_imports import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")