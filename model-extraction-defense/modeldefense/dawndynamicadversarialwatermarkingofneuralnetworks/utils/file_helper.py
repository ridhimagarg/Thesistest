import os
from typing import Dict, Any, List, Tuple

def load_file(file_path: str) -> List[Tuple]:
    with open(file_path, "rb") as f:
        return pickle.load(f)