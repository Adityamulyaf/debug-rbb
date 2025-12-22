# utils/model_registry.py

from typing import Dict, List, Tuple

# BGR, bukan RGB
PALETTES_BY_MODEL: Dict[str, Dict[str, Tuple[int,int,int]]] = {
    "pole2": {
        "greenpole": (0, 255, 0),
        "redpole": (0, 0, 255),
    },

    "avoid_640_352_tiny_2": {
        "blackball": (0, 0, 0),
        "greenball": (0, 255, 0),
        "greenlight": (0, 100, 0),
        "redball": (0, 0, 255),
        "redlight": (0, 0, 100),
    },

    "speed": {
        "blackball": (0, 255, 255),
        "greenball": (0, 0, 255),
        "greenlight": (0, 0, 255),
        "redball": (0, 255, 0),
        "redlight": (0, 255, 0),
   }
}

NAMES_BY_MODEL: Dict[str, List[str]] = {
    "pole2":  ["greenpole", "redpole"],
    "avoid_640_352_tiny_2": ["blackball", "greenball", "greenlight", "redball", "redlight"],
    "speed": ["blackball", "greenball", "greenlight", "redball", "redlight"],
}

def get_cls_dict_by_model(model_key: str, override_num: int = None) -> dict:
    """
    Kembalikan mapping {id: name} sesuai model.
    Jika override_num diberikan (mis. --category-num), batasi jumlah kelas.
    """
    names = NAMES_BY_MODEL.get(model_key)
    if not names:
        raise SystemExit(f"Unknown model_key '{model_key}'. Tambahkan di NAMES_BY_MODEL.")
    if override_num is not None:
        names = names[:override_num]
    return {i: n for i, n in enumerate(names)}

def get_color_map_by_model(model_key: str) -> dict:
    """Kembalikan mapping {class_name: (B,G,R)}; bisa kosong kalau belum di-setup."""
    return PALETTES_BY_MODEL.get(model_key, {})
