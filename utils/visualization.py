"""visualization.py

The BBoxVisualization class implements drawing of nice looking
bounding boxes based on object detection results.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

# Constants
ALPHA = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
TEXT_THICKNESS = 1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def gen_colors(num_colors: int) -> List[Tuple[int, int, int]]:
    """Generate deterministic different BGR colors."""
    import random
    import colorsys

    if num_colors <= 0:
        return []

    hsvs = [[float(x) / num_colors, 1.0, 0.7] for x in range(num_colors)]
    random.seed(1234)  # deterministic
    random.shuffle(hsvs)
    rgbs = [colorsys.hsv_to_rgb(*x) for x in hsvs]
    # convert to BGR uint8
    bgrs = [(int(r*255), int(g*255), int(b*255)) for (r, g, b) in rgbs]
    # swap to BGR order
    bgrs = [(b, g, r) for (r, g, b) in bgrs]
    return bgrs


def draw_boxed_text(img: np.ndarray, text: str, topleft: Tuple[int, int], color: Tuple[int, int, int]) -> np.ndarray:
    """Draw a translucent boxed text."""
    assert img.dtype == np.uint8
    img_h, img_w = img.shape[:2]
    if topleft[0] >= img_w or topleft[1] >= img_h:
        return img

    margin = 3
    (tw, th), _ = cv2.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICKNESS)
    w = tw + margin * 2
    h = th + margin * 2

    # patch background
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[...] = color
    cv2.putText(patch, text, (margin + 1, h - margin - 2), FONT, TEXT_SCALE,
                WHITE, thickness=TEXT_THICKNESS, lineType=cv2.LINE_8)
    cv2.rectangle(patch, (0, 0), (w - 1, h - 1), BLACK, thickness=1)

    # clip if near border
    w = min(w, img_w - topleft[0])
    h = min(h, img_h - topleft[1])

    roi = img[topleft[1]:topleft[1] + h, topleft[0]:topleft[0] + w, :]
    cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, roi, 1 - ALPHA, 0, roi)
    return img


class BBoxVisualization:
    """Nice drawing of bounding boxes.

    Args:
        cls_dict:   dict {class_id: class_name}
        colors:     optional list of BGR tuples indexed by class_id
        color_map:  optional dict {class_name: (B, G, R)}; overrides 'colors'
        default_color: fallback BGR if class_id/name out of range/map
    """

    def __init__(
        self,
        cls_dict: Dict[int, str],
        colors: Optional[List[Tuple[int, int, int]]] = None,
        color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
        default_color: Tuple[int, int, int] = (255, 255, 255),
    ):
        self.cls_dict = dict(cls_dict)
        self.colors = list(colors) if colors is not None else gen_colors(len(self.cls_dict))
        self.color_map = dict(color_map) if color_map is not None else {}
        self.default_color = tuple(default_color)

    def _color_for(self, class_id: int) -> Tuple[int, int, int]:
        name = self.cls_dict.get(int(class_id))
        if name is not None and name in self.color_map:
            return self.color_map[name]
        if len(self.colors) > 0:
            return self.colors[int(class_id) % len(self.colors)]
        return self.default_color

    def draw_bboxes(
        self,
        img: np.ndarray,
        boxes: List[Tuple[float, float, float, float]],
        confs: List[float],
        clss: List[int],
        thickness: int = 2,
        font_scale: float = TEXT_SCALE,
    ) -> np.ndarray:
        """Draw detected bounding boxes on the original image."""
        for bb, cf, cl in zip(boxes, confs, clss):
            cl = int(cl)

            # aman untuk float -> int dan out-of-bound
            x_min, y_min, x_max, y_max = map(int, bb[:4])
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = max(x_min + 1, x_max)
            y_max = max(y_min + 1, y_max)

            color = self._color_for(cl)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

            cls_name = self.cls_dict.get(cl, f'CLS{cl}')
            try:
                score = float(cf)
            except Exception:
                score = 0.0
            label = f'{cls_name} {score:.2f}'

            # geser label sedikit ke dalam bbox
            txt_loc = (max(x_min + 2, 0), max(y_min + 2, 0))

            # sementara gunakan TEXT_SCALE global agar boxed text konsisten;
            # kalau mau mengikuti argumen 'font_scale', bisa ganti konstanta di draw_boxed_text.
            img = draw_boxed_text(img, label, txt_loc, color)

        return img
