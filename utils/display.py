import time
import cv2


def open_window(window_name, title, width=None, height=None):
    """Open the display window."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowTitle(window_name, title)
    if width and height:
        cv2.resizeWindow(window_name, width, height)


def show_help_text(img, help_text):
    """Draw help text on image."""
    cv2.putText(img, help_text, (11, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,
                (32, 32, 32), 4, cv2.LINE_AA)
    cv2.putText(img, help_text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,
                (240, 240, 240), 1, cv2.LINE_AA)
    return img


def show_fps(img, fps):
    """Draw FPS number at top-left corner of the image."""
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    fps_text = f'FPS: {fps:.2f}'
    cv2.putText(img, fps_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, fps_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
    return img


def set_display(window_name, full_scrn):
    """Set display window to either full screen or normal."""
    if full_scrn:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_NORMAL)


class FpsCalculator:
    """Helper class for calculating frames-per-second (FPS)."""

    def __init__(self, decay_factor=0.95):
        self.fps = 0.0
        self.tic = time.time()
        self.decay_factor = decay_factor

    def update(self):
        toc = time.time()
        curr_fps = 1.0 / (toc - self.tic)
        self.fps = curr_fps if self.fps == 0.0 else self.fps
        self.fps = self.fps * self.decay_factor + curr_fps * (1 - self.decay_factor)
        self.tic = toc
        return self.fps

    def reset(self):
        self.fps = 0.0


class ScreenToggler:
    """Helper class for toggling between non-fullscreen and fullscreen."""

    def __init__(self):
        self.full_scrn = False

    def toggle(self, window_name):
        self.full_scrn = not self.full_scrn
        set_display(window_name, self.full_scrn)


def draw_hud(
    img,
    lines,
    x=10,
    y=20,
    dy=18,
    font_scale=0.6,
    fg=(240, 240, 240),
    bg=(20, 20, 20),
    thickness=1,
    anchor='rt'  # 'lt' = kiri atas, 'rt' = kanan atas
):
    """Draw multiple lines of HUD text (same style as show_fps)."""
    font = cv2.FONT_HERSHEY_PLAIN
    line_type = cv2.LINE_AA
    h, w = img.shape[:2]

    for i, line in enumerate(lines):
        yy = y + i * dy
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        # Hitung posisi teks
        if anchor in ('rt', 'top-right'):
            xx = w - x - tw
        else:
            xx = x

        # background stroke
        cv2.putText(img, line, (xx + 1, yy + 1), font, font_scale, bg, 4, line_type)
        # foreground text
        cv2.putText(img, line, (xx, yy), font, font_scale, fg, thickness, line_type)

    return img
