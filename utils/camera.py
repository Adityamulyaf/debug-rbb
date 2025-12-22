"""camera.py

This code implements the Camera class, which encapsulates code to
handle IP CAM, USB webcam or the Jetson onboard camera.  In
addition, this Camera class is further extended to take a video
file or an image file as input.
"""

import logging
import threading
import subprocess

import numpy as np
import cv2


# The following flag is used to control whether to use a GStreamer
# pipeline to open USB webcam source. If set to False, we just open
# the webcam using cv2.VideoCapture(index) machinery, i.e., relying
# on cv2's built-in function to capture images from the webcam.
USB_GSTREAMER = True


def add_camera_args(parser):
    """Add parser argument for camera options."""
    parser.add_argument('--image', type=str, default=None,
                        help='image file name, e.g. dog.jpg')
    parser.add_argument('--video', type=str, default=None,
                        help='video file name, e.g. traffic.mp4')
    parser.add_argument('--video_looping', action='store_true',
                        help='loop around the video file [False]')
    parser.add_argument('--rtsp', type=str, default=None,
                        help=('RTSP H.264 stream, e.g. '
                              'rtsp://admin:123456@192.168.1.64:554'))
    parser.add_argument('--rtsp_latency', type=int, default=200,
                        help='RTSP latency in ms [200]')
    parser.add_argument('--usb', type=int, default=0,
                        help='USB webcam device id (/dev/video?) [0]')
    parser.add_argument('--gstr', type=str, default=None,
                        help='GStreamer string [None]')
    parser.add_argument('--onboard', type=int, default=None,
                        help='Jetson onboard camera [None]')
    parser.add_argument('--copy_frame', action='store_true',
                        help='copy video frame internally [False]')
    parser.add_argument('--do_resize', action='store_true',
                        help='resize image/video [False]')
    parser.add_argument('--width', type=int, default=640,
                        help='image width [640]')
    parser.add_argument('--height', type=int, default=360,
                        help='image height [360]')
    parser.add_argument('--fps', type=int, default=30,
                        help='camera FPS request [30]')
    return parser


def open_cam_rtsp(uri, width, height, latency):
    """Open an RTSP URI (IP CAM)."""
    gst_elements = subprocess.check_output(['gst-inspect-1.0']).decode()
    if 'omxh264dec' in gst_elements:
        # Use hardware H.264 decoder on Jetson platforms
        gst_str = ('rtspsrc location={} latency={} ! '
                   'rtph264depay ! h264parse ! omxh264dec ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! videoconvert ! '
                   'appsink').format(uri, latency, width, height)
    elif 'avdec_h264' in gst_elements:
        # Otherwise try to use the software decoder 'avdec_h264'
        # NOTE: in case resizing images is necessary, try adding
        #       a 'videoscale' into the pipeline
        gst_str = ('rtspsrc location={} latency={} ! '
                   'rtph264depay ! h264parse ! avdec_h264 ! '
                   'videoconvert ! appsink').format(uri, latency)
    else:
        raise RuntimeError('H.264 decoder not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def _probe_usb(dev, width, height, fps):
    """Try opening USB cam at index dev and validate we can read 1 frame."""
    cap = open_cam_usb(dev, width, height, fps)
    if not cap or not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return None
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None
    return cap


def open_cam_usb(dev, width, height, fps=30):
    """Open a USB webcam."""
    if USB_GSTREAMER:
        gst_str = (
            f'v4l2src device=/dev/video{dev} io-mode=2 ! '
            f'image/jpeg,width={width},height={height},framerate={fps}/1 ! '
            f'jpegdec ! videoconvert ! '
            f'appsink drop=true max-buffers=1 sync=false'
        )
        return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        return cap


def open_cam_gstr(gstr, width, height):
    """Open camera using a GStreamer string.

    Example:
    gstr = 'v4l2src device=/dev/video0 ! video/x-raw, width=(int){width}, height=(int){height} ! videoconvert ! appsink'
    """
    gst_str = gstr.format(width=width, height=height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_onboard(width, height):
    """Open the Jetson onboard camera."""
    gst_elements = subprocess.check_output(['gst-inspect-1.0']).decode()
    if 'nvcamerasrc' in gst_elements:
        # On versions of L4T prior to 28.1, you might need to add
        # 'flip-method=2' into gst_str below.
        gst_str = ('nvcamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    elif 'nvarguscamerasrc' in gst_elements:
        gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1920, height=(int)1080, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=2 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    else:
        raise RuntimeError('onboard camera source not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def grab_img(cam):
    """Run in sub-thread to continuously grab frames."""
    while cam.thread_running:
        _, cam.img_handle = cam.cap.read()
        if cam.img_handle is None:
            # logging.warning('Camera: cap.read() returns None...')
            break
    cam.thread_running = False


class Camera:
    """Camera class which supports reading images from these video sources:

    1. Image (jpg, png, etc.) file, repeating indefinitely
    2. Video file
    3. RTSP (IP CAM)
    4. USB webcam
    5. Jetson onboard camera
    """

    def __init__(self, args):
        self.args = args
        self.is_opened = False
        self.video_file = ''
        self.video_looping = args.video_looping
        self.thread_running = False
        self.img_handle = None
        self.copy_frame = args.copy_frame
        self.do_resize = args.do_resize
        self.img_width = args.width
        self.img_height = args.height
        self.cap = None
        self.thread = None
        self._open()  # try to open the camera

    def _open(self):
        """Open camera based on command line arguments."""
        if self.cap is not None:
            raise RuntimeError('camera is already opened!')
        a = self.args
        if a.image:
            logging.info('Camera: using an image file %s', a.image)
            self.cap = 'image'
            self.img_handle = cv2.imread(a.image)
            if self.img_handle is not None:
                if self.do_resize:
                    self.img_handle = cv2.resize(self.img_handle, (a.width, a.height))
                self.is_opened = True
                self.img_height, self.img_width, _ = self.img_handle.shape
        elif a.video:
            logging.info('Camera: using a video file %s', a.video)
            self.video_file = a.video
            self.cap = cv2.VideoCapture(a.video)
            self._start()
        elif a.rtsp:
            logging.info('Camera: using RTSP stream %s', a.rtsp)
            self.cap = open_cam_rtsp(a.rtsp, a.width, a.height, a.rtsp_latency)
            self._start()
        elif a.usb is not None:
            # Try /dev/video<a.usb> first; if a.usb==0 and it fails, fallback to 1.
            primary = a.usb
            fallback = 1 if primary == 0 else None  # specific request: 0 -> 1
            logging.info('Camera: trying USB webcam /dev/video%d', primary)
            cap = _probe_usb(primary, a.width, a.height, a.fps)

            if cap is None and fallback is not None:
                logging.warning(
                    'Camera: /dev/video%d not available, trying /dev/video%d',
                    primary, fallback
                )
                cap = _probe_usb(fallback, a.width, a.height, a.fps)

            if cap is None:
                raise RuntimeError(
                    'No usable USB camera found (tried /dev/video%d%s)' %
                    (primary, f' and /dev/video{fallback}' if fallback is not None else '')
                )

            self.cap = cap
            self._start()
        elif a.gstr is not None:
            logging.info('Camera: using GStreamer string "%s"', a.gstr)
            self.cap = open_cam_gstr(a.gstr, a.width, a.height)
            self._start()
        elif a.onboard is not None:
            logging.info('Camera: using Jetson onboard camera')
            self.cap = open_cam_onboard(a.width, a.height)
            self._start()
        else:
            raise RuntimeError('no camera type specified!')

    def isOpened(self):
        return self.is_opened

    def _start(self):
        if not hasattr(self.cap, 'isOpened') or not self.cap.isOpened():
            logging.warning('Camera: starting while cap is not opened!')
            return

        # Try to grab the 1st image and determine width and height
        _, self.img_handle = self.cap.read()
        if self.img_handle is None:
            logging.warning('Camera: cap.read() returns no image!')
            self.is_opened = False
            return

        self.is_opened = True
        if self.video_file:
            if not self.do_resize:
                self.img_height, self.img_width, _ = self.img_handle.shape
        else:
            self.img_height, self.img_width, _ = self.img_handle.shape
            # start the child thread if not using a video file source
            # i.e., rtsp, usb or onboard
            assert not self.thread_running
            self.thread_running = True
            self.thread = threading.Thread(target=grab_img, args=(self,))
            self.thread.start()

    def _stop(self):
        if self.thread_running:
            self.thread_running = False
            self.thread.join()

    def read(self):
        """Read a frame from the camera object.

        Returns None if the camera runs out of image or error.
        """
        if not self.is_opened:
            return None

        if self.video_file:
            _, img = self.cap.read()
            if img is None:
                logging.info('Camera: reaching end of video file')
                if self.video_looping:
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.video_file)
                _, img = self.cap.read()
            if img is not None and self.do_resize:
                img = cv2.resize(img, (self.img_width, self.img_height))
            return img
        elif self.cap == 'image':
            return np.copy(self.img_handle)
        else:
            if self.copy_frame:
                return self.img_handle.copy()
            else:
                return self.img_handle

    def release(self):
        self._stop()
        try:
            # self.cap may be a string ('image') — guard with hasattr
            if hasattr(self.cap, 'release'):
                self.cap.release()
        except Exception:
            pass
        self.is_opened = False

    def __del__(self):
        self.release()
