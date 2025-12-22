# utils/ocsort.py

from typing import List, Tuple, Dict, Set, Optional
import math
import numpy as np

BTL_HIGH_THRESH        = 0.55   # minimum confidence utk membuat track baru
BTL_LOW_THRESH         = 0.10   # tidak dipakai di OC-SORT lite ini (disimpan demi kompatibilitas)
BTL_MATCH_THRESH       = 0.30   # IoU minimum untuk asosiasi (observation-centric IoU)
BTL_TRACK_BUFFER       = 100    # maksimum frame boleh "hilang" (max_age) sebelum dibuang
BTL_CLASSES_TO_TRACK   = None   # contoh: {0,3} atau None utk semua

OCS_DIR_COS_THRESH     = -1.0   # ambang cosine arah; jika < nilai ini → penalti. -1.0 artinya nonaktif.
OCS_DIR_PENALTY        = 0.20   # penurunan skor match saat gagal gating arah (0.2 ⇒ skor *= 0.8)
OCS_MIN_HITS           = 0      # bisa dipakai untuk hanya output track yg sudah cukup stabil (0=nonaktif)

def set_defaults(high_thresh=None, low_thresh=None, match_thresh=None,
                 track_buffer=None, classes_to_track=None,
                 dir_cos_thresh=None, dir_penalty=None, min_hits=None):
    global BTL_HIGH_THRESH, BTL_LOW_THRESH, BTL_MATCH_THRESH
    global BTL_TRACK_BUFFER, BTL_CLASSES_TO_TRACK
    global OCS_DIR_COS_THRESH, OCS_DIR_PENALTY, OCS_MIN_HITS

    if high_thresh is not None:
        BTL_HIGH_THRESH = float(high_thresh)
    if low_thresh is not None:
        BTL_LOW_THRESH = float(low_thresh)
    if match_thresh is not None:
        BTL_MATCH_THRESH = float(match_thresh)
    if track_buffer is not None:
        BTL_TRACK_BUFFER = int(track_buffer)
    if classes_to_track is not None:
        BTL_CLASSES_TO_TRACK = set(classes_to_track) if classes_to_track is not None else None
    if dir_cos_thresh is not None:
        OCS_DIR_COS_THRESH = float(dir_cos_thresh)
    if dir_penalty is not None:
        OCS_DIR_PENALTY = float(dir_penalty)
    if min_hits is not None:
        OCS_MIN_HITS = int(min_hits)

# Det: [x1,y1,x2,y2, score, cls]
Det = List[float]

# Util bbox & IoU
def iou(a, b):
    # type: (Tuple[float,float,float,float], Tuple[float,float,float,float]) -> float
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = a_area + b_area - inter
    return inter / denom if denom > 0 else 0.0

def xyxy_to_xyah(b):
    # [x1,y1,x2,y2] -> [x,y,a,h]
    x1, y1, x2, y2 = b
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w * 0.5
    cy = y1 + h * 0.5
    a = (w / h) if h > 1e-6 else 0.0
    return np.array([cx, cy, a, h], dtype=np.float32)

def xyah_to_xyxy(s):
    # [x,y,a,h] -> [x1,y1,x2,y2]
    x, y, a, h = float(s[0]), float(s[1]), float(s[2]), float(s[3])
    w = a * h
    x1 = x - w * 0.5
    y1 = y - h * 0.5
    x2 = x + w * 0.5
    y2 = y + h * 0.5
    return (x1, y1, x2, y2)

def center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

# Simple Kalman Filter (NumPy only)
class KalmanFilter(object):
    def __init__(self):
        ndim, dt = 4, 1.
        # State: [x, y, a, h, vx, vy, va, vh]
        self._motion_mat = np.eye(2*ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2*ndim, dtype=np.float32)

        # Std (tuning sederhana)
        self._std_pos = 1.0
        self._std_vel = 0.5
        self._std_meas = 1.0

    def initiate(self, measurement):
        # measurement: [x,y,a,h]
        mean = np.zeros((8,), dtype=np.float32)
        mean[:4] = measurement
        P = np.eye(8, dtype=np.float32)
        P[:4, :4] *= (self._std_pos ** 2) * 10.0
        P[4:, 4:] *= (self._std_vel ** 2) * 100.0
        return mean, P

    def predict(self, mean, cov):
        F = self._motion_mat
        Q = np.eye(8, dtype=np.float32) * (self._std_vel ** 2)
        mean = F.dot(mean)
        cov = F.dot(cov).dot(F.T) + Q
        return mean, cov

    def project(self, mean, cov):
        H = self._update_mat
        R = np.eye(4, dtype=np.float32) * (self._std_meas ** 2)
        m = H.dot(mean)
        S = H.dot(cov).dot(H.T) + R
        return m, S

    def update(self, mean, cov, measurement):
        # Kalman update
        m, S = self.project(mean, cov)
        K = cov.dot(self._update_mat.T).dot(np.linalg.inv(S))
        innovation = measurement - m
        mean = mean + K.dot(innovation)
        I = np.eye(8, dtype=np.float32)
        cov = (I - K.dot(self._update_mat)).dot(cov)
        return mean, cov

# Track
class _OCTrack(object):
    __slots__ = ("tid","cls","score","age","hits","missed",
                 "kf_mean","kf_cov","last_obs_bbox","prev_obs_center")
    def __init__(self, tid, bbox_xyxy, score, cls, kf):
        self.tid = tid
        self.cls = int(cls)
        self.score = float(score)
        self.age = 1
        self.hits = 1
        self.missed = 0
        self.last_obs_bbox = (float(bbox_xyxy[0]), float(bbox_xyxy[1]),
                              float(bbox_xyxy[2]), float(bbox_xyxy[3]))
        self.prev_obs_center = None  # untuk vektor arah

        z = xyxy_to_xyah(self.last_obs_bbox)
        mean, cov = kf.initiate(z)
        self.kf_mean, self.kf_cov = mean, cov

    def predict(self, kf):
        self.kf_mean, self.kf_cov = kf.predict(self.kf_mean, self.kf_cov)
        self.age += 1
        # pred bbox (tidak dipakai dalam asosiasi OC, tapi berguna untuk debug/visual)
        pred_xyah = self.kf_mean[:4]
        # return pred bbox jika mau dipakai
        return xyah_to_xyxy(pred_xyah)

    def update_with_detection(self, kf, det_bbox_xyxy, det_score):
        # update Kalman dengan observasi baru
        z = xyxy_to_xyah(det_bbox_xyxy)
        self.kf_mean, self.kf_cov = kf.update(self.kf_mean, self.kf_cov, z)
        
        old_center = center(self.last_obs_bbox)
        self.prev_obs_center = old_center

		# lalu update ke observasi baru
        self.last_obs_bbox = (float(det_bbox_xyxy[0]), float(det_bbox_xyxy[1]),float(det_bbox_xyxy[2]), float(det_bbox_xyxy[3]))
        self.score = float(det_score)
        self.hits += 1
        self.missed = 0

        # update state track
        self.last_obs_bbox = (float(det_bbox_xyxy[0]), float(det_bbox_xyxy[1]),
                              float(det_bbox_xyxy[2]), float(det_bbox_xyxy[3]))
        self.score = float(det_score)
        self.hits += 1
        self.missed = 0

    def mark_missed(self):
        self.missed += 1

# =========================
# Matching (Observation-Centric + optional direction gating)
# =========================
def _cosine(a, b):
    ax, ay = a
    bx, by = b
    da = math.hypot(ax, ay)
    db = math.hypot(bx, by)
    if da < 1e-6 or db < 1e-6:
        return 1.0  # kalau tidak ada arah yg jelas, anggap cocok
    return (ax*bx + ay*by) / (da * db)

def _ocsort_match(tracks, dets_xyxy, iou_thresh, dir_cos_thresh, dir_penalty):
    """
    Greedy matching berdasarkan Observation-Centric IoU.
    Jika gagal gating arah (cos < dir_cos_thresh), skor dipenalti.
    """
    pairs = []  # (score, ti, di)
    for ti, tr in enumerate(tracks):
        tbox = tr.last_obs_bbox
        tcenter_prev = tr.prev_obs_center
        for di, dbox in enumerate(dets_xyxy):
            score = iou(tbox, dbox)  # OC IoU (observasi terakhir)
            if score < iou_thresh:
                continue
            # direction gating penalti (opsional)
            if tcenter_prev is not None and dir_cos_thresh > -1.0:
                cur_c = center(dbox)
                last_c = center(tbox)
                v_track = (last_c[0] - tcenter_prev[0], last_c[1] - tcenter_prev[1])
                v_det   = (cur_c[0]  - last_c[0],      cur_c[1]  - last_c[1])
                cosv = _cosine(v_track, v_det)
                if cosv < dir_cos_thresh:
                    score *= (1.0 - dir_penalty)
            pairs.append((score, ti, di))
    pairs.sort(reverse=True, key=lambda x: x[0])

    used_t, used_d = set(), set()
    matches = []
    for sc, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti); used_d.add(di)
        matches.append((ti, di))

    unmatched_tracks = [i for i in range(len(tracks)) if i not in used_t]
    unmatched_dets   = [i for i in range(len(dets_xyxy)) if i not in used_d]
    return matches, unmatched_tracks, unmatched_dets

class OCSortLite(object):
    """
    OC-SORT (lite):
      - Kalman filter (NumPy, no SciPy)
      - Observation-centric IoU untuk asosiasi
      - Opsi gating arah (cosine) dengan penalti
      - Greedy matching (cukup utk jumlah bbox moderat)

    Parameter bisa:
      • Diatur via argumen __init__ (prioritas tertinggi)
      • Pakai default modul BTL_* / OCS_* di atas (kalau argumen None)
      • Diubah runtime dengan set_params()
    """
    def __init__(self,
                 high_thresh=None,          # type: Optional[float]
                 low_thresh=None,           # type: Optional[float] (ignored)
                 match_thresh=None,         # type: Optional[float]
                 track_buffer=None,         # type: Optional[int]
                 classes_to_track=None,     # type: Optional[Set[int]]
                 dir_cos_thresh=None,       # type: Optional[float]
                 dir_penalty=None,          # type: Optional[float]
                 min_hits=None              # type: Optional[int]
                 ):
        # threshold & buffer
        self.high_thresh   = float(BTL_HIGH_THRESH   if high_thresh   is None else high_thresh)
        self.low_thresh    = float(BTL_LOW_THRESH    if low_thresh    is None else low_thresh)  # not used
        self.match_thresh  = float(BTL_MATCH_THRESH  if match_thresh  is None else match_thresh)
        self.max_age       = int(  BTL_TRACK_BUFFER  if track_buffer  is None else track_buffer)
        base_classes       = BTL_CLASSES_TO_TRACK if classes_to_track is None else classes_to_track
        self.classes_to_track = set(base_classes) if base_classes is not None else None

        # OC-SORT specific
        self.dir_cos_thresh = float(OCS_DIR_COS_THRESH if dir_cos_thresh is None else dir_cos_thresh)
        self.dir_penalty    = float(OCS_DIR_PENALTY    if dir_penalty    is None else dir_penalty)
        self.min_hits       = int(  OCS_MIN_HITS       if min_hits       is None else min_hits)

        # state
        self._next_tid = 1
        self._tracks = {}  # type: Dict[int, _OCTrack]
        self._kf = KalmanFilter()

    # ==== Runtime parameter control ====
    def set_params(self,
                   high_thresh=None, low_thresh=None, match_thresh=None,
                   track_buffer=None, classes_to_track=None,
                   dir_cos_thresh=None, dir_penalty=None, min_hits=None):
        if high_thresh is not None:
            self.high_thresh = float(high_thresh)
        if low_thresh is not None:
            self.low_thresh = float(low_thresh)  # not used
        if match_thresh is not None:
            self.match_thresh = float(match_thresh)
        if track_buffer is not None:
            self.max_age = int(track_buffer)
        if classes_to_track is not None:
            self.classes_to_track = set(classes_to_track) if classes_to_track is not None else None
        if dir_cos_thresh is not None:
            self.dir_cos_thresh = float(dir_cos_thresh)
        if dir_penalty is not None:
            self.dir_penalty = float(dir_penalty)
        if min_hits is not None:
            self.min_hits = int(min_hits)

    def get_params(self):
        return {
            "high_thresh": self.high_thresh,
            "low_thresh": self.low_thresh,
            "match_thresh": self.match_thresh,
            "track_buffer(max_age)": self.max_age,
            "classes_to_track": set(self.classes_to_track) if self.classes_to_track is not None else None,
            "dir_cos_thresh": self.dir_cos_thresh,
            "dir_penalty": self.dir_penalty,
            "min_hits": self.min_hits,
        }

    # ==== Core ====
    def _filter_classes(self, dets):
        # type: (List[Det]) -> List[Det]
        if self.classes_to_track is None:
            return dets
        out = []
        for d in dets:
            cls_id = int(d[5])
            if cls_id in self.classes_to_track:
                out.append(d)
        return out

    def update(self, dets, img_size=None):
        # type: (List[Det], Tuple[int,int]) -> List[Tuple[int,float,float,float,float,int,float]]
        # 1) filter kelas
        dets = self._filter_classes(dets)

        # 2) pisah bbox & meta
        det_boxes = [(d[0], d[1], d[2], d[3]) for d in dets]
        det_scores = [float(d[4]) for d in dets]
        det_clses = [int(d[5]) for d in dets]

        # 3) predict semua track
        tids = list(self._tracks.keys())
        tracks_list = [self._tracks[tid] for tid in tids]
        for tr in tracks_list:
            tr.predict(self._kf)

        # 4) match OC-IoU + (opsional) penalti arah
        m, umt_idx, umd_idx = _ocsort_match(
            tracks_list, det_boxes, self.match_thresh, self.dir_cos_thresh, self.dir_penalty
        )

        used_det = set()
        # 5) update matched
        for (ti, di) in m:
            tr = tracks_list[ti]
            box = det_boxes[di]
            tr.update_with_detection(self._kf, box, det_scores[di])
            used_det.add(di)

        # 6) unmatched tracks -> missed
        for i in umt_idx:
            tracks_list[i].mark_missed()

        # 7) unmatched detections -> buat track baru jika score >= high_thresh
        for di in umd_idx:
            if det_scores[di] >= self.high_thresh:
                tid = self._next_tid; self._next_tid += 1
                tr = _OCTrack(tid, det_boxes[di], det_scores[di], det_clses[di], self._kf)
                self._tracks[tid] = tr

        # 8) buang tracks yg "terlalu lama hilang"
        to_del = [tid for tid, tr in self._tracks.items() if tr.missed > self.max_age]
        for tid in to_del:
            del self._tracks[tid]

        # 9) output hanya tracks yg terlihat (missed==0) & optional min_hits
        outs = []
        for tid, tr in self._tracks.items():
            if tr.missed == 0 and tr.hits >= self.min_hits:
                x1, y1, x2, y2 = tr.last_obs_bbox
                outs.append((tid, x1, y1, x2, y2, tr.cls, tr.score))
        return outs

