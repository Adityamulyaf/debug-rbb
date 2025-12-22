# utils/bytetrack_lite.py
# ByteTrack-lite (tanpa SciPy) — kompatibel Python 3.6

from typing import List, Tuple, Dict, Set, Optional
import math

BTL_HIGH_THRESH        = 0.55   # confidence min utk stage HIGH
BTL_LOW_THRESH         = 0.10   # confidence min utk stage LOW
BTL_MATCH_THRESH       = 0.30   # IoU min utk dianggap match
BTL_TRACK_BUFFER       = 100     # berapa frame boleh hilang sebelum dibuang
BTL_CLASSES_TO_TRACK   = None   # contoh: {0,3} atau None utk semua

def set_defaults(high_thresh=None, low_thresh=None, match_thresh=None,
                 track_buffer=None, classes_to_track=None):
    """Ganti default modul (akan dipakai sebagai default pada instance baru)."""
    global BTL_HIGH_THRESH, BTL_LOW_THRESH, BTL_MATCH_THRESH, BTL_TRACK_BUFFER, BTL_CLASSES_TO_TRACK
    if high_thresh is not None:
        BTL_HIGH_THRESH = float(high_thresh)
    if low_thresh is not None:
        BTL_LOW_THRESH = float(low_thresh)
    if match_thresh is not None:
        BTL_MATCH_THRESH = float(match_thresh)
    if track_buffer is not None:
        BTL_TRACK_BUFFER = int(track_buffer)
    if classes_to_track is not None:
        # boleh None atau iterable of int
        BTL_CLASSES_TO_TRACK = set(classes_to_track) if classes_to_track is not None else None

# Det: [x1,y1,x2,y2, score, cls]
Det = List[float]

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

def greedy_match(tracks_boxes, dets_boxes, iou_thresh):
    # type: (List[Tuple[float,float,float,float]], List[Tuple[float,float,float,float]], float)
    # -> Tuple[List[Tuple[int,int]], List[int], List[int]]
    pairs = []
    for ti, tb in enumerate(tracks_boxes):
        for di, db in enumerate(dets_boxes):
            v = iou(tb, db)
            if v >= iou_thresh:
                pairs.append((v, ti, di))
    pairs.sort(reverse=True, key=lambda x: x[0])

    t_used, d_used = set(), set()
    matches = []
    for v, ti, di in pairs:
        if ti in t_used or di in d_used:
            continue
        t_used.add(ti); d_used.add(di)
        matches.append((ti, di))

    unmatched_tracks = [i for i in range(len(tracks_boxes)) if i not in t_used]
    unmatched_dets   = [i for i in range(len(dets_boxes))   if i not in d_used]
    return matches, unmatched_tracks, unmatched_dets

class _Track(object):
    __slots__ = ("tid","bbox","score","cls","age","hits","missed")
    def __init__(self, tid, bbox, score, cls):
        # type: (int, Tuple[float,float,float,float], float, int) -> None
        self.tid = tid
        self.bbox = bbox
        self.score = score
        self.cls = cls
        self.age = 1
        self.hits = 1
        self.missed = 0

    def update(self, bbox, score):
        # type: (Tuple[float,float,float,float], float) -> None
        self.bbox = bbox
        self.score = score
        self.hits += 1
        self.missed = 0
        self.age += 1

    def mark_missed(self):
        # type: () -> None
        self.missed += 1
        self.age += 1

class ByteTrackLite(object):
    """
    ByteTrack-lite (two-stage association):
      - high dets (>= high_thresh) → primary match
      - low  dets (>= low_thresh, < high_thresh) → secondary match
      - Greedy IoU matching (cukup untuk jumlah bbox moderat)

    Parameter bisa:
      • Diatur via argumen __init__ (prioritas tertinggi)
      • Pakai default modul BTL_* di atas (kalau argumen None)
      • Diubah runtime dengan set_params()
    """
    def __init__(self,
                 high_thresh=None,          # type: Optional[float]
                 low_thresh=None,           # type: Optional[float]
                 match_thresh=None,         # type: Optional[float]
                 track_buffer=None,         # type: Optional[int]
                 classes_to_track=None      # type: Optional[Set[int]]
                 ):
        # type: (...) -> None
        # fallback ke default modul bila None
        self.high_thresh   = float(BTL_HIGH_THRESH   if high_thresh   is None else high_thresh)
        self.low_thresh    = float(BTL_LOW_THRESH    if low_thresh    is None else low_thresh)
        self.match_thresh  = float(BTL_MATCH_THRESH  if match_thresh  is None else match_thresh)
        self.track_buffer  = int(  BTL_TRACK_BUFFER  if track_buffer  is None else track_buffer)
        base_classes       = BTL_CLASSES_TO_TRACK if classes_to_track is None else classes_to_track
        self.classes_to_track = set(base_classes) if base_classes is not None else None

        self._next_tid = 1
        self._tracks = {}  # type: Dict[int, _Track]

    # ==== Runtime parameter control ====
    def set_params(self,
                   high_thresh=None, low_thresh=None, match_thresh=None,
                   track_buffer=None, classes_to_track=None):
        # type: (Optional[float], Optional[float], Optional[float], Optional[int], Optional[Set[int]]) -> None
        if high_thresh is not None:
            self.high_thresh = float(high_thresh)
        if low_thresh is not None:
            self.low_thresh = float(low_thresh)
        if match_thresh is not None:
            self.match_thresh = float(match_thresh)
        if track_buffer is not None:
            self.track_buffer = int(track_buffer)
        if classes_to_track is not None:
            self.classes_to_track = set(classes_to_track) if classes_to_track is not None else None

    def get_params(self):
        # type: () -> Dict[str, object]
        return {
            "high_thresh": self.high_thresh,
            "low_thresh": self.low_thresh,
            "match_thresh": self.match_thresh,
            "track_buffer": self.track_buffer,
            "classes_to_track": set(self.classes_to_track) if self.classes_to_track is not None else None
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
        dets = self._filter_classes(dets)

        # split high/low
        high, low = [], []
        for d in dets:
            if d[4] >= self.high_thresh:
                high.append(d)
            elif d[4] >= self.low_thresh:
                low.append(d)

        tids = list(self._tracks.keys())
        tr_boxes = [self._tracks[tid].bbox for tid in tids]

        # stage 1: match HIGH
        hi_boxes = [(d[0], d[1], d[2], d[3]) for d in high]
        m1, umt, umd_hi = greedy_match(tr_boxes, hi_boxes, self.match_thresh)

        used_hi = set()
        for (ti, di) in m1:
            tid = tids[ti]
            d = high[di]; used_hi.add(di)
            self._tracks[tid].update((d[0], d[1], d[2], d[3]), float(d[4]))

        # stage 2: match LOW untuk tracks yang belum ter-match
        rem_tids = [tids[i] for i in umt]
        rem_tr_boxes = [self._tracks[tid].bbox for tid in rem_tids]
        lo_boxes = [(d[0], d[1], d[2], d[3]) for d in low]
        m2, umt2, umd_lo = greedy_match(rem_tr_boxes, lo_boxes, self.match_thresh)

        used_lo = set()
        for (ti, di) in m2:
            tid = rem_tids[ti]
            d = low[di]; used_lo.add(di)
            self._tracks[tid].update((d[0], d[1], d[2], d[3]), float(d[4]))

        # tracks yang tetap tidak match setelah 2 tahap → missed
        missed_tids = [rem_tids[i] for i in umt2]
        for tid in missed_tids:
            self._tracks[tid].mark_missed()

        # deteksi HIGH yang belum dipakai → buat track baru
        for i, d in enumerate(high):
            if i in used_hi:
                continue
            tid = self._next_tid; self._next_tid += 1
            self._tracks[tid] = _Track(tid, (d[0], d[1], d[2], d[3]), float(d[4]), int(d[5]))

        # buang tracks yang missed > buffer
        to_del = [tid for tid, tr in self._tracks.items() if tr.missed > self.track_buffer]
        for tid in to_del:
            del self._tracks[tid]

        # keluarkan hanya tracks yang "terlihat" di frame ini (missed == 0)
        outs = []
        for tid, tr in self._tracks.items():
            if tr.missed == 0:
                x1, y1, x2, y2 = tr.bbox
                outs.append((tid, x1, y1, x2, y2, tr.cls, tr.score))
        return outs
