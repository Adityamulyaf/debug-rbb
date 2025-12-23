# avoid.py
#!/usr/bin/env python3

import os
import time
import argparse
import cv2
import numpy as np
import rospy

from datetime import datetime
from utils.display import open_window, set_display, show_fps, draw_hud
from utils.model_registry import get_cls_dict_by_model, get_color_map_by_model
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from utils.utils_avoid import UTILS_MISI_AVOID


WINDOW_NAME = 'AVOID'

# Mission parameters
SPEED_MISI = 2.0
MAX_ANGULAR_VEL = 1.0  # Used in END phase

# Waypoint ranges
START_RANGE = 2
END_RANGE = 3
BACK_RANGE = 2

def get_roi_params(H, W):
    """Calculate ROI parameters based on frame dimensions"""
    Y_ROI = 240
    KONSTANTA_BOLA = 100
    return Y_ROI, KONSTANTA_BOLA

# Trapezoid ROI parameters
TRAP_TOP_HALF_WIDTH = 80 # Half-width at top (y = H/2)
TRAP_BOTTOM_HALF_RATIO = 0.5  # Bottom half-width as ratio of W


# ARGUMENT

def register(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("Mission Avoid")

    g.add_argument('--avoid-model', type=str, default='avoid_640_352_tiny_2')
    g.add_argument('--avoid-category-num', type=int, default=5)
    g.add_argument('--avoid-conf-thresh', type=float, default=0.7)
    g.add_argument('--avoid-engine-dir', type=str, default='yolo')
    g.add_argument('--avoid-letter-box', action='store_true')
    g.add_argument('--avoid-keep-window', action='store_true')

    g.add_argument('--avoid-tracker', type=str, default='bytetrack', choices=['none','bytetrack','ocsort'])
    g.add_argument('--avoid-track-classes', type=str, default='all',
                   help='kelas yang akan di-track (id dipisah koma) atau "all"')
    g.add_argument('--avoid-track-high', type=float, default=0.35)
    g.add_argument('--avoid-track-low', type=float, default=0.05)
    g.add_argument('--avoid-track-iou', type=float, default=0.20)
    g.add_argument('--avoid-track-buffer', type=int, default=100)

    g.add_argument('--avoid-end-wp', type=int, default=3,
                   help='cur_wp >= nilai ini berubah ke fase END. -1 = nonaktif')
    g.add_argument('--avoid-allow-start', type=str, default='greenball,redball,blackball',
                   help='kelas diizinkan pada fase START (nama, koma, atau "all")')
    g.add_argument('--avoid-allow-end', type=str, default='blackball,greenlight,redlight',
                   help='kelas diizinkan pada fase END (nama, koma, atau "all")')
    g.add_argument('--avoid-back-mode', action='store_true',
                   help='aktifkan urutan kebalikan: END ke START (mode BACK)')
        
    # threshold per kelas: "greenlight:0.25,redlight:0.25"
    g.add_argument('--avoid-class-thresh', type=str, default='greenball:0.4',
                    help='override conf thresh per kelas, format "name:0.3,name:0.2"')


# HELPER FUNCTIONS
def _parse_track_classes(s: str, num_classes: int):
    if not s or s.strip().lower() in ('all', '*'):
        return None
    ids = set()
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        cid = int(tok)
        if cid < 0 or cid >= num_classes:
            raise SystemExit(f'ERROR: class id {cid} out of range [0..{num_classes-1}]')
        ids.add(cid)
    return ids

def _attach_tracker(name, classes_to_track, args):
    if name == 'bytetrack':
        from utils.bytetrack_lite import ByteTrackLite
        return ByteTrackLite(
            high_thresh=args.avoid_track_high,
            low_thresh=args.avoid_track_low,
            match_thresh=args.avoid_track_iou,
            track_buffer=args.avoid_track_buffer,
            classes_to_track=classes_to_track
        )
    if name == 'ocsort':
        from utils.ocsort import OCSortLite
        return OCSortLite(
            high_thresh=0.60,
            match_thresh=0.30,
            track_buffer=100,
            classes_to_track=classes_to_track,
            dir_cos_thresh=-1.0,
            dir_penalty=0.20,
            min_hits=0
        )
    return None

def _names_to_ids(list_str, cls_dict):
    if not list_str or list_str.strip().lower() in ('all', '*'):
        return None
    inv = {name.lower(): cid for cid, name in cls_dict.items()}
    out = set()
    for raw in list_str.split(','):
        name = raw.strip().lower()
        if not name:
            continue
        if name not in inv:
            raise SystemExit(f'ERROR: unknown class name \"{name}\". Known: {list(inv.keys())}')
        out.add(inv[name])
    return out

def _parse_class_thresh(s, cls_dict):
    if not s:
        return {}
    inv = {name.lower(): cid for cid, name in cls_dict.items()}
    out = {}
    for pair in s.split(','):
        pair = pair.strip()
        if not pair:
            continue
        if ':' not in pair:
            raise SystemExit('ERROR: invalid --avoid-class-thresh format.')
        nm, th = pair.split(':', 1)
        nm = nm.strip().lower()
        try:
            thf = float(th.strip())
        except Exception:
            raise SystemExit('ERROR: invalid threshold value in --avoid-class-thresh')
        if nm not in inv:
            raise SystemExit(f'ERROR: unknown class name \"{nm}\" in --avoid-class-thresh')
        out[inv[nm]] = thf
    return out

def _center_of(box):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    y_bar = int((y1 + y2) / 2)
    cy = int(y_bar + (y2 - y_bar) / 2)
    return cx, cy

def _point_in_trapezoid(px, py, W, H, top_half_width=40, bottom_half_width=None):
    """
    Check if point (px, py) is inside trapezoid ROI.
    
    Applied trapezoid ROI filter to gb/rb/bb_terdekat
    """
    if bottom_half_width is None:
        bottom_half_width = W // 2
    
    y_top = H // 2
    y_bottom = H
    x_center = W // 2
    
    # Check y range
    if not (y_top <= py <= y_bottom):
        return False
    
    # Interpolate width based on y position
    progress = (py - y_top) / (y_bottom - y_top)
    half_width_at_y = top_half_width + progress * (bottom_half_width - top_half_width)
    
    x_left = x_center - half_width_at_y
    x_right = x_center + half_width_at_y
    
    return x_left <= px <= x_right

def _draw_trapezoid_roi(img, W, H, top_half_width=40, bottom_half_width=None, color=(0, 255, 0), thickness=2):
    """Draw trapezoid ROI on frame"""
    if bottom_half_width is None:
        bottom_half_width = W // 2
    
    y_top = 220
    y_bottom = H
    x_center = W // 2
    
    # 4 corner points (clockwise from top-left)
    pts = np.array([
        [x_center - top_half_width, y_top],
        [x_center + top_half_width, y_top],
        [x_center + bottom_half_width, y_bottom],
        [x_center - bottom_half_width, y_bottom]
    ], dtype=np.int32)
    
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    return img

def _filter_by_allowed(boxes, confs, clss, conf_th, per_cls_thresh, allowed_ids):
    """Filter detections based on allowed classes and thresholds"""
    if allowed_ids is None and not per_cls_thresh:
        return boxes, confs, clss

    fb, fc, fk = [], [], []
    for (b, sc, c) in zip(boxes, confs, clss):
        c = int(c)
        if (allowed_ids is not None) and (c not in allowed_ids):
            continue
        thr = conf_th
        if per_cls_thresh:
            thr = max(conf_th, float(per_cls_thresh.get(c, 0.0)))
        if sc < thr:
            continue
        fb.append(b)
        fc.append(sc)
        fk.append(c)
    return fb, fc, fk

def _clamp(value, min_val, max_val):
    """Clamp value to range [min_val, max_val]"""
    return max(min_val, min(value, max_val))


# MISSION PHASE
def _avoid_start_step(img, manda, W, H, boxes, confs, clss, name2id, utils_avoid, GAIN_YAW, Y_ROI):
    """
    START phase navigation
    
    Applied trapezoid ROI filter to all ball detections
    """
    vel_forward = SPEED_MISI
    
    id_gb = name2id.get("greenball")
    id_rb = name2id.get("redball")
    id_bb = name2id.get("blackball")
    list_gb, list_rb, list_bb = [], [], []

    for (box, _, cls) in zip(boxes, confs, clss):
        if cls == id_gb:
            list_gb.append(box)
        elif cls == id_rb:
            list_rb.append(box)
        elif cls == id_bb:
            list_bb.append(box)

    gb_terdekat = utils_avoid.pilih_bola_terdekat(list_gb)
    rb_terdekat = utils_avoid.pilih_bola_terdekat(list_rb)
    bb_terdekat = utils_avoid.pilih_bola_terdekat(list_bb)

    # Convert to bbox center first
    gb_terdekat = utils_avoid.cari_titik_tengah_bbox(gb_terdekat)
    rb_terdekat = utils_avoid.cari_titik_tengah_bbox(rb_terdekat)
    bb_terdekat = utils_avoid.cari_titik_tengah_bbox(bb_terdekat)
    
    # Apply trapezoid ROI filter
    trap_top = TRAP_TOP_HALF_WIDTH
    trap_bottom = int(TRAP_BOTTOM_HALF_RATIO * W)
    
    if gb_terdekat is not None:
        if not _point_in_trapezoid(gb_terdekat[0], gb_terdekat[1], W, H, trap_top, trap_bottom):
            gb_terdekat = None
    
    if rb_terdekat is not None:
        if not _point_in_trapezoid(rb_terdekat[0], rb_terdekat[1], W, H, trap_top, trap_bottom):
            rb_terdekat = None
    
    if bb_terdekat is not None:
        if not _point_in_trapezoid(bb_terdekat[0], bb_terdekat[1], W, H, trap_top, trap_bottom):
            bb_terdekat = None
    
    # Filter based on Y_ROI
    gb_terdekat = gb_terdekat if (gb_terdekat is not None and gb_terdekat[1] >= Y_ROI) else None
    rb_terdekat = rb_terdekat if (rb_terdekat is not None and rb_terdekat[1] >= Y_ROI) else None
    bb_terdekat = bb_terdekat if (bb_terdekat is not None and bb_terdekat[1] >= Y_ROI) else None
    
    titik_tengah = None  # Initialize
    
    # Case 1: 3 balls detected
    if gb_terdekat is not None and rb_terdekat is not None and bb_terdekat is not None:
        titik_tengah = utils_avoid.titik_tengah_3_bola(gb_terdekat, rb_terdekat, bb_terdekat)
        vel_belok = utils_avoid.cari_kecepatan_sudut(GAIN_YAW, titik_tengah, W)
        manda.set_velocity(vel_forward, vel_belok)
    
    # Case 2: GREEN + RED only
    elif gb_terdekat is not None and rb_terdekat is not None and bb_terdekat is None:
        titik_tengah = utils_avoid.titik_tengah_2_bola(gb_terdekat, rb_terdekat)
        vel_belok = utils_avoid.cari_kecepatan_sudut(GAIN_YAW, titik_tengah, W)
        manda.set_velocity(vel_forward, vel_belok)
    
    # Case 3: GREEN + BLACK
    elif gb_terdekat is not None and bb_terdekat is not None and rb_terdekat is None:
        titik_gb = utils_avoid.titik_tengah_1_bola(gb_terdekat, W, H, Y_ROI, reverse=True)
        if titik_gb is not None:
            titik_gb_x = _clamp(titik_gb[0], 0, W)
            target_x = (gb_terdekat[0] - (gb_terdekat[0] - bb_terdekat[0]) / 2) if titik_gb_x <= bb_terdekat[0] <= gb_terdekat[0] else (gb_terdekat[0] + bb_terdekat[0]) / 2
            titik_tengah = (int(_clamp(target_x, 0, W)), gb_terdekat[1])
            vel_belok = utils_avoid.cari_kecepatan_sudut(GAIN_YAW, titik_tengah, W)
            manda.set_velocity(vel_forward, vel_belok)
    
    # Case 4: RED + BLACK
    elif rb_terdekat is not None and bb_terdekat is not None and gb_terdekat is None:
        titik_rb = utils_avoid.titik_tengah_1_bola(rb_terdekat, W, H, Y_ROI, reverse=False)
        if titik_rb is not None:
            titik_rb_x = _clamp(titik_rb[0], 0, W)
            target_x = (rb_terdekat[0] + (bb_terdekat[0] - rb_terdekat[0]) / 2) if rb_terdekat[0] <= bb_terdekat[0] <= titik_rb_x else (rb_terdekat[0] + bb_terdekat[0]) / 2
            titik_tengah = (int(_clamp(target_x, 0, W)), rb_terdekat[1])
            vel_belok = utils_avoid.cari_kecepatan_sudut(GAIN_YAW, titik_tengah, W)
            manda.set_velocity(vel_forward, vel_belok)
    
    # Case 5: Only GREEN
    elif gb_terdekat is not None and rb_terdekat is None and bb_terdekat is None:
        titik_tengah = utils_avoid.titik_tengah_1_bola(gb_terdekat, W, H, Y_ROI, reverse=True)
        if titik_tengah is not None:
            vel_belok = utils_avoid.cari_kecepatan_sudut(GAIN_YAW, titik_tengah, W)
            manda.set_velocity(vel_forward, vel_belok)
    
    # Case 6: Only RED
    elif rb_terdekat is not None and gb_terdekat is None and bb_terdekat is None:
        titik_tengah = utils_avoid.titik_tengah_1_bola(rb_terdekat, W, H, Y_ROI, reverse=False)
        if titik_tengah is not None:
            vel_belok = utils_avoid.cari_kecepatan_sudut(GAIN_YAW, titik_tengah, W)
            manda.set_velocity(vel_forward, vel_belok)
    
    # Case 7: Only BLACK or no balls
    else:
        # No balls visible, move to WP if available
        if (getattr(manda, 'target_lat', None) is not None) and (getattr(manda, 'target_lon', None) is not None):
            manda.move(manda.target_lat, manda.target_lon)
    
    # Draw target point if exists
    if titik_tengah is not None:
        titik_tengah_int = (int(titik_tengah[0]), int(titik_tengah[1]))
        cv2.circle(img, titik_tengah_int, 5, (255, 0, 255), -1)

    steering = None
    return steering, vel_forward, "START"

def _avoid_end_step(manda, W, H, boxes, confs, clss, name2id, utils_avoid, progress, idx, lat, lon, C=0):
    """
    END phase navigation
    
    Add END_RANGE += len(list_wp_muter) AFTER waypoint insertion
    """
    global END_RANGE  # Need to modify global END_RANGE
    vel_forward = SPEED_MISI

    id_gl = name2id.get("greenlight")
    id_rl = name2id.get("redlight")
    id_bb = name2id.get("blackball")

    yaw = getattr(manda, 'current_yaw', None)
    p_x = getattr(manda, 'p_x', None)
    p_y = getattr(manda, 'p_y', None)
    
    # Trapezoid ROI parameters
    trap_top_half = TRAP_TOP_HALF_WIDTH
    trap_bottom_half = int(TRAP_BOTTOM_HALF_RATIO * W)

    gl_terdekat, rl_terdekat, bb_terdekat = None, None, None
    list_bb = []

    for (box, conf, cls) in zip(boxes, confs, clss):
        if cls == id_gl and conf > 0.8:
            gl_terdekat = box
        elif cls == id_rl and conf > 0.8:
            rl_terdekat = box
        elif cls == id_bb:
            list_bb.append(box)

    bb_terdekat = utils_avoid.pilih_bola_terdekat(list_bb)
    gl_terdekat = utils_avoid.cari_titik_tengah_bbox(gl_terdekat)
    rl_terdekat = utils_avoid.cari_titik_tengah_bbox(rl_terdekat)
    bb_terdekat = utils_avoid.cari_titik_tengah_bbox(bb_terdekat)

    steering = None

    if (gl_terdekat is not None) and (progress is not None) and (not progress.get("gl_done", False)):
        error_x_normalized = (gl_terdekat[0] - (W // 2)) / (W // 2)
        vel_belok = -MAX_ANGULAR_VEL * error_x_normalized
        manda.set_velocity(vel_forward, vel_belok)
        steering = vel_belok
        time.sleep(1)
        
        # Check if greenlight is inside trapezoid ROI
        if _point_in_trapezoid(gl_terdekat[0], gl_terdekat[1], W, H, trap_top_half, trap_bottom_half):
            pos_lampu = utils_avoid.hitung_posisi_lampu((p_x, p_y, yaw), JARAK_LAMPU=5)
            rospy.loginfo_throttle(5.0, f"Green light detected, lamp position = {pos_lampu}")
            
            list_wp_muter = utils_avoid.buat_wp_mengitari_lampu(
                pos_lampu, JUMLAH_TITIK=3,
                origin=(manda.home_lat, manda.home_lon),
                RADIUS=2
            )
            
            rospy.loginfo("Inserting green light circle waypoints...")
            
            # Increment AFTER logging, not before
            for i, (wp_lat, wp_lon) in enumerate(list_wp_muter):
                lat.insert(idx + i, wp_lat)
                lon.insert(idx + i, wp_lon)
            
            # Update END_RANGE AFTER dynamic waypoint insertion
            END_RANGE += len(list_wp_muter)
            
            rospy.loginfo(f"Added {len(list_wp_muter)} waypoints, END_RANGE now = {END_RANGE}")
            time.sleep(3)
            
            progress["gl_done"] = True

    elif (rl_terdekat is not None) and (progress is not None) and (not progress.get("rl_done", False)):
        error_x_normalized = (rl_terdekat[0] - (W // 2)) / (W // 2)
        vel_belok = -MAX_ANGULAR_VEL * error_x_normalized
        manda.set_velocity(vel_forward, vel_belok)
        steering = vel_belok

        # Check if redlight is inside trapezoid ROI
        if _point_in_trapezoid(rl_terdekat[0], rl_terdekat[1], W, H, trap_top_half, trap_bottom_half):
            utils_avoid.posisi_lampu_merah(
                (p_x, p_y, yaw),
                JARAK_LAMPU=5,
                origin=(manda.home_lat, manda.home_lon)
            )
            
            rospy.loginfo("Red light condition triggered")
            progress["rl_done"] = True

    elif bb_terdekat is not None:
        vel_belok = utils_avoid.titik_tuju(bb_terdekat, W, C)
        manda.set_velocity(vel_forward, vel_belok)
        steering = vel_belok
    else:
        if (getattr(manda, 'target_lat', None) is not None) and (getattr(manda, 'target_lon', None) is not None):
            manda.move(lat[idx], lon[idx])
        steering = 0

    return steering, vel_forward, "END", lat, lon, idx

def _avoid_back_step(img, manda, W, H, boxes, confs, clss, name2id, utils_avoid, progress, GAIN_YAW, Y_ROI):
    """
    BACK phase navigation
    """
    vel_forward = SPEED_MISI
    
    id_gb = name2id.get("greenball")
    id_rb = name2id.get("redball")
    id_bb = name2id.get("blackball")
    list_gb, list_rb, list_bb = [], [], []

    for (box, _, cls) in zip(boxes, confs, clss):
        if cls == id_gb:
            list_gb.append(box)
        elif cls == id_rb:
            list_rb.append(box)
        elif cls == id_bb:
            list_bb.append(box)

    gb_terdekat = utils_avoid.pilih_bola_terdekat(list_gb)
    rb_terdekat = utils_avoid.pilih_bola_terdekat(list_rb)
    bb_terdekat = utils_avoid.pilih_bola_terdekat(list_bb)

    # Convert to bbox center
    gb_terdekat = utils_avoid.cari_titik_tengah_bbox(gb_terdekat)
    rb_terdekat = utils_avoid.cari_titik_tengah_bbox(rb_terdekat)
    bb_terdekat = utils_avoid.cari_titik_tengah_bbox(bb_terdekat)
    
    # Filter based on Y_ROI
    gb_terdekat = gb_terdekat if (gb_terdekat is not None and gb_terdekat[1] >= Y_ROI) else None
    rb_terdekat = rb_terdekat if (rb_terdekat is not None and rb_terdekat[1] >= Y_ROI) else None
    bb_terdekat = bb_terdekat if (bb_terdekat is not None and bb_terdekat[1] >= Y_ROI) else None
    
    titik_tengah = None
    
    # Same logic as START but with reverse directions
    if gb_terdekat is not None and rb_terdekat is not None and bb_terdekat is not None:
        titik_tengah = utils_avoid.titik_tengah_3_bola(gb_terdekat, rb_terdekat, bb_terdekat)
        vel_belok = utils_avoid.cari_kecepatan_sudut(GAIN_YAW, titik_tengah, W)
        manda.set_velocity(vel_forward, vel_belok)
    
    elif gb_terdekat is not None and rb_terdekat is not None and bb_terdekat is None:
        titik_tengah = utils_avoid.titik_tengah_2_bola(gb_terdekat, rb_terdekat)
        vel_belok = utils_avoid.cari_kecepatan_sudut(GAIN_YAW, titik_tengah, W)
        manda.set_velocity(vel_forward, vel_belok)
    
    elif gb_terdekat is not None and bb_terdekat is not None and rb_terdekat is None:
        titik_gb = utils_avoid.titik_tengah_1_bola(gb_terdekat, W, H, Y_ROI, reverse=False)
        if titik_gb is not None:
            titik_gb_x = _clamp(titik_gb[0], 0, W)
            target_x = (gb_terdekat[0] - (gb_terdekat[0] - bb_terdekat[0]) / 2) if titik_gb_x <= bb_terdekat[0] <= gb_terdekat[0] else (gb_terdekat[0] + bb_terdekat[0]) / 2
            titik_tengah = (int(_clamp(target_x, 0, W)), gb_terdekat[1])
            vel_belok = utils_avoid.cari_kecepatan_sudut(GAIN_YAW, titik_tengah, W)
            manda.set_velocity(vel_forward, vel_belok)
    
    elif rb_terdekat is not None and bb_terdekat is not None and gb_terdekat is None:
        titik_rb = utils_avoid.titik_tengah_1_bola(rb_terdekat, W, H, Y_ROI, reverse=True)
        if titik_rb is not None:
            titik_rb_x = _clamp(titik_rb[0], 0, W)
            target_x = (rb_terdekat[0] + (bb_terdekat[0] - rb_terdekat[0]) / 2) if rb_terdekat[0] <= bb_terdekat[0] <= titik_rb_x else (rb_terdekat[0] + bb_terdekat[0]) / 2
            titik_tengah = (int(_clamp(target_x, 0, W)), rb_terdekat[1])
            vel_belok = utils_avoid.cari_kecepatan_sudut(GAIN_YAW, titik_tengah, W)
            manda.set_velocity(vel_forward, vel_belok)
    
    elif gb_terdekat is not None and rb_terdekat is None and bb_terdekat is None:
        titik_tengah = utils_avoid.titik_tengah_1_bola(gb_terdekat, W, H, Y_ROI, reverse=False)
        if titik_tengah is not None:
            vel_belok = utils_avoid.cari_kecepatan_sudut(GAIN_YAW, titik_tengah, W)
            manda.set_velocity(vel_forward, vel_belok)
    
    elif rb_terdekat is not None and gb_terdekat is None and bb_terdekat is None:
        titik_tengah = utils_avoid.titik_tengah_1_bola(rb_terdekat, W, H, Y_ROI, reverse=True)
        if titik_tengah is not None:
            vel_belok = utils_avoid.cari_kecepatan_sudut(GAIN_YAW, titik_tengah, W)
            manda.set_velocity(vel_forward, vel_belok)
    
    else:
        if (getattr(manda, 'target_lat', None) is not None) and (getattr(manda, 'target_lon', None) is not None):
            manda.move(manda.target_lat, manda.target_lon)
    
    # Draw target point
    if titik_tengah is not None:
        titik_tengah_int = (int(titik_tengah[0]), int(titik_tengah[1]))
        cv2.circle(img, titik_tengah_int, 5, (255, 0, 255), -1)
    
    steering_start = None
    return steering_start, vel_forward, "BACK"


# MAIN DETECTION LOOP

def _loop_and_detect(manda, cam, trt_yolo, conf_th, vis, lat, lon, idx,
                     tracker=None, classes_to_track=None,
                     cls_dict=None, allow_start_ids=None, allow_end_ids=None,
                     end_wp=-1, per_cls_thresh=None, back_mode=False):
    """
    Main detection and navigation loop
    
    Calculate LIMIT_START/END/BACK BEFORE loop (not inside)
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    wp_status = "INIT"
    last_wp_reached = -1

    H = getattr(cam, "img_height", None)
    W = getattr(cam, "img_width", None)
    if H is not None:
        H = int(H)
    if W is not None:
        W = int(W)

    idx = 0
    
    # Sync manda state
    try:
        if 0 <= idx < len(lat):
            manda.target_lat = lat[idx]
            manda.target_lon = lon[idx]
            manda.current_wp_index = int(idx)
    except Exception:
        pass

    assert cls_dict is not None
    num_classes = len(cls_dict)
    names = [cls_dict[i] for i in range(num_classes)]
    name2id = {v: k for k, v in cls_dict.items()}

    utils_avoid = UTILS_MISI_AVOID()
  
    # Calculate ROI parameters based on frame size
    Y_ROI, KONSTANTA_BOLA = get_roi_params(H, W)
    
    GAIN_YAW = {
        "LEFT": 2.5,
        "RIGHT": 2.0
    }
    
    red = (0, 0, 255)
    green = (0, 255, 0)
    
    # Tracking statistics
    seen_ids = [set() for _ in range(num_classes)]
    cum_counts = [0 for _ in range(num_classes)]

    end_progress = {"gl_done": False, "rl_done": False, "announced_back": False}
 
    back_phase = bool(back_mode)
    last_phase = None
    
    # Calculate phase limits BEFORE loop, not inside
    LIMIT_START = START_RANGE
    LIMIT_END = LIMIT_START + END_RANGE
    LIMIT_BACK = LIMIT_END + BACK_RANGE
    
    while not rospy.is_shutdown():
        try:
            if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
                break
        except Exception:
            pass

        img = cam.read()
        if img is None:
            break

        if W is None or H is None:
            H, W = img.shape[:2]
            H, W = int(H), int(W)
            # Recalculate ROI params if dimensions changed
            Y_ROI, KONSTANTA_BOLA = get_roi_params(H, W)

        # Trapezoid parameters
        trap_top_half = TRAP_TOP_HALF_WIDTH
        trap_bottom_half = int(TRAP_BOTTOM_HALF_RATIO * W)
        
        # Phase limits are calculated BEFORE loop
        
        # Detection
        try:
            res = trt_yolo.detect(img, conf_th)
        except Exception as e:
            rospy.logwarn(f"trt_yolo.detect raised exception: {e}")
            res = ([], [], [])
        if res is None:
            boxes, confs, clss = [], [], []
        else:
            try:
                boxes, confs, clss = res
                if boxes is None or confs is None or clss is None:
                    boxes, confs, clss = [], [], []
            except Exception:
                boxes, confs, clss = [], [], []

        # Determine current phase
        cur_wp = int(getattr(manda, 'current_wp_index'))
        forced_end = bool(getattr(manda, 'end_phase', False))
        
        # Use pre-calculated limits
        if idx < LIMIT_START:
            phase = "START"
        elif idx < LIMIT_END:
            phase = "END"
        elif idx < LIMIT_BACK:
            phase = "BACK"
        else:
            phase = "DONE"
        
        rospy.logdebug_throttle(2.0, f"[AVOID] idx={idx}, phase={phase}, limits: START={LIMIT_START}, END={LIMIT_END}, BACK={LIMIT_BACK}")
                
        if phase != last_phase:
            rospy.loginfo(f"[AVOID] Phase transition -> {phase}")
            last_phase = phase

        # Filter classes by phase
        if phase == "START":
            allowed_ids = allow_start_ids
        elif phase == "END":
            allowed_ids = allow_end_ids
        else:  # BACK: no filtering
            allowed_ids = None

        boxes, confs, clss = _filter_by_allowed(
            boxes, confs, clss, conf_th, per_cls_thresh, allowed_ids
        )

        # Per-frame counts for tracking
        frame_counts = [0 for _ in range(num_classes)]

        # Tracking
        if tracker is not None:
            dets = []
            for (box, sc, c) in zip(boxes, confs, clss):
                c = int(c)
                if (classes_to_track is None) or (c in classes_to_track):
                    x1b, y1b, x2b, y2b = box
                    dets.append([float(x1b), float(y1b), float(x2b), float(y2b), float(sc), int(c)])
            tracks = tracker.update(dets, (H, W)) if (H is not None and W is not None) else []
            for (tid, x1b, y1b, x2b, y2b, cls_id, score) in tracks:
                c = int(cls_id)
                if 0 <= c < num_classes:
                    frame_counts[c] += 1
                    if tid not in seen_ids[c]:
                        seen_ids[c].add(tid)
                        cum_counts[c] += 1
        
        # Execute phase behavior
        if phase == "START":
            steering, vel_forward, note = _avoid_start_step(
                img, manda, W, H, boxes, confs, clss,
                name2id, utils_avoid, GAIN_YAW, Y_ROI
            )
            
        elif phase == "END":
            steering, vel_forward, note, lat, lon, idx = _avoid_end_step(
                manda, W, H, boxes, confs, clss,
                name2id, utils_avoid, end_progress, idx, lat, lon,
                KONSTANTA_BOLA
            )
            
        else:  # BACK
            steering, vel_forward, note = _avoid_back_step(
                img, manda, W, H, boxes, confs, clss,
                name2id, utils_avoid, end_progress, GAIN_YAW, Y_ROI
            )
            phase = note

        # Check mission completion
        if manda.has_reached_final() and idx >= (len(lat) - 1):
            frame_dict = {names[i]: frame_counts[i] for i in range(num_classes)}
            cum_dict = {names[i]: cum_counts[i] for i in range(num_classes)}
            try:
                manda.report(
                    task="avoid",
                    counts={"frame": frame_dict, "cum": cum_dict},
                    extra={
                        "wp_index": idx,
                        "last_wp_reached": last_wp_reached,
                        "wp_status": "DONE",
                        "phase": phase,
                        "status": "done"
                    },
                    throttle_key="avoid_done"
                )
            except Exception:
                pass
            return True
               
        # Waypoint reached check
        if manda.has_reached_target():
            last_wp_reached = idx
            wp_status = f"WP {idx + 1} REACHED"

            rospy.loginfo(
                "[AVOID] Waypoint %d reached | lat=%.7f lon=%.7f",
                idx + 1, lat[idx], lon[idx]
            )

            idx += 1
            if idx < len(lat):
                manda.target_lat = lat[idx]
                manda.target_lon = lon[idx]

                rospy.loginfo(
                    "[AVOID] Moving to WP %d | lat=%.7f lon=%.7f",
                    idx, lat[idx], lon[idx]
                )
            else:
                wp_status = "FINAL WP REACHED"

        # Draw bboxes
        img = vis.draw_bboxes(img, boxes, confs, clss)

        # FPS
        toc = time.time()
        dt = max(toc - tic, 1e-6)
        fps = 1.0 / dt
        tic = toc
        img = show_fps(img, fps)

        frame_dict = {names[i]: frame_counts[i] for i in range(num_classes)}
        
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        hud = [
            f"{now_str}",
            f"AVOID | WP {idx + 1}/{len(lat)}",
            f"STATUS: {wp_status} | PHASE: {phase}",
            " ".join(f"{names[i].upper()}:{frame_counts[i]}" for i in range(num_classes)),
            f"FWD {SPEED_MISI:.1f} m/s" + ("" if steering is None else f" | YAW {steering:+.2f}")
        ]

        draw_hud(img, hud)
        
        # Draw trapezoid ROI
        _draw_trapezoid_roi(img, W, H, trap_top_half, trap_bottom_half, green, 2)
        y_mid = (H // 2 + H) // 2
        cv2.line(img, (0, Y_ROI), (W, Y_ROI), red, 1, cv2.LINE_AA)

        # Report status
        cum_dict = {names[i]: cum_counts[i] for i in range(num_classes)}
        try:
            manda.report(
                task="avoid",
                counts={"frame": frame_dict, "cum": cum_dict},
                extra={"wp_index": cur_wp, "phase": phase, "steering": steering, "status": "running"}
            )
        except Exception:
            pass

        cv2.imshow(WINDOW_NAME, img)
        k = cv2.waitKey(1)
        if k in (27, ord('q')):
            break
        elif k in (ord('f'), ord('F')):
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)
    
    return True


def run(manda, lat, lon, args, cam, trt=None):
    """Entry point for avoid mission"""
    cls_dict = get_cls_dict_by_model(args.avoid_model, override_num=args.avoid_category_num)
    color_map = get_color_map_by_model(args.avoid_model)
    vis = BBoxVisualization(cls_dict, color_map=color_map)
    
    if trt is None:
        engine_path = os.path.join(args.avoid_engine_dir, f'{args.avoid_model}.trt')
        if not os.path.isfile(engine_path):
            raise SystemExit(f'ERROR: TRT engine not found: {engine_path}')
        trt = TrtYOLO(args.avoid_model, len(cls_dict), args.avoid_letter_box)

        # Warmup
        try:
            h = int(getattr(cam, "img_height", 360))
            w = int(getattr(cam, "img_width", 640))
            dummy = np.zeros((h, w, 3), dtype=np.uint8)
            _ = trt.detect(dummy, max(0.05, float(getattr(args, "avoid_conf_thresh", 0.15))))
        except Exception:
            pass

    # Tracker setup
    classes_to_track = _parse_track_classes(args.avoid_track_classes, len(cls_dict))
    tracker = None if args.avoid_tracker == 'none' else _attach_tracker(args.avoid_tracker, classes_to_track, args)

    # Window setup
    try:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            open_window(WINDOW_NAME, 'Mission: AVOID', cam.img_width, cam.img_height)
    except Exception:
        open_window(WINDOW_NAME, 'Mission: AVOID', cam.img_width, cam.img_height)

    # Parse allow lists and per-class thresholds
    allow_start_ids = _names_to_ids(args.avoid_allow_start, cls_dict)
    allow_end_ids = _names_to_ids(args.avoid_allow_end, cls_dict)
    per_cls_thresh = _parse_class_thresh(args.avoid_class_thresh, cls_dict)

    try:
        # Set final target
        manda.final_lat, manda.final_lon = float(lat[-1]), float(lon[-1])
        
        ok = _loop_and_detect(
            manda, cam, trt, args.avoid_conf_thresh, vis,
            lat, lon, 0,
            tracker=tracker, classes_to_track=classes_to_track,
            cls_dict=cls_dict,
            allow_start_ids=allow_start_ids, allow_end_ids=allow_end_ids,
            end_wp=args.avoid_end_wp, per_cls_thresh=per_cls_thresh,
            back_mode=bool(getattr(args, "avoid_back_mode", False)),
        )
        return ok
    finally:
        if not bool(getattr(args, "avoid_keep_window", False)):
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except Exception:
                pass
