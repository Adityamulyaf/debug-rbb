#!/usr/bin/env python3
# pole.py

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

WINDOW_NAME = 'POLE'

SPEED_MISI = 2.0

# Hijau (tiang hijau) diutamakan di sisi kanan (area bawah frame)
HIJAU_KANAN_JAUH   = "HIJAU_KANAN_JAUH"    # xr..x_right
HIJAU_KANAN_DEKAT  = "HIJAU_KANAN_DEKAT"   # mid..xr
HIJAU_KIRI_JAUH    = "HIJAU_KIRI_JAUH"     # 0..0.25W
HIJAU_KIRI_DEKAT   = "HIJAU_KIRI_DEKAT"    # 0.25W..mid

# Merah (tiang merah) diutamakan di sisi kiri
MERAH_KIRI_JAUH    = "MERAH_KIRI_JAUH"     # x_left..xl
MERAH_KIRI_DEKAT   = "MERAH_KIRI_DEKAT"    # xl..mid
MERAH_KANAN_JAUH   = "MERAH_KANAN_JAUH"    # 0.75W..W
MERAH_KANAN_DEKAT  = "MERAH_KANAN_DEKAT"   # mid..0.75W


def register(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("Mission Pole")
    g.add_argument('--pole-model', type=str, default='pole2')
    g.add_argument('--pole-category-num', type=int, default=2, help='0=greenpole,1=redpole')
    g.add_argument('--pole-conf-thresh', type=float, default=0.80) #0.65
    g.add_argument('--pole-engine-dir', type=str, default='yolo')
    g.add_argument('--pole-letter-box', action='store_true')

    # Tracker & kelas yang di-track
    g.add_argument('--pole-tracker', type=str, default='bytetrack', choices=['none','bytetrack','ocsort'])
    g.add_argument('--pole-track-classes', type=str, default='0,1')

    # Tuning ByteTrack-lite (opsional)
    g.add_argument('--pole-track-high',   type=float, default=0.35, help='high score thresh (match stage-1)')
    g.add_argument('--pole-track-low',    type=float, default=0.05, help='low  score thresh (stage-2 extension)')
    g.add_argument('--pole-track-iou',    type=float, default=0.20, help='IoU match thresh')
    g.add_argument('--pole-track-buffer', type=int,   default=100,   help='missed buffer before drop track')


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
            high_thresh=args.pole_track_high,
            low_thresh=args.pole_track_low,
            match_thresh=args.pole_track_iou,
            track_buffer=args.pole_track_buffer,
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


def _center_of(box):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    y_bar = int((y1 + y2) / 2)
    cy = y2
    return cx, cy


def _choose_pos_from_center(cx, cy, W, x_mid, xl, xr, x_left, x_right, y_top, y_bot, cls_id):
    """Tentukan kategori posisi (HIJAU_*/MERAH_*) berdasarkan center & kelas."""
    if cy < y_top or cy > y_bot:
        return None
    if cls_id == 0:  # hijau diutamakan kanan
        if x_mid <= cx <= W:
            if xr <= cx <= x_right:   return HIJAU_KANAN_JAUH
            if x_mid <= cx < xr:      return HIJAU_KANAN_DEKAT
        else:
            if 0 <= cx < int(0.25 * W):        return HIJAU_KIRI_JAUH
            if int(0.25 * W) <= cx < x_mid:    return HIJAU_KIRI_DEKAT
    else:  # merah diutamakan kiri
        if 0 <= cx <= x_mid:
            if x_left <= cx <= xl:    return MERAH_KIRI_JAUH
            if xl < cx <= x_mid:      return MERAH_KIRI_DEKAT
        else:
            if int(0.75 * W) <= cx <= W:           return MERAH_KANAN_JAUH
            if x_mid < cx < int(0.75 * W):         return MERAH_KANAN_DEKAT
    return None

def _loop_and_detect(
        manda, cam, trt_yolo, conf_th, vis,
        lat, lon, tracker=None, classes_to_track=None):
    
    full_scrn = False
    fps = 0.0
    tic = time.time()
    wp_status = "INIT"
    last_wp_reached = -1

    H = getattr(cam, "img_height", None)
    W = getattr(cam, "img_width", None)
    idx = 0  # indeks waypoint saat ini
    manda.current_wp_index = idx
    
    # State untuk cumulative counting by track-id
    seen_green_ids, seen_red_ids = set(), set()
    cum_green, cum_red = 0, 0

    while not rospy.is_shutdown():
        # keluar jika window ditutup
        try:
            if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
                break
        except Exception:
            pass

        img = cam.read()
        if img is None:
            break

        if W is None or H is None:
            H, W = img.shape[:2] # H = Height, W = Width

        y_top = int(0.46 * H)
        y_bot = H
        x_mid = W // 2
        x_left  = int(0.156 * W)
        x_right = int(0.844 * W)
        xl = (x_left + x_mid) // 2
        xr = (x_mid + x_right) // 2

        # Deteksi
        boxes, confs, clss = trt_yolo.detect(img, conf_th)

        # Garis bantu
        white, red, green = (255, 255, 255), (0, 0, 255), (0, 255, 0)

        # Tracking & Counting
        frame_green, frame_red = 0, 0
        green_pole, red_pole = None, None  # representative position per class (pilih skor terbesar)

        if tracker is not None:
            # siapkan det list: [x1,y1,x2,y2,score,cls]
            dets = []
            Himg, Wimg = img.shape[0], img.shape[1]
            for (x1, y1, x2, y2), sc, c in zip(boxes, confs, clss):
                c = int(c)
                if (classes_to_track is None) or (c in classes_to_track):
                    dets.append([float(x1), float(y1), float(x2), float(y2), float(sc), int(c)])

            tracks = tracker.update(dets, (Himg, Wimg))  # (tid,x1,y1,x2,y2,cls,score)

            # count visible per frame + cumulative unique by tid
            best_green = (None, -1.0, None)  # (pos, score, tid)
            best_red   = (None, -1.0, None)

            for (tid, x1, y1, x2, y2, cls_id, score) in tracks:
                cx, cy = _center_of((x1, y1, x2, y2))

                if int(cls_id) == 0:
                    frame_green += 1
                    if tid not in seen_green_ids:
                        seen_green_ids.add(tid)
                        cum_green += 1
                    pos = _choose_pos_from_center(cx, cy, W, x_mid, xl, xr, x_left, x_right, y_top, y_bot, 0)
                    if pos is not None and score > best_green[1]:
                        best_green = (pos, score, tid)
                else:
                    frame_red += 1
                    if tid not in seen_red_ids:
                        seen_red_ids.add(tid)
                        cum_red += 1
                    pos = _choose_pos_from_center(cx, cy, W, x_mid, xl, xr, x_left, x_right, y_top, y_bot, 1)
                    if pos is not None and score > best_red[1]:
                        best_red = (pos, score, tid)

            green_pole = best_green[0]
            red_pole   = best_red[0]

            # gambar ID kecil di dekat bbox
            for (tid, x1, y1, x2, y2, cls_id, score) in tracks:
                label = f"ID{tid}"
                org = (int(x1) + 4, int(y1) - 6 if int(y1) - 6 > 10 else int(y1) + 14)
                cv2.putText(img, label, org, cv2.FONT_HERSHEY_PLAIN, 0.9, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(img, label, org, cv2.FONT_HERSHEY_PLAIN, 0.9, (255,255,255), 1, cv2.LINE_AA)
        # else: tidak ada fallback — biarkan frame_* = 0 dan pos = None

        # Command
        vel_forward = SPEED_MISI
        if   green_pole == HIJAU_KANAN_JAUH and red_pole is None:
            manda.set_velocity(vel_forward,  0.78); steering = +0.45
            print("Ke kiri 45")
        elif green_pole == HIJAU_KANAN_DEKAT and red_pole is None:
            manda.set_velocity(vel_forward,  0.83); steering = +0.50
            print("Ke kiri 50")
        elif green_pole == HIJAU_KIRI_JAUH   and red_pole is None:
            manda.set_velocity(vel_forward,  0.88); steering = +0.60
            print("Ke kiri 60")
        elif green_pole == HIJAU_KIRI_DEKAT  and red_pole is None:
            manda.set_velocity(vel_forward,  0.85); steering = +0.50
            print("Ke kiri 50")
        elif green_pole is None and red_pole == MERAH_KIRI_JAUH:
            manda.set_velocity(vel_forward, -0.65); steering = -0.45
            print("Ke kanan 45")
        elif green_pole is None and red_pole == MERAH_KIRI_DEKAT:
            manda.set_velocity(vel_forward, -0.60); steering = -0.50
            print("Ke kanan 50")
        elif green_pole is None and red_pole == MERAH_KANAN_JAUH:
            manda.set_velocity(vel_forward, -0.60); steering = -0.60
            print("Ke kanan 60")
        elif green_pole is None and red_pole == MERAH_KANAN_DEKAT:
            manda.set_velocity(vel_forward, -0.60); steering = -0.50
            print("Ke kanan 50")
        else:
            steering = None
            manda.move(lat[idx], lon[idx])
            print("Lurus")

        # selesai bila sampai target
        if manda.has_reached_final() and idx >= (len(lat) - 1):
            manda.report(
                task="pole",
                counts={"frame_green": frame_green, "frame_red": frame_red,
                        "cum_green": cum_green, "cum_red": cum_red},
                extra={"wp_index": int(getattr(manda, "current_wp_index", -1)),
                       "green_pole": green_pole, "red_pole": red_pole, "status": "done"},
                throttle_key="pole_done"
            )
            return True

        if manda.has_reached_target() and last_wp_reached != idx:
            last_wp_reached = idx
            wp_status = f"WP {idx + 1} REACHED"

            rospy.loginfo(
                "[POLE] Waypoint %d reached | lat=%.7f lon=%.7f",
                idx + 1, lat[idx], lon[idx]
            )

            idx += 1
            if idx < len(lat):
                manda.target_lat = lat[idx]
                manda.target_lon = lon[idx]
                manda.current_wp_index = idx

                rospy.loginfo(
                    "[POLE] Moving to WP %d | lat=%.7f lon=%.7f",
                    idx, lat[idx], lon[idx]
                )
            else:
                wp_status = "FINAL WP REACHED"

        # Visual
        img = vis.draw_bboxes(img, boxes, confs, clss)

        # FPS & HUD
        toc = time.time()
        dt = max(toc - tic, 1e-6)
        fps = 1.0 / dt  
        tic = toc
        img = show_fps(img, fps)
            
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        hud = [
            f"{now_str}",
            f"POLE | WP {idx + 1}/{len(lat)}",
            f"STATUS: {wp_status}",
            f"Frame: GREEN {frame_green} | RED {frame_red}",
            f"Count: GREEN {cum_green} | RED {cum_red}",
            f"FWD {SPEED_MISI:.1f} m/s" + ("" if steering is None else f" | YAW {steering:+.2f}")
        ]
        draw_hud(img, hud)
                
        cv2.line(img, (0, y_top), (W, y_top), white, 1)
        cv2.line(img, (x_mid, y_top), (x_mid, y_bot), white, 1)
        for x in (x_left, xl):
            cv2.line(img, (x, y_top), (x, y_bot), red, 1)
        for x in (xr, x_right):
            cv2.line(img, (x, y_top), (x, y_bot), green, 1)

        # Report
        manda.report(
            task="pole",
            counts={"frame_green": frame_green, "frame_red": frame_red,
                    "cum_green": cum_green, "cum_red": cum_red},
            extra={
                "wp_index": idx,
                "last_wp_reached": last_wp_reached,
                "wp_status": wp_status,
                "green_pole": green_pole,
                "red_pole": red_pole,
                "status": "running"
            }
        )

        # Key handling
        cv2.imshow(WINDOW_NAME, img)
        k = cv2.waitKey(1)
        if k in (27, ord('q')):  # ESC / q
            break
        elif k in (ord('f'), ord('F')):
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)

    return True


def run(manda, lat, lon, args, cam, trt=None):
    cls_dict = get_cls_dict_by_model(args.pole_model, override_num=args.pole_category_num)
    color_map = get_color_map_by_model(args.pole_model)
    vis = BBoxVisualization(cls_dict, color_map=color_map)

    if trt is None:
        engine_path = os.path.join(args.pole_engine_dir, f'{args.pole_model}.trt')
        if not os.path.isfile(engine_path):
            raise SystemExit(f'ERROR: TRT engine not found: {engine_path}')
        trt = TrtYOLO(args.pole_model, len(cls_dict), args.pole_letter_box)

        # warmup sekali (aman kalau gagal)
        try:
            h = int(getattr(cam, "img_height", 360))
            w = int(getattr(cam, "img_width", 640))
            dummy = np.zeros((h, w, 3), dtype=np.uint8)
            _ = trt.detect(dummy, max(0.05, float(getattr(args, "pole_conf_thresh", 0.15))))
        except Exception:
            pass

    # tracker
    classes_to_track = _parse_track_classes(args.pole_track_classes, len(cls_dict))
    tracker = _attach_tracker(args.pole_tracker, classes_to_track, args)

    # window
    try:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            open_window(WINDOW_NAME, 'Mission: POLE', cam.img_width, cam.img_height)
    except Exception:
        open_window(WINDOW_NAME, 'Mission: POLE', cam.img_width, cam.img_height)

    try:
        manda.final_lat, manda.final_lon = lat[-1], lon[-1]
        ok = _loop_and_detect(
            manda, cam, trt, args.pole_conf_thresh, vis,
            tracker=tracker, classes_to_track=classes_to_track,
            lat=lat, lon=lon
        )
        rospy.loginfo(f"Pole finished", {ok})
        return ok
    finally:
        if not bool(getattr(args, "pole_keep_window", False)):
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except Exception:
                pass
