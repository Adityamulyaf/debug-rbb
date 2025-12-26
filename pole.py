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

# Mission parameters
SPEED_MISI = 1.5
SPEED_TURNING = 1.5  # Speed when turning

LOGINFO_THROTTLE = 0.5

# ROI parameters
def get_roi_params(H, W):
    """Calculate ROI parameters based on frame dimensions"""
    Y_TOP = int(0.46 * H)
    Y_BOT = H
    X_MID = W // 2
    X_LEFT = int(0.156 * W)
    X_RIGHT = int(0.844 * W)
    XL = (X_LEFT + X_MID) // 2
    XR = (X_MID + X_RIGHT) // 2
    return Y_TOP, Y_BOT, X_MID, X_LEFT, X_RIGHT, XL, XR


# Position categories
HIJAU_KANAN_JAUH = "HIJAU_KANAN_JAUH"
HIJAU_KANAN_DEKAT = "HIJAU_KANAN_DEKAT"
HIJAU_KIRI_JAUH = "HIJAU_KIRI_JAUH"
HIJAU_KIRI_DEKAT = "HIJAU_KIRI_DEKAT"

MERAH_KIRI_JAUH = "MERAH_KIRI_JAUH"
MERAH_KIRI_DEKAT = "MERAH_KIRI_DEKAT"
MERAH_KANAN_JAUH = "MERAH_KANAN_JAUH"
MERAH_KANAN_DEKAT = "MERAH_KANAN_DEKAT"


# ARGUMENT
def register(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("Mission Pole")
    g.add_argument('--pole-model', type=str, default='pole2')
    g.add_argument('--pole-category-num', type=int, default=2)
    g.add_argument('--pole-conf-thresh', type=float, default=0.65)
    g.add_argument('--pole-engine-dir', type=str, default='yolo')
    g.add_argument('--pole-letter-box', action='store_true')

    g.add_argument('--pole-tracker', type=str, default='bytetrack', choices=['none','bytetrack','ocsort'])
    g.add_argument('--pole-track-classes', type=str, default='0,1')

    g.add_argument('--pole-track-high', type=float, default=0.35)
    g.add_argument('--pole-track-low', type=float, default=0.05)
    g.add_argument('--pole-track-iou', type=float, default=0.20)
    g.add_argument('--pole-track-buffer', type=int, default=100)


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
    cy = y2  # Bottom of box
    return cx, cy


def _choose_pos_from_center(cx, cy, W, x_mid, xl, xr, x_left, x_right, y_top, y_bot, cls_id):
    """Determine position category based on center & class"""
    if cy < y_top or cy > y_bot:
        return None
    
    if cls_id == 0:  # green - prioritized on right
        if x_mid <= cx <= W:
            if xr <= cx <= x_right:
                return HIJAU_KANAN_JAUH
            if x_mid <= cx < xr:
                return HIJAU_KANAN_DEKAT
        else:
            if 0 <= cx < int(0.25 * W):
                return HIJAU_KIRI_JAUH
            if int(0.25 * W) <= cx < x_mid:
                return HIJAU_KIRI_DEKAT
    else:  # red - prioritized on left
        if 0 <= cx <= x_mid:
            if x_left <= cx <= xl:
                return MERAH_KIRI_JAUH
            if xl < cx <= x_mid:
                return MERAH_KIRI_DEKAT
        else:
            if int(0.75 * W) <= cx <= W:
                return MERAH_KANAN_JAUH
            if x_mid < cx < int(0.75 * W):
                return MERAH_KANAN_DEKAT
    return None


# MAIN DETECTION LOOP
def _loop_and_detect(manda, cam, trt_yolo, conf_th, vis, lat, lon, 
                     tracker=None, classes_to_track=None):
    """
    Main detection and navigation loop
    
    STEERING: Uses DIRECT detections (boxes, confs, clss)
    COUNTING: Uses tracker (for unique ID counting)
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    wp_status = "INIT"
    last_wp_reached = -1

    H = getattr(cam, "img_height", None)
    W = getattr(cam, "img_width", None)
    idx = 0
    manda.current_wp_index = idx
    
    # Tracking state
    seen_green_ids, seen_red_ids = set(), set()
    cum_green, cum_red = 0, 0
    
    # Control rate
    control_rate = rospy.Rate(10)  # 10 Hz

    rospy.loginfo("[POLE] Starting mission - Steering: DIRECT detection, Counting: Tracker")

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

        # Calculate ROI
        y_top, y_bot, x_mid, x_left, x_right, xl, xr = get_roi_params(H, W)

        boxes, confs, clss = trt_yolo.detect(img, conf_th)

        # Guide lines colors
        white, red, green = (255, 255, 255), (0, 0, 255), (0, 255, 0)

        # STEERING DECISION
        
        green_pole = None
        red_pole = None
        
        # Find best green and red poles from direct detections
        best_green = (None, -1.0)  # (pos, score)
        best_red = (None, -1.0)
        
        for (box, conf, cls) in zip(boxes, confs, clss):
            cls = int(cls)
            
            cx, cy = _center_of(box)
            pos = _choose_pos_from_center(cx, cy, W, x_mid, xl, xr, 
                                         x_left, x_right, y_top, y_bot, cls)
            
            if pos is None:
                continue
            
            # Choose best by confidence
            if cls == 0:  # Green
                if conf > best_green[1]:
                    best_green = (pos, conf)
            else:  # Red
                if conf > best_red[1]:
                    best_red = (pos, conf)
        
        green_pole = best_green[0]
        red_pole = best_red[0]
        
        # STEERING COMMANDS
        vel_forward = SPEED_MISI
        steering = None
        
        if green_pole == HIJAU_KANAN_JAUH and red_pole == MERAH_KIRI_JAUH:
            vel_forward = SPEED_TURNING
            manda.set_velocity(vel_forward, 0.0)
            steering = +0.45
            rospy.loginfo_throttle(LOGINFO_THROTTLE, "Green right far, Red left far : Turn 0")
            
        elif green_pole == HIJAU_KANAN_JAUH and red_pole == MERAH_KIRI_DEKAT:
            vel_forward = SPEED_TURNING
            manda.set_velocity(vel_forward, -0.45)
            steering = +0.45
            rospy.loginfo_throttle(LOGINFO_THROTTLE, "Green right far, Red left near: Turn right 45")
            
        elif green_pole == HIJAU_KANAN_DEKAT and red_pole == MERAH_KIRI_DEKAT:
            vel_forward = SPEED_TURNING
            manda.set_velocity(vel_forward, 0.0)
            steering = +0.45
            rospy.loginfo_throttle(LOGINFO_THROTTLE, "Green right near, Red left near: Turn 0")
            
        elif green_pole == HIJAU_KANAN_DEKAT and red_pole == MERAH_KIRI_JAUH:
            vel_forward = SPEED_TURNING
            manda.set_velocity(vel_forward, 0.45)
            steering = +0.45
            rospy.loginfo_throttle(LOGINFO_THROTTLE, "Green right near, Red left far: Turn left 45")
        
        elif green_pole == HIJAU_KANAN_JAUH and red_pole is None:
            vel_forward = SPEED_TURNING
            manda.set_velocity(vel_forward, 0.45)
            steering = +0.45
            rospy.loginfo_throttle(LOGINFO_THROTTLE, "Green right far: Turn left 45")
            
        elif green_pole == HIJAU_KANAN_DEKAT and red_pole is None:
            vel_forward = SPEED_TURNING
            manda.set_velocity(vel_forward, 0.50)
            steering = +0.50
            rospy.loginfo_throttle(LOGINFO_THROTTLE, "Green right near: Turn left 50")
            
        elif green_pole == HIJAU_KIRI_JAUH and red_pole is None:
            vel_forward = SPEED_TURNING
            manda.set_velocity(vel_forward, 0.60)
            steering = +0.60
            rospy.loginfo_throttle(LOGINFO_THROTTLE, "Green left far: Turn left 60")
            
        elif green_pole == HIJAU_KIRI_DEKAT and red_pole is None:
            vel_forward = SPEED_TURNING
            manda.set_velocity(vel_forward, 0.50)
            steering = +0.50
            rospy.loginfo_throttle(LOGINFO_THROTTLE, "Green left near: Turn left 50")
            
        elif green_pole is None and red_pole == MERAH_KIRI_JAUH:
            vel_forward = SPEED_TURNING
            manda.set_velocity(vel_forward, -0.45)
            steering = -0.45
            rospy.loginfo_throttle(LOGINFO_THROTTLE, "Red left far: Turn right 45")
            
        elif green_pole is None and red_pole == MERAH_KIRI_DEKAT:
            vel_forward = SPEED_TURNING
            manda.set_velocity(vel_forward, -0.50)
            steering = -0.50
            rospy.loginfo_throttle(LOGINFO_THROTTLE, "Red left near: Turn right 50")
            
        elif green_pole is None and red_pole == MERAH_KANAN_JAUH:
            vel_forward = SPEED_TURNING
            manda.set_velocity(vel_forward, -0.60)
            steering = -0.60
            rospy.loginfo_throttle(LOGINFO_THROTTLE, "Red right far: Turn right 60")
            
        elif green_pole is None and red_pole == MERAH_KANAN_DEKAT:
            vel_forward = SPEED_TURNING
            manda.set_velocity(vel_forward, -0.50)
            steering = -0.50
            rospy.loginfo_throttle(LOGINFO_THROTTLE, "Red right near: Turn right 50")
            
        else:
            # No poles detected - use waypoint navigation
            steering = None
            manda.move(lat[idx], lon[idx])
            rospy.loginfo_throttle(LOGINFO_THROTTLE, "No poles: WP navigation")
        
        frame_green, frame_red = 0, 0
        
        if tracker is not None:
            # Prepare detections for tracker
            dets = []
            for (box, conf, cls) in zip(boxes, confs, clss):
                cls = int(cls)
                if (classes_to_track is None) or (cls in classes_to_track):
                    x1, y1, x2, y2 = box
                    dets.append([float(x1), float(y1), float(x2), float(y2), 
                               float(conf), int(cls)])
            
            # Update tracker
            tracks = tracker.update(dets, (H, W))
            
            # Count per frame + cumulative by unique ID
            for (tid, x1, y1, x2, y2, cls_id, score) in tracks:
                
                if int(cls_id) == 0:  # Green
                    frame_green += 1
                    if tid not in seen_green_ids:
                        seen_green_ids.add(tid)
                        cum_green += 1
                else:  # Red
                    frame_red += 1
                    if tid not in seen_red_ids:
                        seen_red_ids.add(tid)
                        cum_red += 1
            
            # Draw track IDs (for debugging)
            for (tid, x1, y1, x2, y2, cls_id, score) in tracks:
                
                label = f"ID{tid}"
                org = (int(x1) + 4, int(y1) - 6 if int(y1) - 6 > 10 else int(y1) + 14)
                cv2.putText(img, label, org, cv2.FONT_HERSHEY_PLAIN, 0.9, 
                           (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, label, org, cv2.FONT_HERSHEY_PLAIN, 0.9, 
                           (255, 255, 255), 1, cv2.LINE_AA)


        # Mission completion check
        
        if manda.has_reached_final() and idx >= (len(lat) - 1):
            manda.report(
                task="pole",
                counts={"frame_green": frame_green, "frame_red": frame_red,
                        "cum_green": cum_green, "cum_red": cum_red},
                extra={"wp_index": int(getattr(manda, "current_wp_index", -1)),
                       "green_pole": green_pole, "red_pole": red_pole, 
                       "status": "done"},
                throttle_key="pole_done"
            )
            rospy.loginfo("[POLE] Mission complete!")
            return True

        # Waypoint reached check
        if manda.has_reached_target() and last_wp_reached != idx:
            last_wp_reached = idx
            wp_status = f"WP {idx + 1} REACHED"

            rospy.loginfo("[POLE] Waypoint %d reached | lat=%.7f lon=%.7f",
                         idx + 1, lat[idx], lon[idx])

            idx += 1
            if idx < len(lat):
                manda.target_lat = lat[idx]
                manda.target_lon = lon[idx]
                manda.current_wp_index = idx
                rospy.loginfo("[POLE] Moving to WP %d | lat=%.7f lon=%.7f",
                             idx + 1, lat[idx], lon[idx])
            else:
                wp_status = "FINAL WP REACHED"

        # Visualization
        
        img = vis.draw_bboxes(img, boxes, confs, clss)

        # FPS
        toc = time.time()
        dt = max(toc - tic, 1e-6)
        fps = 1.0 / dt
        tic = toc
        img = show_fps(img, fps)

        # HUD
        now_str = datetime.now().strftime("%H:%M:%S")
        hud = [
            f"POLE | {now_str}",
            f"WP {idx + 1}/{len(lat)} | {wp_status}",
            f"Green: {green_pole or 'NONE'} | Red: {red_pole or 'NONE'}",
            f"Frame: G{frame_green} R{frame_red} | Count: G{cum_green} R{cum_red}",
            f"Speed {vel_forward:.1f} m/s" + ("" if steering is None else f" | Yaw {steering:+.2f}")
        ]
        draw_hud(img, hud)

        # Draw ROI lines
        cv2.line(img, (0, y_top), (W, y_top), white, 1)
        cv2.line(img, (x_mid, y_top), (x_mid, y_bot), white, 1)
        for x in (x_left, xl):
            cv2.line(img, (x, y_top), (x, y_bot), red, 1)
        for x in (xr, x_right):
            cv2.line(img, (x, y_top), (x, y_bot), green, 1)

        # Report status
        manda.report(
            task="pole",
            counts={"frame_green": frame_green, "frame_red": frame_red,
                    "cum_green": cum_green, "cum_red": cum_red},
            extra={"wp_index": idx, "last_wp_reached": last_wp_reached,
                   "wp_status": wp_status, "green_pole": green_pole, 
                   "red_pole": red_pole, "status": "running"}
        )

        # Display
        cv2.imshow(WINDOW_NAME, img)
        k = cv2.waitKey(1)
        if k in (27, ord('q')):
            break
        elif k in (ord('f'), ord('F')):
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)
        
        # Control rate
        control_rate.sleep()

    return True


def run(manda, lat, lon, args, cam, trt=None):
    """Entry point for pole mission"""
    cls_dict = get_cls_dict_by_model(args.pole_model, override_num=args.pole_category_num)
    color_map = get_color_map_by_model(args.pole_model)
    vis = BBoxVisualization(cls_dict, color_map=color_map)

    if trt is None:
        engine_path = os.path.join(args.pole_engine_dir, f'{args.pole_model}.trt')
        if not os.path.isfile(engine_path):
            raise SystemExit(f'ERROR: TRT engine not found: {engine_path}')
        trt = TrtYOLO(args.pole_model, len(cls_dict), args.pole_letter_box)

        # Warmup
        try:
            h = int(getattr(cam, "img_height", 360))
            w = int(getattr(cam, "img_width", 640))
            dummy = np.zeros((h, w, 3), dtype=np.uint8)
            _ = trt.detect(dummy, max(0.05, float(getattr(args, "pole_conf_thresh", 0.15))))
        except Exception:
            pass

    # Tracker (for counting only)
    classes_to_track = _parse_track_classes(args.pole_track_classes, len(cls_dict))
    tracker = _attach_tracker(args.pole_tracker, classes_to_track, args)

    # Window
    try:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            open_window(WINDOW_NAME, 'Mission: POLE', cam.img_width, cam.img_height)
    except Exception:
        open_window(WINDOW_NAME, 'Mission: POLE', cam.img_width, cam.img_height)

    try:
        manda.final_lat, manda.final_lon = lat[-1], lon[-1]
        ok = _loop_and_detect(
            manda, cam, trt, args.pole_conf_thresh, vis,
            lat=lat, lon=lon,
            tracker=tracker, classes_to_track=classes_to_track
        )
        rospy.loginfo(f"[POLE] Mission finished: {ok}")
        return ok
    finally:
        if not bool(getattr(args, "pole_keep_window", False)):
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except Exception:
                pass
