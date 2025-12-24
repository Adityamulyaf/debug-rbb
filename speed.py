#!/usr/bin/env python3
# speed.py

import os
import time
import argparse
import cv2
import numpy as np
import rospy

from utils.display import open_window, set_display, show_fps, draw_hud
from utils.model_registry import get_cls_dict_by_model, get_color_map_by_model
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO


WINDOW_NAME = 'SPEED'

# Mission parameters
SPEED_MISI = 1.0

# Waypoint route configuration
SPEED_ROUTE_RED = [3, 4, 5, 6, 7]
SPEED_ROUTE_GREEN = [5, 4, 3, 6, 7]

WP_BRANCH_DECISION = 2
SPEED_WP_OFFSET = 0
MIN_MAX_DISTANCE = 1 # Minimum value for max_distance

# COUNTER BUFFER CONFIGURATION
GREENLIGHT_BUFFER_THRESH = 10 # Require 10 consecutive frames to confirm greenlight
REDLIGHT_BUFFER_THRESH = 10 # Require 10 consecutive frames to confirm redlight

# BLACKBALL AVOIDANCE CONFIGURATION
BLACKBALL_AVOID_DISTANCE = 150 # Pixel distance to start avoiding blackball
BLACKBALL_AVOID_GAIN = 0.8 # Gain for avoidance steering
BLACKBALL_SAFETY_MARGIN = 50 # Additional margin for safety


# ARGUMENT

def register(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("Mission Speed")
    g.add_argument('--speed-model', type=str, default='avoid')
    g.add_argument('--speed-category-num', type=int, default=5, help='0=blackball,1=greenball,2=greenlight,3=redball,4=redlight')
    g.add_argument('--speed-conf-thresh', type=float, default=0.15)
    g.add_argument('--speed-engine-dir', type=str, default='yolo')
    g.add_argument('--speed-letter-box', action='store_true')

    g.add_argument('--speed-tracker', type=str, default='bytetrack', choices=['none','bytetrack','ocsort'])
    g.add_argument('--speed-track-classes', type=str, default='0,1,2,3,4')

    g.add_argument('--speed-track-high', type=float, default=0.35, help='high score thresh (match stage-1)')
    g.add_argument('--speed-track-low', type=float, default=0.05, help='low score thresh (stage-2 extension)')
    g.add_argument('--speed-track-iou', type=float, default=0.20, help='IoU match thresh')
    g.add_argument('--speed-track-buffer', type=int, default=100, help='missed buffer before drop track')
    g.add_argument('--speed-keep-window', action='store_true', help='keep window open after mission')
    
    # Blackball avoidance parameters
    g.add_argument('--speed-blackball-distance', type=int, default=150, help='Distance to start avoiding blackball (pixels)')
    g.add_argument('--speed-blackball-gain', type=float, default=0.8, help='Gain for blackball avoidance steering')


# HELPER

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
            high_thresh=args.speed_track_high,
            low_thresh=args.speed_track_low,
            match_thresh=args.speed_track_iou,
            track_buffer=args.speed_track_buffer,
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
    cy = int(y_bar + (y2 - y_bar) / 2)
    return cx, cy


def _calculate_avoidance_vector(blackball_center, vessel_center, W, H):
    if blackball_center is None:
        return 0.0
    
    bx, by = blackball_center
    vx, vy = vessel_center
    
    # Vector from vessel to blackball
    dx = bx - vx
    dy = by - vy
    
    # Distance to blackball
    distance = np.sqrt(dx*dx + dy*dy)
    
    if distance < 1e-6:
        return 0.0
    
    # Avoidance direction: perpendicular to threat vector
    # If blackball is on left (dx < 0), steer right (negative yaw)
    # If blackball is on right (dx > 0), steer left (positive yaw)
    
    # Normalize distance (closer = stronger avoidance)
    threat_level = max(0, 1.0 - (distance / BLACKBALL_AVOID_DISTANCE))
    
    # Avoidance direction (opposite to blackball X position)
    avoidance_yaw = -np.sign(dx) * threat_level * BLACKBALL_AVOID_GAIN
    
    return np.clip(avoidance_yaw, -1.0, 1.0)


# MAIN DETECTION LOOP

def _loop_and_detect(
        manda, cam, trt_yolo, conf_th, vis,
        tracker=None, classes_to_track=None,
        blackball_avoid_distance=BLACKBALL_AVOID_DISTANCE,
        blackball_avoid_gain=BLACKBALL_AVOID_GAIN):
    """
    Main detection and navigation loop
    
    Features:
    - Counter buffer for greenlight/redlight detection stability
    - Blackball avoidance after route decision
    - Thread-safe speed_route_queue operations
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()

    H = getattr(cam, "img_height", None)
    W = getattr(cam, "img_width", None)

    # Always reset SPEED-specific state at start
    manda.speed_first_branch = None  # 3 or 4 (first branch)
    manda.speed_route_queue = []   # list of WP order (color route)
    manda.speed_active_wp = None  # WP currently targeting from queue
    
    # Counter buffers for light detection stability
    greenlight_counter = 0
    redlight_counter = 0
    
    # Confirmed light detection flags
    greenlight_confirmed = False
    redlight_confirmed = False

    while True:
        # Exit if window closed
        try:
            if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
                break
        except Exception:
            pass

        img = cam.read()
        if img is None:
            break

        if W is None or H is None:
            H, W = img.shape[:2]  # H = Height, W = Width

        y_top = int(0.46 * H)
        y_bot = H
        x_mid = W // 2
        x_left = int(0.156 * W)
        x_right = int(0.844 * W)
        xl = (x_left + x_mid) // 2
        xr = (x_mid + x_right) // 2

        # Debug current state
        curr_wp = int(getattr(manda, "current_wp_index", -1))
        try:
            reached_now = bool(manda.has_reached_target())
        except Exception:
            reached_now = False
        
        rospy.logdebug_throttle(5.0, 
            f"[SPEED] curr_wp={curr_wp}, route_queue={manda.speed_route_queue}, "
            f"active_wp={manda.speed_active_wp}, reached={reached_now}, "
            f"GL_cnt={greenlight_counter}, RL_cnt={redlight_counter}")

        # Detection
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        
        best_green = None
        best_red = None
        best_blackball = None  # NEW: Track blackball for avoidance

        if tracker is not None:
            # Prepare detection list: [x1,y1,x2,y2,score,cls]
            dets = []
            Himg, Wimg = img.shape[0], img.shape[1]
            for (x1, y1, x2, y2), sc, c in zip(boxes, confs, clss):
                c = int(c)
                if (classes_to_track is None) or (c in classes_to_track):
                    dets.append([float(x1), float(y1), float(x2), float(y2), float(sc), int(c)])

            tracks = tracker.update(dets, (Himg, Wimg))  # (tid,x1,y1,x2,y2,cls,score)

            # Select representative bbox per class (use score if available)
            best_green_score = -1.0
            best_red_score = -1.0
            best_blackball_score = -1.0
            
            for (tid, x1, y1, x2, y2, cls_id, score) in tracks:
                cls_id = int(cls_id)
                bbox = (int(x1), int(y1), int(x2), int(y2))
                try:
                    sc_val = float(score)
                except Exception:
                    sc_val = -1.0

                if cls_id == 1:  # greenlight (class 1 in model)
                    if (best_green is None) or (sc_val > best_green_score):
                        best_green = bbox
                        best_green_score = sc_val
                elif cls_id == 4:  # redlight (class 4 in model)
                    if (best_red is None) or (sc_val > best_red_score):
                        best_red = bbox
                        best_red_score = sc_val
                elif cls_id == 0:  # blackball (class 0 in model)
                    if (best_blackball is None) or (sc_val > best_blackball_score):
                        best_blackball = bbox
                        best_blackball_score = sc_val

                # Draw track ID near bbox
                label = f"ID{tid}"
                org = (int(x1) + 4,
                       int(y1) - 6 if int(y1) - 6 > 10 else int(y1) + 14)
                cv2.putText(img, label, org,
                            cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, label, org,
                            cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # Fallback: select representative based on YOLO conf
            best_green_conf = -1.0
            best_red_conf = -1.0
            best_blackball_conf = -1.0
            
            for (x1, y1, x2, y2), sc, c in zip(boxes, confs, clss):
                cls_id = int(c)
                bbox = (int(x1), int(y1), int(x2), int(y2))
                try:
                    sc_val = float(sc)
                except Exception:
                    sc_val = -1.0
                    
                if cls_id == 1:  # greenlight
                    if (best_green is None) or (sc_val > best_green_conf):
                        best_green = bbox
                        best_green_conf = sc_val
                elif cls_id == 4:  # redlight
                    if (best_red is None) or (sc_val > best_red_conf):
                        best_red = bbox
                        best_red_conf = sc_val
                elif cls_id == 0:  # blackball
                    if (best_blackball is None) or (sc_val > best_blackball_conf):
                        best_blackball = bbox
                        best_blackball_conf = sc_val


        # Update counters based on current frame detection
        if best_green is not None:
            greenlight_counter += 1
        else:
            greenlight_counter = max(0, greenlight_counter - 1)  # Decay slowly
        
        if best_red is not None:
            redlight_counter += 1
        else:
            redlight_counter = max(0, redlight_counter - 1)  # Decay slowly
        
        # Confirm light detection when counter exceeds threshold
        if greenlight_counter >= GREENLIGHT_BUFFER_THRESH and not greenlight_confirmed:
            greenlight_confirmed = True
            rospy.loginfo("[SPEED] GREENLIGHT CONFIRMED")
        
        if redlight_counter >= REDLIGHT_BUFFER_THRESH and not redlight_confirmed:
            redlight_confirmed = True
            rospy.loginfo("[SPEED] REDLIGHT CONFIRMED")

        # Calculate bbox centers
        green_center = None
        red_center = None
        blackball_center = None

        if best_green is not None:
            gx1, gy1, gx2, gy2 = best_green
            green_center = ((gx1 + gx2) // 2, (gy1 + gy2) // 2)

        if best_red is not None:
            rx1, ry1, rx2, ry2 = best_red
            red_center = ((rx1 + rx2) // 2, (ry1 + ry2) // 2)
        
        if best_blackball is not None:
            bx1, by1, bx2, by2 = best_blackball
            blackball_center = ((bx1 + bx2) // 2, (by1 + by2) // 2)

        if W is None or H is None:
            H, W = img.shape[:2]
        x_mid = W // 2
        xki = W // 4
        xka = (3 * W) // 4
        y_awal = (2 * H) // 3

        bG = None
        bR = None

        using_ball_logic = False
        vel_forward = SPEED_MISI
        max_yaw_rate = 1.0

        max_distance = max(MIN_MAX_DISTANCE, W // 2)

        # Classify GREEN position (greenball class 1)
        if green_center is not None:
            gx, gy = green_center
            if x_mid <= gx <= W:  # right side
                if y_awal <= gy <= H:
                    if x_mid <= gx <= xka:
                        bG = "Gka1"
                    elif xka < gx <= W:
                        bG = "Gka2"
            elif 0 <= gx <= x_mid:  # left side
                if y_awal <= gy <= H:
                    if 0 <= gx <= xki:
                        bG = "Gki2"
                    elif xki < gx <= x_mid:
                        bG = "Gki1"

        # Classify RED position (redball class 3)
        if red_center is not None:
            rx, ry = red_center
            if x_mid <= rx <= W:  # right side
                if y_awal <= ry <= H:
                    if x_mid <= rx <= xka:
                        bR = "Rka1"
                    elif xka < rx <= W:
                        bR = "Rka2"
            elif 0 <= rx <= x_mid:  # left side
                if y_awal <= ry <= H:
                    if 0 <= rx <= xki:
                        bR = "Rki2"
                    elif xki < rx <= x_mid:
                        bR = "Rki1"

        blackball_avoidance_yaw = 0.0
        
        # Only apply blackball avoidance if we're on a color route
        if manda.speed_route_queue and blackball_center is not None:
            vessel_center = (W // 2, H)  # Assume vessel at bottom center
            
            # Calculate distance to blackball
            dx = blackball_center[0] - vessel_center[0]
            dy = blackball_center[1] - vessel_center[1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Apply avoidance if blackball is too close
            if distance < blackball_avoid_distance:
                blackball_avoidance_yaw = _calculate_avoidance_vector(
                    blackball_center, vessel_center, W, H
                )
                rospy.loginfo_throttle(1.0, 
                    f"[SPEED] AVOIDING BLACKBALL: distance={distance:.1f}px, "
                    f"avoidance_yaw={blackball_avoidance_yaw:.3f}")
                
                # Visual indicator for blackball avoidance
                cv2.circle(img, blackball_center, int(blackball_avoid_distance), 
                          (0, 0, 255), 2)  # Red circle showing avoidance zone
                cv2.line(img, vessel_center, blackball_center, (0, 0, 255), 2)

        # If both green and red exist --> find midpoint (go between them)
        if (green_center is not None) and (red_center is not None):
            mx = (green_center[0] + red_center[0]) // 2
            my = (green_center[1] + red_center[1]) // 2

            # Distance from midpoint to vertical center line
            vertical_center_x = x_mid
            distance_to_center = mx - vertical_center_x

            # Yaw rate proportional to distance
            yaw_rate = (distance_to_center / float(max_distance)) * max_yaw_rate
            
            # Add blackball avoidance component
            yaw_rate += blackball_avoidance_yaw

            rospy.logdebug_throttle(2.0, f"[SPEED] Both lights detected, YAW RATE: {yaw_rate:.3f}")
            manda.set_velocity(vel_forward, -yaw_rate)
            using_ball_logic = True

            cv2.putText(img, f"Midpoint: ({mx},{my})",
                        (mx + 10, max(10, my - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.line(img, green_center, red_center, (255, 255, 255), 2)
            cv2.circle(img, (mx, my), 5, (255, 255, 255), -1)

        # If only one light (green or red) --> steer coarsely
        else:
            base_yaw = 0.0
            
            if bG == "Gka1" and bR is None:
                rospy.logdebug_throttle(2.0, "[SPEED] Green Gka1: Turn left")
                base_yaw = 0.5
                using_ball_logic = True
            elif bG == "Gka2" and bR is None:
                rospy.logdebug_throttle(2.0, "[SPEED] Green Gka2: Turn left")
                base_yaw = 0.5
                using_ball_logic = True
            elif bG == "Gki2" and bR is None:
                rospy.logdebug_throttle(2.0, "[SPEED] Green Gki2: Turn left sharp")
                base_yaw = 0.9
                using_ball_logic = True
            elif bG == "Gki1" and bR is None:
                rospy.logdebug_throttle(2.0, "[SPEED] Green Gki1: Turn left")
                base_yaw = 0.8
                using_ball_logic = True
            elif bG is None and bR == "Rki2":
                rospy.logdebug_throttle(2.0, "[SPEED] Red Rki2: Turn right")
                base_yaw = -0.5
                using_ball_logic = True
            elif bG is None and bR == "Rki1":
                rospy.logdebug_throttle(2.0, "[SPEED] Red Rki1: Turn right")
                base_yaw = -0.5
                using_ball_logic = True
            elif bG is None and bR == "Rka1":
                rospy.logdebug_throttle(2.0, "[SPEED] Red Rka1: Turn right sharp")
                base_yaw = -0.8
                using_ball_logic = True
            elif bG is None and bR == "Rka2":
                rospy.logdebug_throttle(2.0, "[SPEED] Red Rka2: Turn right sharp")
                base_yaw = -0.9
                using_ball_logic = True
            
            if using_ball_logic:
                # Combine base steering with blackball avoidance
                final_yaw = base_yaw + blackball_avoidance_yaw
                manda.set_velocity(vel_forward, final_yaw)

        # If not using light-based steering (use WP navigation)
        if not using_ball_logic:
            # Thread-safe access to speed_route_queue
            if hasattr(manda, '_speed_lock'):
                with manda._speed_lock:
                    has_route = bool(manda.speed_route_queue)
                    if has_route and manda.speed_active_wp is None and manda.speed_route_queue:
                        manda.speed_active_wp = manda.speed_route_queue[0]
            else:
                has_route = bool(manda.speed_route_queue)
                if has_route and manda.speed_active_wp is None and manda.speed_route_queue:
                    manda.speed_active_wp = manda.speed_route_queue[0]
            
            # If already have color route (decided at WP2)
            if has_route:
                # If no active_wp, set to first element of queue
                if manda.speed_active_wp is None and manda.speed_route_queue:
                    manda.speed_active_wp = manda.speed_route_queue[0]
                    try:
                        idx = manda.speed_active_wp + SPEED_WP_OFFSET
                        manda.target_lat = manda.latitude_array[idx]
                        manda.target_lon = manda.longitude_array[idx]
                        manda.current_wp_index = manda.speed_active_wp
                    except Exception as e:
                        rospy.logwarn("[SPEED] Failed to set initial color route target: %s", e)

                # Check if reached active WP
                if manda.has_reached_target():
                    rospy.loginfo(f"[SPEED] Reached WP{manda.speed_active_wp} (color route)")

                    # Thread-safe queue pop
                    if hasattr(manda, '_speed_lock'):
                        with manda._speed_lock:
                            if (manda.speed_route_queue and
                                    manda.speed_route_queue[0] == manda.speed_active_wp):
                                manda.speed_route_queue.pop(0)
                            
                            if manda.speed_route_queue:
                                manda.speed_active_wp = manda.speed_route_queue[0]
                                try:
                                    idx = manda.speed_active_wp + SPEED_WP_OFFSET
                                    manda.target_lat = manda.latitude_array[idx]
                                    manda.target_lon = manda.longitude_array[idx]
                                    manda.current_wp_index = manda.speed_active_wp
                                except Exception as e:
                                    rospy.logwarn("[SPEED] Failed to set next color route target: %s", e)
                                rospy.loginfo(f"[SPEED] Moving to next WP{manda.speed_active_wp}")
                            else:
                                rospy.loginfo("[SPEED] Color route complete, stopped at last WP")
                                manda.set_velocity(0.0, 0.0)
                                manda.report(
                                    task="speed",
                                    counts={},
                                    extra={"wp_index": int(getattr(manda, 'current_wp_index', -1)),
                                           "greenlight_confirmed": greenlight_confirmed,
                                           "redlight_confirmed": redlight_confirmed,
                                           "status": "done"},
                                    throttle_key="speed_done"
                                )
                                return True
                    else:
                        if (manda.speed_route_queue and
                                manda.speed_route_queue[0] == manda.speed_active_wp):
                            manda.speed_route_queue.pop(0)
                        
                        if manda.speed_route_queue:
                            manda.speed_active_wp = manda.speed_route_queue[0]
                            try:
                                idx = manda.speed_active_wp + SPEED_WP_OFFSET
                                manda.target_lat = manda.latitude_array[idx]
                                manda.target_lon = manda.longitude_array[idx]
                                manda.current_wp_index = manda.speed_active_wp
                            except Exception as e:
                                rospy.logwarn("[SPEED] Failed to set next color route target: %s", e)
                            rospy.loginfo(f"[SPEED] Moving to next WP{manda.speed_active_wp}")
                        else:
                            rospy.loginfo("[SPEED] Color route complete, stopped at last WP")
                            manda.set_velocity(0.0, 0.0)
                            manda.report(
                                task="speed",
                                counts={},
                                extra={"wp_index": int(getattr(manda, 'current_wp_index', -1)),
                                       "greenlight_confirmed": greenlight_confirmed,
                                       "redlight_confirmed": redlight_confirmed,
                                       "status": "done"},
                                throttle_key="speed_done"
                            )
                            return True
                else:
                    # Not yet reached, continue to target_lat/target_lon
                    # Apply blackball avoidance to waypoint navigation too
                    try:
                        if blackball_avoidance_yaw != 0.0:
                            # Steer with avoidance
                            manda.set_velocity(SPEED_MISI, blackball_avoidance_yaw)
                            rospy.logdebug_throttle(2.0, f"[SPEED] WP nav with blackball avoidance: {blackball_avoidance_yaw:.3f}")
                        else:
                            # Normal waypoint navigation
                            manda.move(manda.target_lat, manda.target_lon, speed=SPEED_MISI)
                            rospy.logdebug_throttle(5.0, f"[SPEED] Moving to WP{int(getattr(manda, 'current_wp_index', -1))} (color route)")
                    except Exception as e:
                        rospy.logwarn("[SPEED] Move failed: %s", e)

            # If NO color route yet
            else:
                if manda.speed_first_branch is not None:
                    rospy.loginfo("[SPEED] Color route already chosen but queue empty. Mission complete (safety).")
                    manda.set_velocity(0.0, 0.0)
                    manda.report(
                        task="speed",
                        counts={},
                        extra={"wp_index": int(getattr(manda, 'current_wp_index', -1)),
                               "greenlight_confirmed": greenlight_confirmed,
                               "redlight_confirmed": redlight_confirmed,
                               "status": "done"},
                        throttle_key="speed_done"
                    )
                    return True
                else:
                    # Not yet chosen branch -> default nav
                    if not manda.has_reached_target():
                        try:
                            manda.move(manda.target_lat, manda.target_lon, speed=SPEED_MISI)
                            rospy.logdebug_throttle(5.0, f"[SPEED] Moving straight to WP{int(getattr(manda, 'current_wp_index', -1))}")
                        except Exception as e:
                            rospy.logwarn("[SPEED] Move failed: %s", e)
                    else:
                        manda.set_velocity(0.0, 0.0)

        greenlight_info = {"bbox": best_green, "confirmed": greenlight_confirmed, 
                          "counter": greenlight_counter} if best_green is not None else None
        redlight_info = {"bbox": best_red, "confirmed": redlight_confirmed,
                        "counter": redlight_counter} if best_red is not None else None
        blackball_info = {"bbox": best_blackball, "center": blackball_center} if best_blackball is not None else None

        rospy.logdebug_throttle(5.0,
            f"[SPEED] GL: {greenlight_confirmed} ({greenlight_counter}/{GREENLIGHT_BUFFER_THRESH}) | "
            f"RL: {redlight_confirmed} ({redlight_counter}/{REDLIGHT_BUFFER_THRESH}) | "
            f"BB: {'YES' if blackball_info else 'NO'}")

        # ROUTE DECISION at WP_BRANCH_DECISION (using CONFIRMED lights)
        curr_wp = int(getattr(manda, "current_wp_index", -1))
        if (curr_wp == WP_BRANCH_DECISION and
                manda.speed_first_branch is None and
                not manda.speed_route_queue):

            if redlight_confirmed and not greenlight_confirmed:
                # Only REDLIGHT confirmed = use red route
                manda.speed_first_branch = 3
                manda.speed_route_queue = SPEED_ROUTE_RED.copy()
                manda.speed_active_wp = None
                rospy.loginfo(f"[SPEED] Decision at WP{WP_BRANCH_DECISION}: REDLIGHT CONFIRMED -- route {SPEED_ROUTE_RED}")

            elif greenlight_confirmed and not redlight_confirmed:
                # Only GREENLIGHT confirmed = use green route
                manda.speed_first_branch = 4
                manda.speed_route_queue = SPEED_ROUTE_GREEN.copy()
                manda.speed_active_wp = None
                rospy.loginfo(f"[SPEED] Decision at WP{WP_BRANCH_DECISION}: GREENLIGHT CONFIRMED -- route {SPEED_ROUTE_GREEN}")

            else:
                # FALLBACK: no confirmed light -> use counter to decide
                if redlight_counter > greenlight_counter:
                    rospy.logwarn(f"[SPEED] WP{WP_BRANCH_DECISION}: No confirmed light, using RED (counter-based)")
                    manda.speed_first_branch = 3
                    manda.speed_route_queue = SPEED_ROUTE_RED.copy()
                    manda.speed_active_wp = None
                else:
                    rospy.logwarn(f"[SPEED] WP{WP_BRANCH_DECISION}: No confirmed light, using GREEN (counter-based)")
                    manda.speed_first_branch = 4
                    manda.speed_route_queue = SPEED_ROUTE_GREEN.copy()
                    manda.speed_active_wp = None

        # Done if reached target
        if (manda.has_reached_target()
                and not manda.speed_route_queue
                and manda.speed_first_branch is None):
            manda.report(
                task="speed",
                counts={},
                extra={"wp_index": int(getattr(manda, 'current_wp_index', -1)),
                       "greenlight_confirmed": greenlight_confirmed,
                       "redlight_confirmed": redlight_confirmed,
                       "status": "done"},
                throttle_key="speed_done"
            )
            return True

        img = vis.draw_bboxes(img, boxes, confs, clss)

        # Draw representative bboxes with enhanced visuals
        if best_green is not None:
            x1, y1, x2, y2 = best_green
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            status = "CONFIRMED" if greenlight_confirmed else f"{greenlight_counter}/{GREENLIGHT_BUFFER_THRESH}"
            cv2.putText(img, f"GREEN {status}", (x1, max(10, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if best_red is not None:
            x1, y1, x2, y2 = best_red
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            status = "CONFIRMED" if redlight_confirmed else f"{redlight_counter}/{REDLIGHT_BUFFER_THRESH}"
            cv2.putText(img, f"RED {status}", (x1, max(10, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        if best_blackball is not None:
            x1, y1, x2, y2 = best_blackball
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(img, "BLACKBALL", (x1, max(10, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # FPS & HUD
        toc = time.time()
        dt = max(toc - tic, 1e-6)
        fps = 1.0 / dt
        tic = toc
        img = show_fps(img, fps)

        hud = [
            f"SPEED | WP {int(getattr(manda, 'current_wp_index', -1))}",
            f"GL: {'✓' if greenlight_confirmed else f'{greenlight_counter}/{GREENLIGHT_BUFFER_THRESH}'} | "
            f"RL: {'✓' if redlight_confirmed else f'{redlight_counter}/{REDLIGHT_BUFFER_THRESH}'}",
            f"BLACKBALL: {'AVOIDING' if blackball_avoidance_yaw != 0 else 'OK'}",
            f"FWD {SPEED_MISI:.1f} m/s"
        ]

        draw_hud(img, hud)

        # Report (running)
        manda.report(
            task="speed",
            counts={},
            extra={"wp_index": int(getattr(manda, 'current_wp_index', -1)),
                   "greenlight_confirmed": greenlight_confirmed,
                   "redlight_confirmed": redlight_confirmed,
                   "greenlight_counter": greenlight_counter,
                   "redlight_counter": redlight_counter,
                   "blackball_detected": blackball_info is not None,
                   "blackball_avoiding": blackball_avoidance_yaw != 0.0,
                   "status": "running"}
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


def run(manda, wp_start, wp_end, args, cam, trt=None):
    """Entry point for speed mission with block-style waypoint handling"""
    cls_dict = get_cls_dict_by_model(args.speed_model, override_num=args.speed_category_num)
    color_map = get_color_map_by_model(args.speed_model)
    vis = BBoxVisualization(cls_dict, color_map=color_map)

    # Engine creation
    if trt is None:
        engine_path = os.path.join(args.speed_engine_dir, f'{args.speed_model}.trt')
        if not os.path.isfile(engine_path):
            raise SystemExit(f'ERROR: TRT engine not found: {engine_path}')
        trt = TrtYOLO(args.speed_model, len(cls_dict), args.speed_letter_box)

        # Warmup
        try:
            h = int(getattr(cam, "img_height", 360))
            w = int(getattr(cam, "img_width", 640))
            dummy = np.zeros((h, w, 3), dtype=np.uint8)
            _ = trt.detect(dummy, max(0.05, float(getattr(args, "speed_conf_thresh", 0.15))))
        except Exception:
            pass

    # Tracker
    classes_to_track = _parse_track_classes(args.speed_track_classes, len(cls_dict))
    tracker = _attach_tracker(args.speed_tracker, classes_to_track, args)

    # Window setup
    try:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            open_window(WINDOW_NAME, 'Mission: SPEED', cam.img_width, cam.img_height)
    except Exception:
        open_window(WINDOW_NAME, 'Mission: SPEED', cam.img_width, cam.img_height)

    # Get blackball parameters from args
    bb_distance = getattr(args, 'speed_blackball_distance', BLACKBALL_AVOID_DISTANCE)
    bb_gain = getattr(args, 'speed_blackball_gain', BLACKBALL_AVOID_GAIN)

    try:
        ok_block = True

        # Safety: ensure manda has waypoint arrays
        lat_arr = getattr(manda, "latitude_array", None) or []
        lon_arr = getattr(manda, "longitude_array", None) or []

        # Clamp indices safely
        try:
            wp_start_i = int(wp_start)
            wp_end_i = int(wp_end)
        except Exception:
            rospy.logwarn("[SPEED] Invalid wp_start/wp_end values: %s..%s", wp_start, wp_end)
            return False

        if wp_start_i < 0:
            wp_start_i = 0
        if wp_end_i < wp_start_i:
            wp_end_i = wp_start_i

        try:
            manda.current_wp_index = int(wp_start_i)
            if wp_start_i < len(lat_arr) and wp_start_i < len(lon_arr):
                manda.target_lat = float(lat_arr[wp_start_i])
                manda.target_lon = float(lon_arr[wp_start_i])
        except Exception as e:
            rospy.logwarn("[SPEED] Failed to set initial target for block start: %s", e)

        # Call continuous detector once for whole block
        try:
            ok = _loop_and_detect(
                manda, cam, trt, args.speed_conf_thresh, vis,
                tracker=tracker, classes_to_track=classes_to_track,
                blackball_avoid_distance=bb_distance,
                blackball_avoid_gain=bb_gain
            )
            if not ok:
                rospy.logwarn("[SPEED] _loop_and_detect returned False for block %d..%d", wp_start_i, wp_end_i)
                ok_block = False
        except Exception as e:
            rospy.logwarn("[SPEED] Error during block execution: %s", e)
            ok_block = False

        # After run, sync manda state to last WP of block
        try:
            manda.current_wp_index = int(wp_end_i)
            if wp_end_i < len(lat_arr) and wp_end_i < len(lon_arr):
                manda.target_lat = float(lat_arr[wp_end_i])
                manda.target_lon = float(lon_arr[wp_end_i])
        except Exception:
            pass

        return bool(ok_block)
    
    finally:
        keep = bool(getattr(args, "speed_keep_window", False))
        if not keep:
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except Exception:
                pass
