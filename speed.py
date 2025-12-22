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

SPEED_MISI = 1.0

# =========================
#   KONFIGURASI RUTE WP
# =========================
# ASUMSI:
#   - WP1  : entry / balik akhir
#   - WP2  : waypoint keputusan (lihat red/green)
#   - WP3  : cabang kanan
#   - WP4  : cabang kiri
#   - WP5  : waypoint setelah loop (di gambar: yang ada sirene)
#
# Jika di WP2 hanya RED, maka:
#   rute: 3 -> 5 -> 6 -> 8 -> 9
SPEED_ROUTE_RED = [3, 4, 5, 6, 7]

# Jika di WP2 hanya GREEN, maka:
#   rute: 4 -> 5 -> 7 -> 8 -> 9
SPEED_ROUTE_GREEN = [5, 4, 3, 6, 7]

# WP tempat ambil keputusan cabang (di gambar: WP2)
WP_BRANCH_DECISION = 2

# Offset index untuk akses latitude_array / longitude_array:
#   - Jika latitude_array[1] = WP1  -> SPEED_WP_OFFSET = 0  (default)
#   - Jika latitude_array[0] = WP1  -> SPEED_WP_OFFSET = -1
SPEED_WP_OFFSET = 0


def register(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("Mission Speed")
    g.add_argument('--speed-model', type=str, default='avoid')
    g.add_argument('--speed-category-num', type=int, default=2, help='0=greenlight,1=redlight')
    g.add_argument('--speed-conf-thresh', type=float, default=0.15)
    g.add_argument('--speed-engine-dir', type=str, default='yolo')
    g.add_argument('--speed-letter-box', action='store_true')

    g.add_argument('--speed-tracker', type=str, default='bytetrack', choices=['none','bytetrack','ocsort'])
    g.add_argument('--speed-track-classes', type=str, default='0,1')

    g.add_argument('--speed-track-high',   type=float, default=0.35, help='high score thresh (match stage-1)')
    g.add_argument('--speed-track-low',    type=float, default=0.05, help='low  score thresh (stage-2 extension)')
    g.add_argument('--speed-track-iou',    type=float, default=0.20, help='IoU match thresh')
    g.add_argument('--speed-track-buffer', type=int,   default=100,   help='missed buffer before drop track')
    g.add_argument('--speed-keep-window', action='store_true', help='keep window open after mission')


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


def _loop_and_detect(
        manda, cam, trt_yolo, conf_th, vis,
        tracker=None, classes_to_track=None):
    
    full_scrn = False
    fps = 0.0
    tic = time.time()

    H = getattr(cam, "img_height", None)
    W = getattr(cam, "img_width", None)

    # --- SELALU reset STATE khusus mission SPEED di awal ---
    manda.speed_first_branch = None      # 3 atau 4 (cabang pertama)
    manda.speed_route_queue = []         # list urutan WP (rute warna)
    manda.speed_active_wp = None         # WP yang sedang dituju dari queue

    while True:
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
            H, W = img.shape[:2]  # H = Height, W = Width

        y_top = int(0.46 * H)
        y_bot = H
        x_mid = W // 2
        x_left  = int(0.156 * W)
        x_right = int(0.844 * W)
        xl = (x_left + x_mid) // 2
        xr = (x_mid + x_right) // 2

        # DEBUG state dasar
        curr_wp = int(getattr(manda, "current_wp_index", -1))
        try:
            reached_now = bool(manda.has_reached_target())
        except Exception:
            reached_now = False
        print(f"[DEBUG] curr_wp={curr_wp}, route_queue={manda.speed_route_queue}, "
              f"active_wp={manda.speed_active_wp}, reached={reached_now}")

        # Deteksi (YOLO)
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        
        best_green = None
        best_red = None

        if tracker is not None:
            # siapkan det list: [x1,y1,x2,y2,score,cls]
            dets = []
            Himg, Wimg = img.shape[0], img.shape[1]
            for (x1, y1, x2, y2), sc, c in zip(boxes, confs, clss):
                c = int(c)
                if (classes_to_track is None) or (c in classes_to_track):
                    dets.append([float(x1), float(y1), float(x2), float(y2), float(sc), int(c)])

            tracks = tracker.update(dets, (Himg, Wimg))  # (tid,x1,y1,x2,y2,cls,score)

            # pilih representative bbox per kelas (gunakan score jika tersedia)
            best_green_score = -1.0
            best_red_score = -1.0
            for (tid, x1, y1, x2, y2, cls_id, score) in tracks:
                cls_id = int(cls_id)
                bbox = (int(x1), int(y1), int(x2), int(y2))
                try:
                    sc_val = float(score)
                except Exception:
                    sc_val = -1.0

                if cls_id == 0:  # greenlight
                    if (best_green is None) or (sc_val > best_green_score):
                        best_green = bbox
                        best_green_score = sc_val
                else:            # redlight
                    if (best_red is None) or (sc_val > best_red_score):
                        best_red = bbox
                        best_red_score = sc_val

                # gambar ID kecil di dekat bbox
                label = f"ID{tid}"
                org = (int(x1) + 4,
                       int(y1) - 6 if int(y1) - 6 > 10 else int(y1) + 14)
                cv2.putText(img, label, org,
                            cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, label, org,
                            cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # fallback: pilih representative berdasarkan conf dari YOLO
            best_green_conf = -1.0
            best_red_conf = -1.0
            for (x1, y1, x2, y2), sc, c in zip(boxes, confs, clss):
                cls_id = int(c)
                bbox = (int(x1), int(y1), int(x2), int(y2))
                try:
                    sc_val = float(sc)
                except Exception:
                    sc_val = -1.0
                if cls_id == 0:
                    if (best_green is None) or (sc_val > best_green_conf):
                        best_green = bbox
                        best_green_conf = sc_val
                else:
                    if (best_red is None) or (sc_val > best_red_conf):
                        best_red = bbox
                        best_red_conf = sc_val

        # hitung center bbox kalau ada
        green_center = None
        red_center   = None

        if best_green is not None:
            gx1, gy1, gx2, gy2 = best_green
            green_center = ((gx1 + gx2) // 2, (gy1 + gy2) // 2)

        if best_red is not None:
            rx1, ry1, rx2, ry2 = best_red
            red_center = ((rx1 + rx2) // 2, (ry1 + ry2) // 2)

        if W is None or H is None:
            H, W = img.shape[:2]
        x_mid  = W // 2
        xki    = W // 4
        xka    = (3 * W) // 4
        y_awal = (2 * H) // 3

        bG = None
        bR = None

        using_ball_logic = False
        vel_forward = SPEED_MISI
        max_yaw_rate = 1.0
        max_distance = W // 2

        # klasifikasi posisi GREEN
        if green_center is not None:
            gx, gy = green_center
            if x_mid <= gx <= W:  # sisi kanan
                if y_awal <= gy <= H:
                    if x_mid <= gx <= xka:
                        bG = "Gka1"
                        print("Gka1")
                    elif xka < gx <= W:
                        bG = "Gka2"
                        print("Gka2")
            elif 0 <= gx <= x_mid:  # sisi kiri
                if y_awal <= gy <= H:
                    if 0 <= gx <= xki:
                        bG = "Gki2"
                        print("Gki2")
                    elif xki < gx <= x_mid:
                        bG = "Gki1"
                        print("Gki1")

        # klasifikasi posisi RED
        if red_center is not None:
            rx, ry = red_center
            if x_mid <= rx <= W:  # sisi kanan
                if y_awal <= ry <= H:
                    if x_mid <= rx <= xka:
                        bR = "Rka1"
                        print("Rka1")
                    elif xka < rx <= W:
                        bR = "Rka2"
                        print("Rka2")
            elif 0 <= rx <= x_mid:  # sisi kiri
                if y_awal <= ry <= H:
                    if 0 <= rx <= xki:
                        bR = "Rki2"
                        print("Rki2")
                    elif xki < rx <= x_mid:
                        bR = "Rki1"
                        print("Rki1")

        # kalau ada hijau dan merah --> cari midpoint di antaranya (masuk tengah)
        if (green_center is not None) and (red_center is not None):
            mx = (green_center[0] + red_center[0]) // 2
            my = (green_center[1] + red_center[1]) // 2

            # jarak midpoint ke garis tengah vertikal
            vertical_center_x = x_mid
            distance_to_center = mx - vertical_center_x

            # yaw_rate proporsional terhadap jarak
            yaw_rate = (distance_to_center / float(max_distance)) * max_yaw_rate

            print("YAW RATE: ", yaw_rate)
            manda.set_velocity(vel_forward, -yaw_rate)
            using_ball_logic = True

            cv2.putText(img, f"Midpoint: ({mx},{my})",
                        (mx + 10, max(10, my - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.line(img, green_center, red_center, (255, 255, 255), 2)
            cv2.circle(img, (mx, my), 5, (255, 255, 255), -1)

        # kalau hanya ada satu bola (hijau / merah) -> belok kasar
        else:
            if bG == "Gka1" and bR is None:
                print("Belok kiri COKK")
                manda.set_velocity(vel_forward, 0.5)
                using_ball_logic = True
            elif bG == "Gka2" and bR is None:
                print("Belok kiri COKK")
                manda.set_velocity(vel_forward, 0.5)
                using_ball_logic = True
            elif bG == "Gki2" and bR is None:
                print("Belok kiri COKK")
                manda.set_velocity(vel_forward, 0.9)
                using_ball_logic = True
            elif bG == "Gki1" and bR is None:
                print("Belok kiri COKK")
                manda.set_velocity(vel_forward, 0.8)
                using_ball_logic = True
            elif bG is None and bR == "Rki2":
                print("Belok kanan COKK")
                manda.set_velocity(vel_forward, -0.5)
                using_ball_logic = True
            elif bG is None and bR == "Rki1":
                print("Belok kanan COKK")
                manda.set_velocity(vel_forward, -0.5)
                using_ball_logic = True
            elif bG is None and bR == "Rka1":
                print("Belok kanan COKK")
                manda.set_velocity(vel_forward, -0.8)
                using_ball_logic = True
            elif bG is None and bR == "Rka2":
                print("Belok kanan COKK")
                manda.set_velocity(vel_forward, -0.9)
                using_ball_logic = True

        # kalau tidak ada green/red yang dipakai untuk steering (pakai navigasi WP)
        if not using_ball_logic:
            # --- kalau SUDAH ada rute warna (diputuskan di WP2) ---
            if manda.speed_route_queue:
                # kalau belum ada active_wp, set ke elemen pertama queue
                if manda.speed_active_wp is None and manda.speed_route_queue:
                    manda.speed_active_wp = manda.speed_route_queue[0]
                    try:
                        idx = manda.speed_active_wp + SPEED_WP_OFFSET
                        manda.target_lat = manda.latitude_array[idx]
                        manda.target_lon = manda.longitude_array[idx]
                        manda.current_wp_index = manda.speed_active_wp
                    except Exception as e:
                        rospy.logwarn("[SPEED] Gagal set target awal rute warna: %s", e)

                # cek apakah sudah sampai di WP aktif
                if manda.has_reached_target():
                    print(f"[SPEED] Sampai di WP{manda.speed_active_wp} (rute warna)")

                    # buang WP yang barusan dicapai
                    if (manda.speed_route_queue and
                            manda.speed_route_queue[0] == manda.speed_active_wp):
                        manda.speed_route_queue.pop(0)

                    if manda.speed_route_queue:
                        # pindah ke WP berikutnya
                        manda.speed_active_wp = manda.speed_route_queue[0]
                        try:
                            idx = manda.speed_active_wp + SPEED_WP_OFFSET
                            manda.target_lat = manda.latitude_array[idx]
                            manda.target_lon = manda.longitude_array[idx]
                            manda.current_wp_index = manda.speed_active_wp
                        except Exception as e:
                            rospy.logwarn("[SPEED] Gagal set target berikutnya rute warna: %s", e)
                        print(f"[SPEED] Pindah target ke WP{manda.speed_active_wp}")
                    else:
                        # queue kosong -> rute selesai
                        print("[SPEED] Rute warna selesai, berhenti di WP terakhir")
                        manda.set_velocity(0.0, 0.0)
                        manda.report(
                            task="speed",
                            counts={},
                            extra={"wp_index": int(getattr(manda, 'current_wp_index', -1)),
                                   "greenlight": None, "redlight": None,
                                   "status": "done"},
                            throttle_key="speed_done"
                        )
                        return True
                else:
                    # belum sampai, lanjut menuju target_lat/target_lon
                    try:
                        manda.move(manda.target_lat, manda.target_lon, speed=SPEED_MISI)
                        print(f"[SPEED] Menuju WP{int(getattr(manda, 'current_wp_index', -1))} (rute warna)")
                    except Exception as e:
                        rospy.logwarn("[SPEED] Move gagal %s", e)

            # --- kalau BELUM ada rute warna ---
            else:
                if manda.speed_first_branch is not None:
                    # artinya: rute warna sudah pernah dipilih dan sekarang queue kosong
                    # -> anggap mission selesai (safety net tambahan)
                    print("[SPEED] Rute warna sudah dipilih tapi queue kosong. Mission speed selesai (safety).")
                    manda.set_velocity(0.0, 0.0)
                    manda.report(
                        task="speed",
                        counts={},
                        extra={"wp_index": int(getattr(manda, 'current_wp_index', -1)),
                               "greenlight": None, "redlight": None,
                               "status": "done"},
                        throttle_key="speed_done"
                    )
                    return True
                else:
                    # belum pernah pilih cabang -> nav default biasa
                    if not manda.has_reached_target():
                        try:
                            manda.move(manda.target_lat, manda.target_lon, speed=SPEED_MISI)
                            print(f"[SPEED] Lurus ke WP",
                                  int(getattr(manda, "current_wp_index", -1)))
                        except Exception as e:
                            rospy.logwarn("[SPEED] Move gagal %s", e)
                    else:
                        manda.set_velocity(0.0, 0.0)

        greenlight_ada = best_green is not None
        redlight_ada   = best_red is not None

        greenlight_info = {"bbox": best_green} if best_green is not None else None
        redlight_info   = {"bbox": best_red}   if best_red is not None else None

        print(f"[SPEED] Greenlight: {'YES' if greenlight_ada else 'NO'}"
              + (f" | bbox={greenlight_info['bbox']}" if greenlight_info else ""),
              f"| Redlight: {'YES' if redlight_ada else 'NO'}"
              + (f" | bbox={redlight_info['bbox']}" if redlight_info else ""))

        # --- Keputusan rute di WP_BRANCH_DECISION (misal WP2) ---
        curr_wp = int(getattr(manda, "current_wp_index", -1))
        if (curr_wp == WP_BRANCH_DECISION and
                manda.speed_first_branch is None and
                not manda.speed_route_queue):

            if redlight_ada and not greenlight_ada:
                # hanya RED -> pakai rute merah
                manda.speed_first_branch = 3
                manda.speed_route_queue = SPEED_ROUTE_RED.copy()
                manda.speed_active_wp = None
                print(f"[SPEED] Keputusan di WP{WP_BRANCH_DECISION}: RED -> rute {SPEED_ROUTE_RED}")

            elif greenlight_ada and not redlight_ada:
                # hanya GREEN -> pakai rute hijau
                manda.speed_first_branch = 4
                manda.speed_route_queue = SPEED_ROUTE_GREEN.copy()
                manda.speed_active_wp = None
                print(f"[SPEED] Keputusan di WP{WP_BRANCH_DECISION}: GREEN -> rute {SPEED_ROUTE_GREEN}")

            else:
                # FALLBACK: tidak lihat apa-apa -> anggap RED
                rospy.logwarn("[SPEED] WP2: tidak melihat light, fallback ke rute RED")
                manda.speed_first_branch = 3
                manda.speed_route_queue = SPEED_ROUTE_RED.copy()
                manda.speed_active_wp = None

        # selesai bila sampai target HANYA kalau
        # tidak pakai rute warna dan belum pernah pilih cabang
        if (manda.has_reached_target()
                and not manda.speed_route_queue
                and manda.speed_first_branch is None):
            manda.report(
                task="speed",
                counts={},
                extra={"wp_index": int(getattr(manda, 'current_wp_index', -1)),
                       "greenlight": greenlight_info, "redlight": redlight_info,
                       "status": "done"},
                throttle_key="speed_done"
            )
            return True

        img = vis.draw_bboxes(img, boxes, confs, clss)

        # Jika representative bbox ada, gambar kotak tebal / label tambahan
        if best_green is not None:
            x1, y1, x2, y2 = best_green
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "GREEN", (x1, max(10, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if best_red is not None:
            x1, y1, x2, y2 = best_red
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, "RED", (x1, max(10, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # FPS & HUD
        toc = time.time()
        dt = max(toc - tic, 1e-6)
        fps = 1.0 / dt
        tic = toc
        img = show_fps(img, fps)

        hud = [
            f"SPEED | WP {int(getattr(manda, 'current_wp_index', -1))}",
            f"GREENLIGHT: {'YES' if greenlight_ada else 'NO'}",
            f"REDLIGHT: {'YES' if redlight_ada else 'NO'}",
            f"FWD {SPEED_MISI:.1f} m/s"
        ]

        draw_hud(img, hud)

        # Report (running)
        manda.report(
            task="speed",
            counts={},
            extra={"wp_index": int(getattr(manda, 'current_wp_index', -1)),
                   "greenlight": greenlight_info, "redlight": redlight_info,
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
    """
    BLOCK-STYLE run for SPEED:
      signature: run(manda, wp_start, wp_end, args, cam, trt=None)
      - akan mengeksekusi mission SPEED untuk seluruh rentang WP [wp_start..wp_end].
      - internal state (speed_route_queue, speed_first_branch, ...) disimpan di `manda`
      - setelah selesai, fungsi sinkronkan manda.current_wp_index ke wp_end.
    """
    cls_dict = get_cls_dict_by_model(args.speed_model, override_num=args.speed_category_num)
    color_map = get_color_map_by_model(args.speed_model)
    vis = BBoxVisualization(cls_dict, color_map=color_map)

    # --- engine creation (UNCHANGED) ---
    if trt is None:
        engine_path = os.path.join(args.speed_engine_dir, f'{args.speed_model}.trt')
        if not os.path.isfile(engine_path):
            raise SystemExit(f'ERROR: TRT engine not found: {engine_path}')
        trt = TrtYOLO(args.speed_model, len(cls_dict), args.speed_letter_box)

        # warmup sekali (aman kalau gagal)
        try:
            h = int(getattr(cam, "img_height", 360))
            w = int(getattr(cam, "img_width", 640))
            dummy = np.zeros((h, w, 3), dtype=np.uint8)
            _ = trt.detect(dummy, max(0.05, float(getattr(args, "speed_conf_thresh", 0.15))))
        except Exception:
            pass

    # tracker (UNCHANGED)
    classes_to_track = _parse_track_classes(args.speed_track_classes, len(cls_dict))
    tracker = _attach_tracker(args.speed_tracker, classes_to_track, args)

    # window (UNCHANGED setup)
    try:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            open_window(WINDOW_NAME, 'Mission: SPEED', cam.img_width, cam.img_height)
    except Exception:
        open_window(WINDOW_NAME, 'Mission: SPEED', cam.img_width, cam.img_height)

    # ------------------ CHANGED: block handling ------------------
    try:
        ok_block = True

        # safety: ensure manda has waypoint arrays
        lat_arr = getattr(manda, "latitude_array", None) or []
        lon_arr = getattr(manda, "longitude_array", None) or []

        # clamp indices safely
        try:
            wp_start_i = int(wp_start)
            wp_end_i   = int(wp_end)
        except Exception:
            rospy.logwarn("[SPEED] Invalid wp_start/wp_end values: %s..%s", wp_start, wp_end)
            return False

        if wp_start_i < 0:
            wp_start_i = 0
        if wp_end_i < wp_start_i:
            wp_end_i = wp_start_i

        # set initial context: use start WP (useful for WP_BRANCH_DECISION logic)
        try:
            manda.current_wp_index = int(wp_start_i)
            if wp_start_i < len(lat_arr) and wp_start_i < len(lon_arr):
                manda.target_lat = float(lat_arr[wp_start_i])
                manda.target_lon = float(lon_arr[wp_start_i])
        except Exception as e:
            rospy.logwarn("[SPEED] Failed to set initial target for block start: %s", e)

        # For SPEED we *can* call the continuous detector once for the whole block
        # because _loop_and_detect already keeps running until its mission is done
        # (it uses manda.speed_route_queue and manda.has_reached_target internally).
        try:
            ok = _loop_and_detect(
                manda, cam, trt, args.speed_conf_thresh, vis,
                tracker=tracker, classes_to_track=classes_to_track
            )
            if not ok:
                rospy.logwarn("[SPEED] _loop_and_detect returned False for block %d..%d", wp_start_i, wp_end_i)
                ok_block = False
        except Exception as e:
            rospy.logwarn("[SPEED] Error during block execution: %s", e)
            ok_block = False

        # After run, sync manda state to last WP of the block
        try:
            manda.current_wp_index = int(wp_end_i)
            if wp_end_i < len(lat_arr) and wp_end_i < len(lon_arr):
                manda.target_lat = float(lat_arr[wp_end_i])
                manda.target_lon = float(lon_arr[wp_end_i])
        except Exception:
            pass

        return bool(ok_block)

    finally:
        # cleanup window: close only if user did not request keep-window
        keep = bool(getattr(args, "speed_keep_window", False))
        if not keep:
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except Exception:
                pass
