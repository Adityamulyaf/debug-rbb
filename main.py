#!/usr/bin/env python3
# main.py

import sys
import os
import time
import argparse
import subprocess
import signal
import atexit
import threading

import rospy
import cv2

# CUDA - simpan referensi untuk kontrol cleanup
_cuda_context = None
_HAS_CUDA = False
try:
    import pycuda.driver as cuda
    cuda.init()
    _cuda_context = cuda.Device(0).make_context()
    _HAS_CUDA = True
except Exception:
    _cuda_context = None
    _HAS_CUDA = False

# Handler ROS1
from handler import MandaHandler

# Misi
from pole import register as pole_register, run as pole_run
from avoid import register as avoid_register, run as avoid_run
from speed import register as speed_register, run as speed_run

# Kamera & TRT
from utils.camera import add_camera_args, Camera
from utils.yolo_with_plugins import TrtYOLO

from utils.model_registry import get_cls_dict_by_model


# Global reference untuk shutdown handler
_manda_handler = None
_camera = None
_lidar_proc = None
_engines = {}  # Simpan referensi engine untuk cleanup
_shutdown_lock = threading.Lock()
_shutdown_done = False


# Registri Misi
MISSIONS_REGISTRY = [
    {"name": "pole",  "register": pole_register,  "run": pole_run,  "prefix": "pole"},
    {"name": "avoid", "register": avoid_register, "run": avoid_run, "prefix": "avoid"},
    {"name": "speed", "register": speed_register, "run": speed_run, "prefix": "speed"},
]


def _cleanup_engines():
    """
    Cleanup TensorRT engines sebelum CUDA context di-destroy.
    Harus dipanggil SEBELUM cuda context pop/detach.
    """
    global _engines
    
    if not _engines:
        return
    
    try:
        for name, engine in list(_engines.items()):
            if engine is not None:
                try:
                    # Destroy TRT engine explicitly
                    if hasattr(engine, 'cuda_ctx') and engine.cuda_ctx is not None:
                        engine.cuda_ctx = None
                    if hasattr(engine, 'context') and engine.context is not None:
                        del engine.context
                    if hasattr(engine, 'engine') and engine.engine is not None:
                        del engine.engine
                    if hasattr(engine, 'trt_logger'):
                        del engine.trt_logger
                except Exception as e:
                    pass  # Ignore errors during cleanup
            _engines[name] = None
        _engines.clear()
    except Exception:
        pass


def _cleanup_cuda_context():
    """
    Pop dan destroy CUDA context dengan aman.
    """
    global _cuda_context
    
    if _cuda_context is None:
        return
        
    try:
        _cuda_context.pop()
        _cuda_context.detach()
        _cuda_context = None
    except Exception:
        pass


def shutdown_handler():
    """
    Handler untuk cleanup saat program shutdown.
    Reset parameter, set mode MANUAL, dan lepas resource.
    PENTING: Urutan cleanup harus benar untuk menghindari segfault.
    """
    global _manda_handler, _camera, _lidar_proc, _shutdown_done
    
    # Prevent multiple shutdown calls
    with _shutdown_lock:
        if _shutdown_done:
            return
        _shutdown_done = True
    
    # Gunakan print karena rospy mungkin sudah shutdown
    print("[SHUTDOWN] Handler triggered. Cleaning up...")
    
    # 1. Stop robot movement first (safety)
    try:
        if _manda_handler is not None:
            try:
                _manda_handler.set_velocity(0, 0)
            except Exception:
                pass
    except Exception:
        pass
    
    # 2. Reset MandaHandler parameters
    try:
        if _manda_handler is not None:
            # Reset parameter ke default aman
            try:
                _manda_handler.set_param("MOT_THR_MAX", 35)
                _manda_handler.set_param("WP_SPEED", 1)
                _manda_handler.set_param("CRUISE_SPEED", 1)
            except Exception as e:
                print(f"[SHUTDOWN] Failed to reset params: {e}")
            
            # Set mode MANUAL untuk keamanan
            try:
                _manda_handler.set_mode("MANUAL")
            except Exception as e:
                print(f"[SHUTDOWN] Failed to set MANUAL mode: {e}")
            
            # Reset aktuator jika ada
            try:
                if hasattr(_manda_handler, 'relay_trig'):
                    _manda_handler.relay_trig(0, True)
                if hasattr(_manda_handler, 'move_cam'):
                    _manda_handler.move_cam(10, 1500)
                if hasattr(_manda_handler, 'move_wblast'):
                    _manda_handler.move_wblast(6, 1800)
                    _manda_handler.move_wblast(7, 1500)
            except Exception as e:
                print(f"[SHUTDOWN] Failed to reset actuators: {e}")
                
    except Exception as e:
        print(f"[SHUTDOWN] Error during manda_handler cleanup: {e}")
    
    # 3. Release kamera SEBELUM TRT cleanup
    try:
        if _camera is not None:
            _camera.release()
            print("[SHUTDOWN] Camera released.")
    except Exception as e:
        print(f"[SHUTDOWN] Failed to release camera: {e}")
    
    # 4. Tutup semua window OpenCV
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    
    # 5. Cleanup TensorRT engines SEBELUM CUDA context
    print("[SHUTDOWN] Cleaning up TensorRT engines...")
    try:
        _cleanup_engines()
    except Exception as e:
        print(f"[SHUTDOWN] TRT cleanup error: {e}")
    
    # 6. Cleanup CUDA context terakhir
    print("[SHUTDOWN] Cleaning up CUDA context...")
    try:
        _cleanup_cuda_context()
    except Exception as e:
        print(f"[SHUTDOWN] CUDA cleanup error: {e}")
    
    # 7. Stop LiDAR process jika ada
    try:
        _stop_lidar(_lidar_proc)
    except Exception:
        pass
    
    print("[SHUTDOWN] Cleanup completed.")


def signal_handler(signum, frame):
    """
    Handler untuk signal SIGINT (Ctrl+C) dan SIGTERM.
    """
    sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
    print(f"\n[SIGNAL] {sig_name} ({signum}) received. Initiating shutdown.")
    
    shutdown_handler()
    
    # Exit tanpa memanggil rospy.signal_shutdown untuk menghindari race condition
    os._exit(0)


def build_parser():
    parser = argparse.ArgumentParser("Mission Orchestrator (ROS1 + Waypoint)")

    # Argumen kamera (img_width, img_height, device, dll.)
    parser = add_camera_args(parser)

    # Argumen masing-masing misi (dipasang via register())
    for m in MISSIONS_REGISTRY:
        try:
            m["register"](parser)
        except Exception as e:
            pass  # Will log after rospy init

    # Mapping misi via CLI (contoh: 1:pole,2:avoid,3:pole)
    parser.add_argument(
        "--mission-plan",
        default="3-9:avoid",
        help='Mapping "WP_INDEX:mission", contoh: "1:pole,2:avoid,3:pole"'
    ) #Rentang

    # Kecepatan default untuk navigasi waypoint ketika tidak ada misi khusus
    parser.add_argument(
        "--nav-speed",
        type=float,
        default=2.0,
        help="Kecepatan (m/s) saat hanya menuju waypoint tanpa misi khusus."
    )

    # Opsi: mulai dari WP index berapa (default 1 -- lewati home di index 0)
    parser.add_argument(
        "--start-wp-index",
        type=int,
        default=1,
        help="Mulai dari waypoint index ini (biasanya 1 untuk melewati home di 0)."
    )

    # Auto-start LiDAR (opsional)
    parser.add_argument(
        "--auto-lidar", action="store_true",
        help="Jika diaktifkan, main.py akan auto-roslaunch RPLIDAR untuk misi yang meminta LiDAR."
    )
    parser.add_argument(
        "--lidar-launch-pkg", type=str, default="rplidar_ros",
        help="Nama package roslaunch untuk LiDAR."
    )
    parser.add_argument(
        "--lidar-launch-file", type=str, default="view_rplidar_a2m12.launch",
        help="Nama file launch LiDAR."
    )
    parser.add_argument(
        "--lidar-port", type=str, default="/dev/ttyUSB0",
        help="Serial port RPLIDAR (contoh: /dev/ttyUSB0)."
    )
    parser.add_argument(
        "--lidar-baud", type=int, default=115200,
        help="Baudrate RPLIDAR (A2/A2M12=115200; A3/S1=256000)."
    )

    return parser


def parse_plan(plan_str):
    """
    Parse string mission-plan:
      - "1:pole,2:avoid" -> {1: "pole", 2: "avoid"}
      - "1-2:pole, 3-8:avoid"      -> {2: "avoid", 3: "avoid", 4: "avoid"}
    """
    plan = {}
    if not plan_str:
        return plan

    toks = [t.strip() for t in plan_str.split(",") if t.strip()]
    for tok in toks:
        if ":" not in tok:
            raise SystemExit(f'Format mission-plan salah di token: "{tok}". Contoh benar: "1:pole,2:avoid" atau "2-4:avoid"')
        left, name = tok.split(":", 1)
        left = left.strip()
        name = name.strip()
        if not name:
            raise SystemExit(f'Format mission-plan salah (nama misi kosong) di token: "{tok}"')

        def put(idx):
            try:
                idx = int(idx)
            except ValueError:
                raise SystemExit(f'Format mission-plan salah (index bukan integer) di token: "{tok}"')
            if idx < 0:
                raise SystemExit(f'Index WP tidak boleh negatif di token: "{tok}"')
            plan[idx] = name

        if "-" in left:
            a, b = left.split("-", 1)
            a, b = a.strip(), b.strip()
            try:
                a_i, b_i = int(a), int(b)
            except ValueError:
                raise SystemExit(f'Range WP harus integer di token: "{tok}"')
            lo, hi = (a_i, b_i) if a_i <= b_i else (b_i, a_i)
            for i in range(lo, hi + 1):
                put(i)
        else:
            put(left)
    return plan


def _mission_lookup(name):
    """
    Ambil entry misi dari registry berdasarkan nama (case-sensitive).
    """
    for m in MISSIONS_REGISTRY:
        if m["name"] == name:
            return m
    return None


def _ensure_engine_for(prefix, args):
    """
    Verifikasi & buat TrtYOLO untuk sebuah misi berdasarkan prefix argumen misi.
    """
    global _engines
    
    try:
        model_name = getattr(args, f"{prefix}_model")
        letter_box = getattr(args, f"{prefix}_letter_box")
        engine_dir = getattr(args, f"{prefix}_engine_dir")
        override_num = getattr(args, f"{prefix}_category_num")
    except AttributeError:
        return None

    eng_path = os.path.join(engine_dir, f"{model_name}.trt")
    if not os.path.isfile(eng_path):
        raise SystemExit(f"[{prefix}] ERROR: TensorRT engine tidak ditemukan: {eng_path}")

    try:
        cls_dict = get_cls_dict_by_model(model_name, override_num=override_num)
        num_cls = len(cls_dict)
    except Exception as e:
        rospy.logwarn("[%s] Tidak bisa memuat cls_dict dari registry: %s. Fallback ke override_num.", prefix, e)
        num_cls = int(override_num)

    if not _HAS_CUDA:
        rospy.logwarn("[%s] CUDA context tidak tersedia; skip preload engine (lanjut tanpa TRT).", prefix)
        return None

    # Buat engine
    trt = TrtYOLO(model_name, num_cls, letter_box)
    
    # Simpan reference untuk cleanup
    _engines[prefix] = trt

    # Warmup sederhana (dummy 640x360)
    try:
        import numpy as np
        dummy = np.zeros((360, 640, 3), dtype=np.uint8)
        _ = trt.detect(dummy, 0.10)
        rospy.loginfo("[%s] TRT warmup sekali (dummy 640x360) OK.", prefix)
    except Exception as e:
        rospy.logwarn("[%s] Warmup gagal (lanjut): %s", prefix, e)

    return trt


def preload_engines(args, mission_plan):
    """
    Preload engine (TrtYOLO) hanya untuk misi yang muncul di mission_plan.
    Return dict: {mission_name: engine_or_None}
    """
    used_missions = set(mission_plan.values())
    engines = {}

    for name in used_missions:
        entry = _mission_lookup(name)
        if entry is None:
            rospy.logwarn("Misi '%s' tidak terdaftar di registry.", name)
            engines[name] = None
            continue

        prefix = entry.get("prefix", name)
        try:
            engines[name] = _ensure_engine_for(prefix, args)
            if engines[name] is not None:
                rospy.loginfo("[%s] Engine preloaded.", name)
            else:
                rospy.loginfo("[%s] Tanpa engine khusus (OK).", name)
        except SystemExit:
            raise

        except Exception as e:
            rospy.logwarn("[%s] Gagal preload engine: %s (lanjut tanpa engine)", name, e)
            engines[name] = None

    return engines


def wait_for_waypoints(manda):
    """
    Tarik waypoints sampai benar-benar ada data (≥ 2 titik).
    Mengembalikan (lat_array, lon_array).
    """
    while not rospy.is_shutdown():
        got = manda.pull_waypoints()
        lat = getattr(manda, "latitude_array", []) or []
        lon = getattr(manda, "longitude_array", []) or []

        if got and len(lat) >= 2 and len(lon) >= 2:
            return lat, lon

        rospy.loginfo("Waiting for waypoints...")
        time.sleep(0.5)

    return [], []


def _preflight_summary(args, cam, manda, mission_plan, engines):
    """
    Buat ringkasan preflight: status kamera, waypoints, engine, dan rencana LiDAR.
    """
    cam_ok = cam.isOpened()
    cam_desc = f"{getattr(cam, 'img_width', '?')}x{getattr(cam, 'img_height', '?')}" if cam_ok else "CLOSED"

    got = manda.pull_waypoints()
    lat = getattr(manda, "latitude_array", []) or []
    lon = getattr(manda, "longitude_array", []) or []
    wp_ok = got and len(lat) >= 2 and len(lon) >= 2
    wp_desc = f"{len(lat)} points" if wp_ok else "Not ready"

    used = sorted(set(mission_plan.values()))
    eng_lines = []
    for name in used:
        eng_lines.append(f"{name}: {'LOADED' if engines.get(name) is not None else 'None'}")

    wants_lidar = bool(getattr(args, "auto_lidar", False) and _missions_need_lidar(args, mission_plan))
    lidar_line = f"auto-start={'ON' if wants_lidar else 'OFF'} port={args.lidar_port} baud={args.lidar_baud}"

    lines = [
        f"CUDA available  : {_HAS_CUDA}",
        f"Camera          : {cam_desc}",
        f"Waypoints       : {wp_desc}",
        "Engines         : " + (", ".join(eng_lines) if eng_lines else "None"),
        f"LIDAR           : {lidar_line}",
    ]
    return "\n".join(lines)


def _confirm_start(summary_text):
    print("=== PREFLIGHT CHECK ===")
    print(summary_text)
    ans = input("Start missions now? [y/N]: ").strip().lower()
    if ans != "y":
        raise SystemExit("Aborted by user.")


def _missions_need_lidar(args, mission_plan):
    """
    Tentukan apakah rencana misi membutuhkan LiDAR.
    """
    used = set(mission_plan.values())
    need = False
    if "pole" in used and bool(getattr(args, "pole_use_lidar", False)):
        need = True
    return need


def _start_lidar_if_needed(args):
    """
    Jalankan roslaunch LiDAR jika diminta dan diperlukan.
    Return Popen process atau None.
    """
    if not bool(getattr(args, "auto_lidar", False)):
        return None

    if not _missions_need_lidar(args, mission_plan=parse_plan(args.mission_plan)):
        rospy.loginfo("Auto LiDAR: tidak diperlukan oleh rencana misi (skip).")
        return None

    pkg  = args.lidar_launch_pkg
    file = args.lidar_launch_file
    port = args.lidar_port
    baud = str(args.lidar_baud)

    cmd = [
        "roslaunch", pkg, file,
        f"serial_port:={port}",
        f"serial_baudrate:={baud}",
    ]
    rospy.loginfo("Auto LiDAR: starting `%s` ...", " ".join(cmd))
    try:
        proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
        time.sleep(2.0)
        rospy.loginfo("Auto LiDAR: launched (PID=%s).", str(proc.pid))
        return proc
    except Exception as e:
        rospy.logwarn("Auto LiDAR gagal start: %s", e)
        return None


def _stop_lidar(proc):
    if proc is None:
        return
    try:
        print(f"[SHUTDOWN] Auto LiDAR: stopping (PID={proc.pid})...")
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except Exception:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass


def main():
    global _manda_handler, _camera, _lidar_proc, _engines
    
    # Register signal handlers SEBELUM apapun
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = build_parser()
    args = parser.parse_args()

    # Parse mission plan
    mission_plan = parse_plan(args.mission_plan)

    # Inisialisasi ROS1
    rospy.init_node("mission_orchestrator", anonymous=False)
    rospy.loginfo("Mission Orchestrator (ROS1) starting ... CUDA=%s", str(_HAS_CUDA))
    
    # Register atexit handler (backup)
    atexit.register(shutdown_handler)
    
    # JANGAN register rospy.on_shutdown - bisa menyebabkan race condition
    # rospy.on_shutdown(shutdown_handler)

    # Kamera tunggal untuk semua misi
    cam = Camera(args)
    _camera = cam
    
    if not cam.isOpened():
        rospy.logerr("ERROR: gagal membuka kamera.")
        shutdown_handler()
        return 1
    rospy.loginfo("Camera opened: %dx%d", cam.img_width, cam.img_height)

    # Node handler (ROS1)
    manda = MandaHandler()
    _manda_handler = manda
    manda.current_wp_index = -1

    # Preload engines (termasuk warmup)
    try:
        engines = preload_engines(args, mission_plan)
        # Update global reference
        for name, eng in engines.items():
            if eng is not None:
                _engines[name] = eng
    except SystemExit as e:
        rospy.logerr(str(e))
        shutdown_handler()
        return 1

    # Preflight summary + konfirmasi
    summary = _preflight_summary(args, cam, manda, mission_plan, engines)
    try:
        _confirm_start(summary)
    except SystemExit as e:
        print(str(e))
        shutdown_handler()
        return 0

    # Auto-start LiDAR bila diminta & diperlukan
    lidar_proc = _start_lidar_if_needed(args)
    _lidar_proc = lidar_proc

    # Auto set mode + arm
    mode = "GUIDED"
    if not manda.dry_run and getattr(manda, "allow_arm", True):
        if not manda.set_mode(mode):
            rospy.logwarn("Set mode %s gagal, coba GUIDED_NOGPS", mode)
            if not manda.set_mode("GUIDED_NOGPS"):
                rospy.logwarn("Set mode GUIDED_NOGPS juga gagal")
        rospy.sleep(0.5)
        if not manda.arm():
            rospy.logwarn("Arm gagal")

    try:
        # Tunggu waypoint benar-benar ada (blocking)
        lat, lon = wait_for_waypoints(manda)
        if rospy.is_shutdown():
            shutdown_handler()
            return 0

        rospy.loginfo("Waypoints received: %d", len(lat))
        if hasattr(manda, "report"):
            try:
                manda.report(task="orchestrator",
                             extra={"event": "wps_received", "count": int(len(lat))},
                             throttle_key="orchestrator_wps_received")
            except Exception:
                pass

        # Mulai dari WP index yang diminta (default 1 -- lewati home)
        i = max(0, int(args.start_wp_index))

        while not rospy.is_shutdown():
            if i >= len(lat):
                rospy.loginfo("All waypoints reached.")
                if hasattr(manda, "report"):
                    try:
                        manda.report(task="orchestrator",
                                     extra={"event": "all_wp_done"},
                                     throttle_key="orchestrator_all_done")
                    except Exception:
                        pass
                break

            # Set target
            manda.target_lat = float(lat[i])
            manda.target_lon = float(lon[i])
            manda.current_wp_index = int(i)

            # Apakah ada misi khusus di WP i?
            mission_name = mission_plan.get(i, None)
            if mission_name:
                entry = _mission_lookup(mission_name)
                if entry is None:
                    rospy.logerr("[WP %d] Mission '%s' tidak terdaftar.", i, mission_name)
                    break

                run = entry["run"]
                rospy.loginfo("[WP %d] Mission: %s", i, mission_name)

                if hasattr(manda, "report"):
                    try:
                        manda.report(task="orchestrator",
                                     extra={"event": "mission_start", "wp": int(i), "name": mission_name},
                                     throttle_key=f"orchestrator_mstart_{i}")
                    except Exception:
                        pass

                engine = engines.get(mission_name)

                j = i
                while j < len(lat) and mission_plan.get(j) == mission_name:
                    j += 1
                start_idx, end_idx = i, j - 1
                rospy.loginfo("[WP %d..%d] Block Mission: %s", start_idx, end_idx, mission_name)

                lat_mission = lat[start_idx:end_idx + 1]
                lon_mission = lon[start_idx:end_idx + 1]
                
                try:
                    ok = run(manda, lat_mission, lon_mission, args, cam, engine)
                except Exception as e:
                    rospy.logwarn("[WP %d..%d] run %s error: %s", start_idx, end_idx, mission_name, e)
                    ok = False

                if hasattr(manda, "report"):
                    try:
                        manda.report(task="orchestrator",
                                     extra={"event": "mission_block_end",
                                            "wp_start": int(start_idx), "wp_end": int(end_idx),
                                            "name": mission_name, "ok": bool(ok)},
                                     throttle_key=f"orchestrator_block_end_{mission_name}_{start_idx}_{end_idx}")
                    except Exception:
                        pass
                
                i = int(end_idx + 1)

            else:
                try:
                    manda.move(lat[i], lon[i], speed=float(args.nav_speed))
                except Exception as e:
                    rospy.logwarn("[WP %d] move() gagal: %s", i, e)

                if manda.has_reached_target():
                    rospy.loginfo("Waypoint %d reached.", i)
                    if hasattr(manda, "report"):
                        try:
                            manda.report(task="orchestrator",
                                         extra={"event": "wp_reached", "wp": int(i)},
                                         throttle_key=f"orchestrator_wpreached_{i}")
                        except Exception:
                            pass
                    i += 1

        # Normal exit
        shutdown_handler()
        return 0

    except KeyboardInterrupt:
        print("\n[MAIN] KeyboardInterrupt caught in main loop.")
        shutdown_handler()
        return 0
    except SystemExit as e:
        print(f"[MAIN] SystemExit: {e}")
        shutdown_handler()
        return 1
    except Exception as e:
        print(f"[MAIN] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        shutdown_handler()
        return 1


if __name__ == "__main__":
    ret = main()
    # Gunakan os._exit untuk menghindari cleanup atexit yang bisa cause segfault
    # karena shutdown_handler sudah dipanggil secara eksplisit
    os._exit(ret)

