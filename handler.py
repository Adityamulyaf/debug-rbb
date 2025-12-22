#!/usr/bin/env python3
# handler_ros1.py

'''
Fitur:
- Waypoint pull (Mission Planner) -- latitude_array, longitude_array
- Gerak ke GPS target (GlobalPositionTarget)
- Velocity command (TwistStamped) -- axis maju bisa x atau y (param ~forward_axis)
- Kunci heading (AttitudeTarget, yaw saja)
- Heading kompas, pose lokal (roll/pitch/yaw dari quaternion)
- Param get/set (MAVROS), set_mode, arm/disarm
- Ubah speed (MAV_CMD_DO_CHANGE_SPEED / 178)
- Servo (183), Relay (181)
- LiDAR opsional (/scan)
- v4l2 helpers (opsional)

ROS params:
~allow_arm        (bool,  default: False) -- boleh arm FCU
~dry_run          (bool,  default: True)  -- log-only, tidak publish/serve FCU
~dist_thr_m       (float, default: 2.0)   -- ambang waypoint tercapai (meter)
~forward_axis     (str,   default: "y")   -- "x" untuk linear.x maju, "y" untuk linear.y

Tambahan:
~report_enable            (bool,  default: False) -- publish JSON ringkas status/misi
~report_topic             (str,   default: "/asv/report")
~report_interval          (float, default: 1.0)  -- throttle report per-key (detik)
~set_speed_log_interval   (float, default: 2.0)  -- throttle log SetSpeed (detik)
~set_speed_log_delta      (float, default: 0.2)  -- minimal perubahan nilai utk log (m/s)
~perf_monitor_enable      (bool,  default: False)
~perf_monitor_interval    (float, default: 10.0) -- interval log perf (detik)
'''

import math
import shutil
import subprocess
from typing import Optional, List, Tuple

import rospy

try:
    from tf import transformations as tft
except Exception:
    tft = None

import os
import json
import time

from geographiclib.geodesic import Geodesic

from std_msgs.msg import Header, Float64
from sensor_msgs.msg import NavSatFix, LaserScan
from geometry_msgs.msg import TwistStamped, PoseStamped, Quaternion
from mavros_msgs.msg import (
    GlobalPositionTarget, WaypointList, State, AttitudeTarget, ParamValue, HomePosition
)
from mavros_msgs.srv import (
    CommandBool, SetMode, ParamSet, ParamGet, WaypointPull, WaypointPush, WaypointPushRequest, CommandLong
)

from std_msgs.msg import String


class MandaHandler(object):
    def __init__(self) -> None:
        # Params
        self.allow_arm: bool = rospy.get_param("~allow_arm", True)
        self.dry_run: bool = rospy.get_param("~dry_run", False)

        self.dist_thr_m: float = rospy.get_param("~dist_thr_m",
                                 rospy.get_param("~distance_threshold_m", 2.0))

        # Arah sumbu maju untuk geometry_msgs/Twist (x atau y)
        self.forward_axis: str = rospy.get_param("~forward_axis", "y").lower()
        if self.forward_axis not in ("x", "y"):
            rospy.logwarn("Param ~forward_axis tidak valid: %s (pakai 'y')", self.forward_axis)
            self.forward_axis = "y"

        # State
        self.armed: bool = False

        # GPS
        self.current_lat: Optional[float] = None
        self.current_lon: Optional[float] = None
        self.target_lat: Optional[float] = None
        self.target_lon: Optional[float] = None
        self.final_lat: Optional[float] = None
        self.final_lon: Optional[float] = None
        self.current_wp_index: Optional[int] = None
        self.latitude_array: List[float] = []
        self.longitude_array: List[float] = []
        self.home_lat: Optional[float] = None
        self.home_lon: Optional[float] = None

        #Position in local frame
        self.p_x: float = 0.0
        self.p_y: float = 0.0
        self.p_z: float = 0.0

        # Attitude/heading
        self.current_heading_deg: float = 0.0
        self.current_roll: float = 0.0
        self.current_pitch: float = 0.0
        self.current_yaw: float = 0.0  # rad

        # LiDAR
        self.front_distance: float = 12.0
        self.left_distance: float = 12.0
        self.right_distance: float = 12.0

        # Services
        self._init_services()

        # Publishers
        self.pub_global = rospy.Publisher('/mavros/setpoint_raw/global', GlobalPositionTarget, queue_size=10)
        self.pub_att    = rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)
        self.pub_vel    = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)

        # Subscribers
        rospy.Subscriber('/mavros/global_position/global', NavSatFix, self._cb_gps, queue_size=10)
        rospy.Subscriber('/mavros/mission/waypoints', WaypointList, self._cb_wps, queue_size=10)
        rospy.Subscriber('/mavros/state', State, self._cb_state, queue_size=10)
        rospy.Subscriber('/mavros/global_position/compass_hdg', Float64, self._cb_compass, queue_size=10)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self._cb_pose_local, queue_size=2)
        rospy.Subscriber('/mavros/home_position/home', HomePosition, self._cb_home, queue_size=10)

        # LiDAR
        try:
            rospy.Subscriber('/scan', LaserScan, self._cb_lidar, queue_size=10)
        except Exception:
            pass

        # Reporting global
        self.report_enable: bool = rospy.get_param("~report_enable", True)
        self.report_topic: str = rospy.get_param("~report_topic", "/asv/report")
        self.report_interval: float = float(rospy.get_param("~report_interval", 1.0))
        self._report_pub = None
        self._report_last_ts = {}

        if self.report_enable:
            self._report_pub = rospy.Publisher(self.report_topic, String, queue_size=10)
            rospy.on_shutdown(self._report_close)

        # Throttle untuk log SetSpeed
        self._set_speed_log_itvl = float(rospy.get_param("~set_speed_log_interval", 2.0))
        self._set_speed_log_delta = float(rospy.get_param("~set_speed_log_delta", 0.2))
        self._spd_last_log_t = 0.0
        self._spd_last_val = None
        self._spd_last_success = None

        # Perf monitor ringan (publish rate)
        self._perf_enable = bool(rospy.get_param("~perf_monitor_enable", False))
        self._perf_itvl = float(rospy.get_param("~perf_monitor_interval", 10.0))
        self._ctr_vel_pubs = 0
        self._ctr_global_pubs = 0
        self._ctr_set_speed_calls = 0
        self._perf_last_t = time.time()
        if self._perf_enable:
            self._perf_timer = rospy.Timer(rospy.Duration(self._perf_itvl), self._perf_tick)

        rospy.loginfo("MandaHandler (ROS1) ready. dry_run=%s allow_arm=%s forward_axis=%s dist_thr_m=%.2f",
                      self.dry_run, self.allow_arm, self.forward_axis, self.dist_thr_m)

    # Init Services
    def _init_services(self) -> None:
        rospy.loginfo("Waiting MAVROS services ...")
        rospy.wait_for_service('/mavros/cmd/arming')
        rospy.wait_for_service('/mavros/set_mode')
        rospy.wait_for_service('/mavros/param/set')
        rospy.wait_for_service('/mavros/param/get')
        rospy.wait_for_service('/mavros/mission/pull')
        rospy.wait_for_service('/mavros/mission/push')
        rospy.wait_for_service('/mavros/cmd/command')

        self.srv_arm       = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.srv_mode      = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        self.srv_param_set = rospy.ServiceProxy('/mavros/param/set', ParamSet)
        self.srv_param_get = rospy.ServiceProxy('/mavros/param/get', ParamGet)

        self.srv_wp_pull   = rospy.ServiceProxy('/mavros/mission/pull', WaypointPull, persistent=True)
        self.srv_wp_push   = rospy.ServiceProxy('/mavros/mission/push', WaypointPush)
        self.srv_cmd_long  = rospy.ServiceProxy('/mavros/cmd/command', CommandLong)
        rospy.loginfo("All MAVROS services available.")

    # Movement primitives
    def move(self, lat: float, lon: float, alt: float = 0.0, speed: Optional[float] = None) -> None:
        """Kirim setpoint global (lat/lon/alt)."""
        self.target_lat = float(lat)
        self.target_lon = float(lon)

        if speed is not None:
            self.set_speed(speed)

        if self.dry_run:
            rospy.loginfo("[DRY] move -- lat=%.7f lon=%.7f alt=%.1f", lat, lon, alt)
            return

        msg = GlobalPositionTarget()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = GlobalPositionTarget.FRAME_GLOBAL_REL_ALT
        msg.type_mask = int(
            GlobalPositionTarget.IGNORE_VX |
            GlobalPositionTarget.IGNORE_VY |
            GlobalPositionTarget.IGNORE_VZ |
            GlobalPositionTarget.IGNORE_AFX |
            GlobalPositionTarget.IGNORE_AFY |
            GlobalPositionTarget.IGNORE_AFZ |
            GlobalPositionTarget.IGNORE_YAW |
            GlobalPositionTarget.IGNORE_YAW_RATE
        )
        msg.latitude  = self.target_lat
        msg.longitude = self.target_lon
        msg.altitude  = float(alt)
        self.pub_global.publish(msg)
        self._ctr_global_pubs += 1

    def set_velocity(self, vel_fwd: float, yaw_rate: float) -> None:
        """
        Command velocity:
          - vel_fwd -- maju (+) / mundur (-) pada axis yang dipilih (x/y)
          - yaw_rate -- angular.z (rad/s)
        """
        if self.dry_run:
            rospy.loginfo("[DRY] set_velocity fwd=%.2f wz=%.2f (axis=%s)", vel_fwd, yaw_rate, self.forward_axis)
            return

        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        if self.forward_axis == "x":
            msg.twist.linear.x = float(vel_fwd)
        else:
            msg.twist.linear.y = float(vel_fwd)
        msg.twist.angular.z = float(yaw_rate)
        self.pub_vel.publish(msg)
        self._ctr_vel_pubs += 1

    def quaternion_from_yaw(self, yaw: float) -> Quaternion:
        """yaw (rad) -- Quaternion (roll=0, pitch=0)."""
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    def set_heading(self, yaw_target_rad: float, thrust: float = 0.0) -> None:
        """Kunci heading (yaw) via AttitudeTarget."""
        if self.dry_run:
            rospy.loginfo("[DRY] set_heading yaw=%.1f deg thrust=%.2f",
                          math.degrees(yaw_target_rad), thrust)
            return

        msg = AttitudeTarget()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()

        msg.orientation = self.quaternion_from_yaw(yaw_target_rad)
        msg.thrust = float(thrust)
        msg.type_mask = int(
            AttitudeTarget.IGNORE_ROLL_RATE |
            AttitudeTarget.IGNORE_PITCH_RATE |
            AttitudeTarget.IGNORE_YAW_RATE
        )
        self.pub_att.publish(msg)

    def check_heading(self, heading_target_deg: float, tol_deg: float = 10.0) -> bool:
        cur = (self.current_heading_deg or 0.0) % 360.0
        tgt = heading_target_deg % 360.0
        err = (tgt - cur + 360.0) % 360.0
        if err > 180.0:
            err -= 360.0
        return abs(err) <= tol_deg
    
    def has_reached_final(self) -> bool:
        d = self.distance_to_final_m()
        if d is None:
            return False
        return d < float(self.dist_thr_m)
    
    def distance_to_final_m(self) -> Optional[float]:
        if self.current_lat is None or self.current_lon is None:
            return None
        if self.final_lat is None or self.final_lon is None:
            return None
        g = Geodesic.WGS84.Inverse(self.current_lat, self.current_lon,
                                   self.final_lat, self.final_lon)
        return float(g['s12'])

    def has_reached_target(self) -> bool:
        d = self.distance_to_target_m()
        if d is None:
            return False
        return d < float(self.dist_thr_m)

    def distance_to_target_m(self) -> Optional[float]:
        if self.current_lat is None or self.current_lon is None:
            return None
        if self.target_lat is None or self.target_lon is None:
            return None
        g = Geodesic.WGS84.Inverse(self.current_lat, self.current_lon,
                                   self.target_lat, self.target_lon)
        return float(g['s12'])

    # FCU / MAVROS helpers
    def pull_waypoints(self) -> bool:
        """Tarik daftar waypoints dari FCU. Saat dry_run -- True (untuk orkestrator)."""
        if self.dry_run:
            rospy.loginfo("[DRY] pull_waypoints() → TRUE")
            return True
        try:
            res = self.srv_wp_pull()
            ok = bool(res.success)
            rospy.loginfo("WaypointPull: %s (pulled=%d)", "OK" if ok else "FAIL", res.wp_received)
            return ok
        except rospy.ServiceException as e:
            rospy.logerr("Waypoint pull failed: %s", e)
            return False

    def arm(self) -> bool:
        if not self.allow_arm:
            rospy.logwarn("arm() blocked: allow_arm=False")
            return False
        if self.dry_run:
            rospy.loginfo("[DRY] arm()")
            return True
        try:
            res = self.srv_arm(True)
            rospy.loginfo("Arm: %s", "OK" if res.success else "FAIL")
            return bool(res.success)
        except rospy.ServiceException as e:
            rospy.logerr("Arm Service failed: %s", e)
            return False

    def disarm(self) -> bool:
        if self.dry_run:
            rospy.loginfo("[DRY] disarm()")
            return True
        try:
            res = self.srv_arm(False)
            rospy.loginfo("Disarm: %s", "OK" if res.success else "FAIL")
            return bool(res.success)
        except rospy.ServiceException as e:
            rospy.logerr("Disarm Service failed: %s", e)
            return False

    def set_mode(self, mode: str) -> bool:
        if self.dry_run:
            rospy.loginfo("[DRY] set_mode(%s)", mode)
            return True
        try:
            res = self.srv_mode(custom_mode=mode)
            rospy.loginfo("SetMode(%s): %s", mode, "OK" if res.mode_sent else "FAIL")
            return bool(res.mode_sent)
        except rospy.ServiceException as e:
            rospy.logerr("SetMode Service failed: %s", e)
            return False

    def set_param(self, param_id: str, integer: Optional[int] = None, real: Optional[float] = None) -> bool:
        if self.dry_run:
            rospy.loginfo("[DRY] set_param(%s) int=%s real=%s", param_id, str(integer), str(real))
            return True
        try:
            pv = ParamValue()
            pv.integer = int(integer) if integer is not None else 0
            pv.real = float(real) if real is not None else 0.0
            res = self.srv_param_set(param_id=param_id, value=pv)
            rospy.loginfo("SetParam(%s): %s", param_id, "OK" if res.success else "FAIL")
            return bool(res.success)
        except rospy.ServiceException as e:
            rospy.logerr("SetParam Service failed: %s", e)
            return False

    def get_param(self, param_id: str):
        try:
            res = self.srv_param_get(param_id=param_id)
            rospy.loginfo("get_param(%s) -> int:%d real:%.6f", param_id, res.value.integer, res.value.real)
            return res.value
        except rospy.ServiceException as e:
            rospy.logerr("GetParam Service failed: %s", e)
            return None

    def set_speed(self, speed_m_s: float) -> bool:
        """Ubah speed target (MAV_CMD_DO_CHANGE_SPEED / 178)."""
        self._ctr_set_speed_calls += 1

        if self.dry_run:
            now = time.time()
            if (now - self._spd_last_log_t) >= self._set_speed_log_itvl or \
               (self._spd_last_val is None) or \
               (abs(speed_m_s - (self._spd_last_val or 0.0)) >= self._set_speed_log_delta):
                rospy.loginfo("[DRY] set_speed(%.3f m/s)", speed_m_s)
                self._spd_last_log_t = now
                self._spd_last_val = speed_m_s
            return True  

        try:
            res = self.srv_cmd_long(
                broadcast=True,
                command=178,
                param1=1.0,   # speed type: ground speed
                param2=float(speed_m_s),
                param3=0.0, param4=0.0, param5=0.0, param6=0.0, param7=0.0
            )

            now = time.time()
            success = bool(res.success)
            should_log = (
                (now - self._spd_last_log_t) >= self._set_speed_log_itvl or
                (self._spd_last_val is None) or
                (abs(speed_m_s - (self._spd_last_val or 0.0)) >= self._set_speed_log_delta) or
                (self._spd_last_success is None) or
                (success != self._spd_last_success)
            )
            if should_log:
                rospy.loginfo("SetSpeed(%.2f): %s", speed_m_s, "OK" if success else "FAIL")
                self._spd_last_log_t = now
                self._spd_last_val = speed_m_s
                self._spd_last_success = success
            return success
        except rospy.ServiceException as e:
            rospy.logerr("SetSpeed failed: %s", e)
            return False

    # Aktuator (servo & relay)
    def move_servo(self, pin_servo: int, pwm: float) -> bool:
        """MAV_CMD_DO_SET_SERVO (183). pwm ~ 1000..2000"""
        if self.dry_run:
            rospy.loginfo("[DRY] move_servo pin=%d pwm=%.1f", pin_servo, pwm)
            return True
        try:
            res = self.srv_cmd_long(
                broadcast=False,
                command=183, confirmation=0,
                param1=float(pin_servo), param2=float(pwm),
                param3=0.0, param4=0.0, param5=0.0, param6=0.0, param7=0.0
            )
            rospy.loginfo("SetServo(pin=%d,pwm=%.1f): %s", pin_servo, pwm, "OK" if res.success else "FAIL")
            return bool(res.success)
        except rospy.ServiceException as e:
            rospy.logerr("SetServo failed: %s", e)
            return False

    def set_relay(self, pin_relay: int, value: bool) -> bool:
        """MAV_CMD_DO_SET_RELAY (181). value: 0/1."""
        if self.dry_run:
            rospy.loginfo("[DRY] set_relay pin=%d val=%s", pin_relay, str(value))
            return True
        try:
            res = self.srv_cmd_long(
                broadcast=False,
                command=181, confirmation=0,
                param1=float(pin_relay),
                param2=1.0 if value else 0.0,
                param3=0.0, param4=0.0, param5=0.0, param6=0.0, param7=0.0
            )
            rospy.loginfo("SetRelay(pin=%d,val=%s): %s", pin_relay, str(value), "OK" if res.success else "FAIL")
            return bool(res.success)
        except rospy.ServiceException as e:
            rospy.logerr("SetRelay failed: %s", e)
            return False


    def _report_close(self):
        pass

    def report(self, task: str, counts: dict = None, extra: dict = None, throttle_key: str = None):
        """
        Kirim report generik.
        - task: nama misi, mis. "pole", "avoid"
        - counts: dict ringkas, mis. {"green":2,"red":1}
        - extra: field tambahan (wp_index, note, dll.)
        - throttle_key: kunci throttle; default=task
        """
        if not self.report_enable:
            return

        key = throttle_key or task
        now = rospy.Time.now().to_sec() if rospy.core.is_initialized() else time.time()
        last = self._report_last_ts.get(key, 0.0)
        if (now - last) < float(self.report_interval):
            return
        self._report_last_ts[key] = now

        payload = {
            "task": task,
            "timestamp": now,
            "lat": self.current_lat,
            "lon": self.current_lon,
            "counts": counts or {},
        }
        if extra:
            payload.update(extra)

        js = json.dumps(payload)

        # publish topic
        if self._report_pub is not None:
            try:
                self._report_pub.publish(String(data=js))
            except Exception:
                rospy.logdebug("Report publish failed")

    # perf monitor tick
    def _perf_tick(self, event):
        now = time.time()
        dt = max(1e-3, now - self._perf_last_t)
        vel_rate = self._ctr_vel_pubs / dt
        glob_rate = self._ctr_global_pubs / dt
        spd_rate = self._ctr_set_speed_calls / dt
        rospy.loginfo("[PERF] pubs/s vel=%.2f global=%.2f set_speed=%.2f (dt=%.1fs)",
                      vel_rate, glob_rate, spd_rate, dt)
        # reset counters
        self._ctr_vel_pubs = 0
        self._ctr_global_pubs = 0
        self._ctr_set_speed_calls = 0
        self._perf_last_t = now

    # Callbacks
    def _cb_gps(self, msg: NavSatFix) -> None:
        self.current_lat = float(msg.latitude)
        self.current_lon = float(msg.longitude)

    def _cb_wps(self, msg: WaypointList) -> None:
        self.latitude_array  = [wp.x_lat for wp in msg.waypoints]
        self.longitude_array = [wp.y_long for wp in msg.waypoints]
        rospy.loginfo("Waypoints synced: %d", len(self.latitude_array))

    def _cb_state(self, msg: State) -> None:
        if self.armed and not msg.armed:
            rospy.logwarn("Vehicle disarmed by FCU.")
        self.armed = bool(msg.armed)

    def _cb_compass(self, msg: Float64) -> None:
        """Callback heading kompas dari /mavros/global_position/compass_hdg (derajat)."""
        try:
            self.current_heading_deg = float(msg.data)
        except Exception:
            self.current_heading_deg = 0.0

    def _cb_pose_local(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        p = msg.pose.position

        # kalau tf.transformations tersedia, pakai itu
        if tft is not None:
            try:
                # tf euler_from_quaternion butuh list [x,y,z,w]
                roll, pitch, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
            except Exception:
                # kalau tf error, fallback ke implementasi manual
                roll, pitch, yaw = self._euler_from_quaternion(q)
        else:
            # kalau tf tidak tersedia, langsung pakai implementasi manual
            roll, pitch, yaw = self._euler_from_quaternion(q)

        self.current_roll  = roll
        self.current_pitch = pitch
        self.current_yaw   = yaw

        self.p_x, self.p_y, self.p_z = p.x, p.y, p.z

    def _cb_lidar(self, msg: LaserScan) -> None:
        try:
            n = len(msg.ranges)
            if n > 0:
                self.front_distance = msg.ranges[n // 2]
                if n >= 20:
                    self.left_distance  = min(msg.ranges[:20])
                    self.right_distance = min(msg.ranges[-20:])
        except Exception:
            pass

    def _cb_home(self, msg: HomePosition) -> None:
        """Callback untuk menyimpan home position (origin ENU)."""
        new_lat = float(msg.geo.latitude)
        new_lon = float(msg.geo.longitude)
        
        # Simpan hanya jika belum ada, atau jika berubah (jarang terjadi)
        if self.home_lat is None or self.home_lon is None:
            self.home_lat = new_lat
            self.home_lon = new_lon
            rospy.loginfo(f"Home position (origin) TERSIMPAN: Lat={new_lat}, Lon={new_lon}")

    # Math utils
    def _euler_from_quaternion(self, q: Quaternion) -> Tuple[float, float, float]:
        """Quaternion -> euler roll, pitch, yaw (rad)."""
        x, y, z, w = q.x, q.y, q.z, q.w
        # roll (x)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        # pitch (y)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        # yaw (z)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def get_azimuth(self, p1: Tuple[float, float], p2: Tuple[float, float],
                    cond: int = 2, ref_heading_deg: float = 0.0) -> float:
        """
        Hitung azimuth (derajat) dari p1 -> p2.
        cond=1 -- (azi - ref_heading_deg)
        cond=2 -- azi di [0..360)
        """
        geo = Geodesic.WGS84
        rs = geo.Inverse(p1[0], p1[1], p2[0], p2[1])
        azi = float(rs['azi1'])
        if cond == 1:
            return azi - ref_heading_deg
        return azi + 360.0 if azi < 0.0 else azi

    # v4l2 helpers
    def set_v4l2(self, control: str, value: int) -> None:
        """Set satu kontrol v4l2-ctl (brightness/contrast/saturation/...)."""
        try:
            if shutil.which("v4l2-ctl") is None:
                rospy.logwarn("v4l2-ctl tidak ditemukan di PATH.")
                return
            cmd = f"v4l2-ctl --set-ctrl={control}={value}"
            subprocess.run(cmd, shell=True, check=False)
            rospy.loginfo("v4l2 set %s=%d", control, value)
        except Exception as e:
            rospy.logwarn("v4l2 failed: %s", e)


def main():
    rospy.init_node('manda_handler', anonymous=False)
    node = MandaHandler()
    rospy.spin()


if __name__ == '__main__':
    main()
