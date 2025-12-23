#!/usr/bin/env python3

"""
MandaHandler - RoboBoat 2026 Mandakini X Handler
Manages waypoint navigation, reporting, and vehicle control
"""

import rospy
import math
import threading
from std_msgs.msg import String, Float32
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Twist
from mavros_msgs.msg import State, WaypointReached
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL

# Import RoboCommand reporter
try:
    from New.report_test import (
        RoboCommandReporter, TaskType, RobotState,
        GateType, ObjectType, Color
    )
    REPORTER_AVAILABLE = True
except ImportError:
    REPORTER_AVAILABLE = False
    print("[HANDLER] WARNING: roboboat_reporter not available")


# CONSTANTS (Top of file for easy tuning)

# Navigation thresholds
DEFAULT_DIST_THRESHOLD_M = 2.0 # Default waypoint reached distance
DEFAULT_SPEED_M_S = 1.5 # Default cruising speed

# Reporting configuration
TEAM_ID = "BUV-UNS"
VEHICLE_ID = "S1"

# Throttle intervals (seconds)
REPORT_THROTTLE_INTERVAL = 0.5


# MANDA HANDLER CLASS

class MandaHandler:
    """
    Main handler for Mandakini X
    Manages navigation, reporting, and mission coordination
    """
    
    def __init__(self):
        """Initialize handler"""
        rospy.loginfo("[HANDLER] Initializing MandaHandler...")

        # State Variables
        self.latitude = 0.0
        self.longitude = 0.0
        self.altitude = 0.0
        self.heading = 0.0 # Degrees (North=0, East=90)
        self.speed = 0.0 # m/s
        
        self.target_lat = None
        self.target_lon = None
        
        # Mission planner waypoints (loaded from mission file)
        self.latitude_array = []
        self.longitude_array = []
        self.current_wp_index = 0
        
        # Distance threshold for waypoint reached
        self.dist_thr_m = DEFAULT_DIST_THRESHOLD_M

        # Threading Locks
        self._gps_lock = threading.Lock()
        self._speed_lock = threading.Lock()
        self._report_lock = threading.Lock()

        # ROS Subscribers
        self.gps_sub = rospy.Subscriber(
            '/mavros/global_position/global',
            NavSatFix,
            self._gps_callback,
            queue_size=1
        )
        
        self.heading_sub = rospy.Subscriber(
            '/mavros/global_position/compass_hdg',
            Float32,
            self._heading_callback,
            queue_size=1
        )
        
        self.state_sub = rospy.Subscriber(
            '/mavros/state',
            State,
            self._state_callback,
            queue_size=1
        )

        # ROS Publishers
        self.cmd_vel_pub = rospy.Publisher(
            '/mavros/setpoint_velocity/cmd_vel_unstamped',
            Twist,
            queue_size=1
        )

        # ROS Services
        rospy.wait_for_service('/mavros/cmd/arming')
        rospy.wait_for_service('/mavros/set_mode')
        
        self.arming_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        

        # RoboCommand Reporter
        self.report_enable = REPORTER_AVAILABLE
        if self.report_enable:
            self.reporter = RoboCommandReporter(
                team_id=TEAM_ID,
                vehicle_id=VEHICLE_ID,
                enable=True
            )
            self.reporter.start_heartbeat(self)
            rospy.loginfo("[HANDLER] RoboCommand reporter initialized")
        else:
            rospy.logwarn("[HANDLER] RoboCommand reporting DISABLED")
        

        # Mission-specific queues (with locks)
        self.speed_route_queue = []
        
        # Report throttling
        self._last_report_time = {}
        
        rospy.loginfo("[HANDLER] MandaHandler initialized successfully")
    

    # ROS CALLBACKS
    
    def _gps_callback(self, msg):
        """GPS position callback"""
        with self._gps_lock:
            self.latitude = msg.latitude
            self.longitude = msg.longitude
            self.altitude = msg.altitude
    
    def _heading_callback(self, msg):
        """Compass heading callback"""
        self.heading = msg.data
    
    def _state_callback(self, msg):
        """Vehicle state callback"""
        if self.report_enable:
            # Update reporter state
            if msg.armed:
                self.reporter.set_state(RobotState.STATE_AUTO)
            else:
                self.reporter.set_state(RobotState.STATE_KILLED)

    # NAVIGATION METHODS
    
    def distance_to_target_m(self):
        """
        Calculate distance to target in meters
        
        Returns:
            float: Distance in meters, or None if no target set
        """
        if self.target_lat is None or self.target_lon is None:
            return None
        
        with self._gps_lock:
            lat1 = self.latitude
            lon1 = self.longitude
        
        lat2 = self.target_lat
        lon2 = self.target_lon
        
        # Haversine formula
        R = 6371000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        
        a = (math.sin(dphi / 2) ** 2 +
             math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def has_reached_target(self, threshold_override=None):
        """
        Check if target waypoint reached
        
        Args:
            threshold_override: Custom threshold in meters (optional)
        
        Returns:
            bool: True if target reached
        """
        d = self.distance_to_target_m()
        if d is None:
            return False
        
        thr = threshold_override if threshold_override is not None else self.dist_thr_m
        return d < float(thr)
    
    def bearing_to_target_deg(self):
        """
        Calculate bearing to target in degrees
        
        Returns:
            float: Bearing in degrees (North=0, East=90), or None if no target
        """
        if self.target_lat is None or self.target_lon is None:
            return None
        
        with self._gps_lock:
            lat1 = math.radians(self.latitude)
            lon1 = math.radians(self.longitude)
        
        lat2 = math.radians(self.target_lat)
        lon2 = math.radians(self.target_lon)
        
        dlon = lon2 - lon1
        
        x = math.sin(dlon) * math.cos(lat2)
        y = (math.cos(lat1) * math.sin(lat2) -
             math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        
        bearing_rad = math.atan2(x, y)
        bearing_deg = math.degrees(bearing_rad)
        
        # Normalize to 0-360
        return (bearing_deg + 360) % 360
    
    def set_speed(self, speed_m_s):
        """
        Set vehicle speed
        
        Args:
            speed_m_s: Speed in m/s
        
        Returns:
            bool: True if command sent successfully
        """
        with self._speed_lock:
            try:
                cmd = Twist()
                cmd.linear.x = float(speed_m_s)
                cmd.linear.y = 0.0
                cmd.linear.z = 0.0
                cmd.angular.z = 0.0
                
                self.cmd_vel_pub.publish(cmd)
                self.speed = speed_m_s
                
                rospy.logdebug_throttle(2.0, f"[HANDLER] Speed set to {speed_m_s:.2f} m/s")
                return True
            except Exception as e:
                rospy.logerr(f"[HANDLER] Failed to set speed: {e}")
                return False
    
    def set_mode(self, mode):
        """
        Set vehicle mode
        
        Args:
            mode: Mode string (e.g., "GUIDED", "AUTO")
        
        Returns:
            bool: True if mode set successfully
        """
        try:
            resp = self.set_mode_client(custom_mode=mode)
            if resp.mode_sent:
                rospy.loginfo(f"[HANDLER] Mode set to {mode}")
                return True
            else:
                rospy.logwarn(f"[HANDLER] Failed to set mode to {mode}")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"[HANDLER] Service call failed: {e}")
            return False
    
    def arm(self):
        """Arm vehicle"""
        try:
            resp = self.arming_client(value=True)
            if resp.success:
                rospy.loginfo("[HANDLER] Vehicle ARMED")
                return True
            else:
                rospy.logwarn("[HANDLER] Failed to ARM vehicle")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"[HANDLER] Arming service call failed: {e}")
            return False
    
    def disarm(self):
        """Disarm vehicle"""
        try:
            resp = self.arming_client(value=False)
            if resp.success:
                rospy.loginfo("[HANDLER] Vehicle DISARMED")
                return True
            else:
                rospy.logwarn("[HANDLER] Failed to DISARM vehicle")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"[HANDLER] Disarming service call failed: {e}")
            return False

    # REPORTING METHODS
    
    def _should_throttle_report(self, throttle_key):
        """
        Check if report should be throttled
        
        Args:
            throttle_key: Unique key for throttling (or None to disable)
        
        Returns:
            bool: True if report should be skipped (throttled)
        """
        if throttle_key is None:
            return False
        
        now = rospy.Time.now().to_sec()
        
        with self._report_lock:
            last_time = self._last_report_time.get(throttle_key, 0.0)
            
            if now - last_time < REPORT_THROTTLE_INTERVAL:
                return True
            
            self._last_report_time[throttle_key] = now
            return False
    
    def report_gate_pass(self, gate_type, throttle_key="gate_pass"):
        """
        Report gate passage (Tasks 1, 3)
        
        Args:
            gate_type: GateType enum value
            throttle_key: Throttling key (None to disable throttle)
        """
        if not self.report_enable:
            return
        
        if self._should_throttle_report(throttle_key):
            return
        
        with self._gps_lock:
            position = (self.latitude, self.longitude)
        
        self.reporter.send_gate_pass(gate_type, position)
    
    def report_object_detected(self, object_type, color, object_id, task_context, throttle_key=None):
        """
        Report object detection (Tasks 2, 3, 4)
        
        Args:
            object_type: ObjectType enum value
            color: Color enum value
            object_id: Unique object ID
            task_context: TaskType enum value
            throttle_key: Throttling key (None to disable throttle)
        """
        if not self.report_enable:
            return
        
        if self._should_throttle_report(throttle_key):
            return
        
        with self._gps_lock:
            position = (self.latitude, self.longitude)
        
        self.reporter.send_object_detected(
            object_type, color, position, object_id, task_context
        )
    
    def set_current_task(self, task_type):
        """
        Update current task for reporter
        
        Args:
            task_type: TaskType enum value
        """
        if self.report_enable:
            self.reporter.set_task(task_type)

    # LEGACY REPORT METHOD
    
    def report(self, task, counts, extra=None, throttle_key="legacy"):
        """
        Legacy report method for backward compatibility
        
        Args:
            task: Task name string
            counts: Dict of detected object counts
            extra: Extra metadata dict
            throttle_key: Throttling key
        """
        if not self.report_enable:
            return
        
        if self._should_throttle_report(throttle_key):
            return
        
        # Log report
        rospy.logdebug_throttle(1.0, 
            f"[HANDLER] Report: task={task}, counts={counts}, extra={extra}")

    # CLEANUP
    
    def shutdown(self):
        """Shutdown handler and close connections"""
        rospy.loginfo("[HANDLER] Shutting down MandaHandler...")
        
        # Stop heartbeat and close reporter
        if self.report_enable:
            self.reporter.close()
        
        rospy.loginfo("[HANDLER] Shutdown complete")


# MAIN (for testing)

if __name__ == '__main__':
    try:
        rospy.init_node('manda_handler_test', anonymous=True)
        handler = MandaHandler()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
