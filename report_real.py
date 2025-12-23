#!/usr/bin/env python3
"""
RoboBoat 2026 Protocol Buffers Reporting Module
Implements RoboCommand communications protocol (TCP, Port 50000)

Requirements:
    pip install protobuf grpcio-tools --break-system-packages

Before using:
    python3 -m grpc_tools.protoc --proto_path=proto --python_out=. proto/report.proto
"""

import socket
import struct
import time
import threading
from datetime import datetime

# Import Google protobuf timestamp
try:
    from google.protobuf.timestamp_pb2 import Timestamp
    TIMESTAMP_AVAILABLE = True
except ImportError:
    TIMESTAMP_AVAILABLE = False
    print("[REPORTER] WARNING: google.protobuf not available")

# Import generated protobuf classes
try:
    from robocommand.roboboat.v1 import report_pb2
    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False
    print("[REPORTER] ERROR: Protobuf schema not compiled!")
    print("[REPORTER] Run this command to compile:")
    print("[REPORTER] python3 -m grpc_tools.protoc --proto_path=proto --python_out=. proto/report.proto")


# ROBOCOMMAND REPORTER

class RoboCommandReporter:
    """
    RoboCommand Protocol Buffers Reporter
    Connects to RoboCommand server via TCP and sends binary protobuf messages
    """
    
    # Network configuration
    ROBOCOMMAND_IP = "10.10.10.1"
    ROBOCOMMAND_PORT = 50000
    
    def __init__(self, team_id, vehicle_id, enable=True):
        """
        Initialize reporter
        
        Args:
            team_id: Team identifier ("TEAMXXX")
            vehicle_id: Vehicle identifier (e.g., "S1")
            enable: Enable/disable reporting
        """
        # Check if protobuf is available
        if not PROTOBUF_AVAILABLE:
            print("[REPORTER] Protobuf not compiled, reporting disabled")
            enable = False
        
        if not TIMESTAMP_AVAILABLE:
            print("[REPORTER] Timestamp not available, reporting disabled")
            enable = False
        
        self.team_id = team_id
        self.vehicle_id = vehicle_id
        self.enabled = enable
        self.seq = 0
        self.sock = None
        self.connected = False
        self.lock = threading.Lock()
        
        # Heartbeat timer (1 Hz required)
        self.heartbeat_timer = None
        self.current_task = report_pb2.TASK_NONE if PROTOBUF_AVAILABLE else 1
        self.robot_state = report_pb2.STATE_AUTO if PROTOBUF_AVAILABLE else 3
        
        if self.enabled:
            self._connect()
    
    def _connect(self):
        """Connect to RoboCommand server"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0)
            self.sock.connect((self.ROBOCOMMAND_IP, self.ROBOCOMMAND_PORT))
            self.connected = True
            print(f"[REPORTER] Connected to RoboCommand at {self.ROBOCOMMAND_IP}:{self.ROBOCOMMAND_PORT}")
        except Exception as e:
            print(f"[REPORTER] Failed to connect to RoboCommand: {e}")
            print(f"[REPORTER] Make sure you're on competition network (10.10.10.0/24)")
            self.connected = False
            self.enabled = False
    
    def _send_message(self, message_bytes):
        """
        Send protobuf message to RoboCommand
        Format: [4-byte length (big-endian)] + [protobuf message]
        
        Args:
            message_bytes: Serialized protobuf message
        
        Returns:
            bool: True if sent successfully
        """
        if not self.enabled or not self.connected:
            return False
        
        try:
            with self.lock:
                # Send message length (4 bytes, big-endian)
                msg_len = len(message_bytes)
                length_bytes = struct.pack('>I', msg_len)
                
                # Send length + message
                self.sock.sendall(length_bytes + message_bytes)
                self.seq += 1
                return True
        except Exception as e:
            print(f"[REPORTER] ❌ Error sending message: {e}")
            self.connected = False
            return False
    
    def _create_timestamp(self):
        """
        Create protobuf Timestamp for current time
        
        Returns:
            Timestamp: Protobuf timestamp
        """
        timestamp = Timestamp()
        timestamp.GetCurrentTime()
        return timestamp
    
    def set_task(self, task_type):
        """
        Update current task
        
        Args:
            task_type: TaskType enum value
        """
        self.current_task = task_type
        print(f"[REPORTER] Task set to: {task_type}")
    
    def set_state(self, robot_state):
        """
        Update robot state
        
        Args:
            robot_state: RobotState enum value
        """
        self.robot_state = robot_state
        print(f"[REPORTER] State set to: {robot_state}")
    

    # HEARTBEAT (1 Hz required)
    
    def start_heartbeat(self, manda):
        """
        Start heartbeat timer (1 Hz as required by competition)
        
        Args:
            manda: MandaHandler instance for getting position/heading/speed
        """
        def _heartbeat_callback():
            if self.enabled and self.connected:
                self.send_heartbeat(
                    position=(manda.latitude, manda.longitude),
                    spd_mps=manda.speed,
                    heading_deg=manda.heading
                )
            
            # Reschedule for next heartbeat
            if self.enabled:
                self.heartbeat_timer = threading.Timer(1.0, _heartbeat_callback)
                self.heartbeat_timer.daemon = True
                self.heartbeat_timer.start()
        
        # Start first heartbeat
        _heartbeat_callback()
        print("[REPORTER] Heartbeat timer started (1 Hz)")
    
    def send_heartbeat(self, position, spd_mps, heading_deg):
        """
        Send heartbeat message (REAL PROTOBUF)
        
        Args:
            position: (lat, lon) tuple
            spd_mps: Speed in m/s
            heading_deg: Heading in degrees (North=0, East=90)
        """
        if not self.enabled:
            return
        
        # Create Report message
        report = report_pb2.Report()
        report.team_id = self.team_id
        report.vehicle_id = self.vehicle_id
        report.seq = self.seq
        report.sent_at.CopyFrom(self._create_timestamp())
        
        # Create Heartbeat message
        heartbeat = report_pb2.Heartbeat()
        heartbeat.state = self.robot_state
        heartbeat.position.latitude = float(position[0])
        heartbeat.position.longitude = float(position[1])
        heartbeat.spd_mps = float(spd_mps)
        heartbeat.heading_deg = float(heading_deg)
        heartbeat.current_task = self.current_task
        
        # Set oneof field
        report.heartbeat.CopyFrom(heartbeat)
        
        # Serialize to binary protobuf
        msg_bytes = report.SerializeToString()
        
        # Send message
        if self._send_message(msg_bytes):
            print(f"[REPORTER] Heartbeat sent (seq={self.seq}, pos={position}, spd={spd_mps:.2f} m/s)")
    
    # ========================================================================
    # TASK-SPECIFIC MESSAGES
    # ========================================================================
    
    def send_gate_pass(self, gate_type, position):
        """
        Send GatePass message (Tasks 1, 3) - REAL PROTOBUF
        
        Args:
            gate_type: GateType enum (GATE_ENTRY, GATE_EXIT, GATE_SPEED_START, GATE_SPEED_END)
            position: (lat, lon) tuple
        """
        if not self.enabled:
            return
        
        # Create Report message
        report = report_pb2.Report()
        report.team_id = self.team_id
        report.vehicle_id = self.vehicle_id
        report.seq = self.seq
        report.sent_at.CopyFrom(self._create_timestamp())
        
        # Create GatePass message
        gate_pass = report_pb2.GatePass()
        gate_pass.type = gate_type
        gate_pass.position.latitude = float(position[0])
        gate_pass.position.longitude = float(position[1])
        
        # Set oneof field
        report.gate_pass.CopyFrom(gate_pass)
        
        # Serialize to binary protobuf
        msg_bytes = report.SerializeToString()
        
        # Send message
        if self._send_message(msg_bytes):
            gate_type_name = report_pb2.GateType.Name(gate_type)
            print(f"[REPORTER] GatePass sent: {gate_type_name} at ({position[0]:.6f}, {position[1]:.6f})")
    
    def send_object_detected(self, object_type, color, position, object_id, task_context):
        """
        Send ObjectDetected message (Tasks 2, 3, 4) - REAL PROTOBUF
        
        Args:
            object_type: ObjectType enum (OBJECT_BUOY, OBJECT_LIGHT_BEACON, etc.)
            color: Color enum (COLOR_GREEN, COLOR_RED, COLOR_BLACK, COLOR_YELLOW)
            position: (lat, lon) tuple
            object_id: Unique object ID (team-scoped)
            task_context: TaskType enum
        """
        if not self.enabled:
            return
        
        # Create Report message
        report = report_pb2.Report()
        report.team_id = self.team_id
        report.vehicle_id = self.vehicle_id
        report.seq = self.seq
        report.sent_at.CopyFrom(self._create_timestamp())
        
        # Create ObjectDetected message
        obj_detected = report_pb2.ObjectDetected()
        obj_detected.object_type = object_type
        obj_detected.color = color
        obj_detected.position.latitude = float(position[0])
        obj_detected.position.longitude = float(position[1])
        obj_detected.object_id = int(object_id)
        obj_detected.task_context = task_context
        
        # Set oneof field
        report.object_detected.CopyFrom(obj_detected)
        
        # Serialize to binary protobuf
        msg_bytes = report.SerializeToString()
        
        # Send message
        if self._send_message(msg_bytes):
            obj_type_name = report_pb2.ObjectType.Name(object_type)
            color_name = report_pb2.Color.Name(color)
            task_name = report_pb2.TaskType.Name(task_context)
            print(f"[REPORTER] ObjectDetected sent: {obj_type_name} {color_name} "
                  f"(ID={object_id}) for {task_name} at ({position[0]:.6f}, {position[1]:.6f})")
    
    def send_object_delivery(self, vessel_color, position, delivery_type):
        """
        Send ObjectDelivery message (Task 4) - REAL PROTOBUF
        
        Args:
            vessel_color: Color enum of target vessel
            position: (lat, lon) tuple
            delivery_type: DeliveryType enum (DELIVERY_WATER, DELIVERY_BALL)
        """
        if not self.enabled:
            return
        
        # Create Report message
        report = report_pb2.Report()
        report.team_id = self.team_id
        report.vehicle_id = self.vehicle_id
        report.seq = self.seq
        report.sent_at.CopyFrom(self._create_timestamp())
        
        # Create ObjectDelivery message
        obj_delivery = report_pb2.ObjectDelivery()
        obj_delivery.vessel_color = vessel_color
        obj_delivery.position.latitude = float(position[0])
        obj_delivery.position.longitude = float(position[1])
        obj_delivery.delivery_type = delivery_type
        
        # Set oneof field
        report.object_delivery.CopyFrom(obj_delivery)
        
        # Serialize to binary protobuf
        msg_bytes = report.SerializeToString()
        
        # Send message
        if self._send_message(msg_bytes):
            color_name = report_pb2.Color.Name(vessel_color)
            delivery_name = report_pb2.DeliveryType.Name(delivery_type)
            print(f"[REPORTER] ObjectDelivery sent: {delivery_name} to {color_name} vessel")
    
    def send_docking(self, dock, slip):
        """
        Send Docking message (Task 5) - REAL PROTOBUF
        
        Args:
            dock: Dock identifier ('N' or 'S')
            slip: Slip number ('1', '2', or '3')
        """
        if not self.enabled:
            return
        
        # Create Report message
        report = report_pb2.Report()
        report.team_id = self.team_id
        report.vehicle_id = self.vehicle_id
        report.seq = self.seq
        report.sent_at.CopyFrom(self._create_timestamp())
        
        # Create Docking message
        docking = report_pb2.Docking()
        docking.dock = str(dock)
        docking.slip = str(slip)
        
        # Set oneof field
        report.docking.CopyFrom(docking)
        
        # Serialize to binary protobuf
        msg_bytes = report.SerializeToString()
        
        # Send message
        if self._send_message(msg_bytes):
            print(f"[REPORTER] ⚓ Docking sent: Dock {dock}, Slip {slip}")
    
    def send_sound_signal(self, signal_type, frequency_hz, assigned_task):
        """
        Send SoundSignal message (Task 6) - REAL PROTOBUF
        
        Args:
            signal_type: SignalType enum (SIGNAL_ONE_BLAST, SIGNAL_TWO_BLAST)
            frequency_hz: Frequency in Hz (600, 800, or 1000)
            assigned_task: TaskType enum assigned by signal
        """
        if not self.enabled:
            return
        
        # Create Report message
        report = report_pb2.Report()
        report.team_id = self.team_id
        report.vehicle_id = self.vehicle_id
        report.seq = self.seq
        report.sent_at.CopyFrom(self._create_timestamp())
        
        # Create SoundSignal message
        sound_signal = report_pb2.SoundSignal()
        sound_signal.signal_type = signal_type
        sound_signal.frequency_hz = int(frequency_hz)
        sound_signal.assigned_task = assigned_task
        
        # Set oneof field
        report.sound_signal.CopyFrom(sound_signal)
        
        # Serialize to binary protobuf
        msg_bytes = report.SerializeToString()
        
        # Send message
        if self._send_message(msg_bytes):
            signal_name = report_pb2.SignalType.Name(signal_type)
            task_name = report_pb2.TaskType.Name(assigned_task)
            print(f"[REPORTER] SoundSignal sent: {signal_name} @ {frequency_hz} Hz → {task_name}")
    
    def close(self):
        """Close connection and stop heartbeat"""
        print("[REPORTER] Shutting down...")
        
        # Stop heartbeat timer
        if self.heartbeat_timer:
            self.heartbeat_timer.cancel()
        
        # Close socket
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        
        self.connected = False
        print("[REPORTER] Connection closed")


# ============================================================================
# HELPER FUNCTIONS (for backward compatibility)
# ============================================================================

def color_string_to_enum(color_str):
    """
    Convert color string to Color enum
    
    Args:
        color_str: Color string (e.g., "yellow", "kuning", "red")
    
    Returns:
        Color enum value
    """
    if not PROTOBUF_AVAILABLE:
        return 0
    
    color_map = {
        'yellow': report_pb2.COLOR_YELLOW,
        'black': report_pb2.COLOR_BLACK,
        'red': report_pb2.COLOR_RED,
        'green': report_pb2.COLOR_GREEN,
    }
    
    return color_map.get(color_str.lower(), report_pb2.COLOR_UNKNOWN)



# MAIN (for testing)

if __name__ == '__main__':
    import rospy
    
    # Test reporter
    rospy.init_node('reporter_test', anonymous=True)
    
    reporter = RoboCommandReporter(
        team_id="TEAMXXX",
        vehicle_id="S1",
        enable=True
    )
    
    if reporter.connected:
        print("\n[TEST] Sending test messages...\n")
        
        # Test heartbeat
        reporter.send_heartbeat(
            position=(28.1235, -82.3456),
            spd_mps=1.5,
            heading_deg=45.0
        )
        
        # Test gate pass
        reporter.send_gate_pass(
            gate_type=report_pb2.GATE_ENTRY,
            position=(28.1235, -82.3456)
        )
        
        # Test object detected
        reporter.send_object_detected(
            object_type=report_pb2.OBJECT_BUOY,
            color=report_pb2.COLOR_GREEN,
            position=(28.1235, -82.3456),
            object_id=1,
            task_context=report_pb2.TASK_NAV_CHANNEL
        )
        
        print("\n[TEST] All messages sent successfully!")
    else:
        print("\n[TEST] Failed to connect to RoboCommand")
    
    reporter.close()