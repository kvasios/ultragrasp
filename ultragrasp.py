"""
Unified UltraGrasp dashboard with gesture bars, zeroed right-hand 6D pose, and
optional robot teleoperation.

Controls:
  - Space: toggle zeroed pose tracking preview
  - r: toggle robot control
  - h: home robot to configured joint target (does not require live mode)
  - Robot / Home buttons: same as r / h
  - WSG gripper (if enabled): follows hand only while robot live mode is on, when robot_control is enabled
  - q: quit
  - Close window / Ctrl+C: exit
"""

from __future__ import annotations

import argparse
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
import sys
import os
import math
from multiprocessing.managers import SharedMemoryManager

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UMI_DIR = os.path.join(ROOT_DIR, "dependencies", "universal_manipulation_interface")
if UMI_DIR not in sys.path:
    sys.path.append(UMI_DIR)

from umi.common.precise_sleep import precise_wait
from umi.real_world.franka_interpolation_controller import FrankaInterpolationController
from umi.real_world.wsg_controller import WSGController

import leap
import numpy as np
import scipy.spatial.transform as st
import yaml
from leap.enums import TrackingMode


def is_right_hand(hand) -> bool:
    if hasattr(hand, "is_right"):
        try:
            return bool(hand.is_right)
        except Exception:
            pass
    hand_type = getattr(hand, "type", None)
    return hand_type is not None and str(hand_type) in {"HandType.Right", "eLeapHandType_Right"}


# Leap frame to plot / command frame rotation (+90 deg around X):
#   x' = x, y' = -z, z' = y
_PLOT_FRAME_ROT = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

_IDENTITY_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


@dataclass
class WSGControlConfig:
    enabled: bool = False
    hostname: str = "172.16.0.4"
    port: int = 1000
    frequency: float = 30.0
    home_to_open: bool = True
    max_speed: float = 400.0
    max_pos: float = 110.0
    gesture: str = "grasp"  # "pinch" or "grasp"
    filter_time_constant: float = 0.05
    command_latency: float = 0.0
    receive_latency: float = 0.0
    use_meters: bool = False
    direct_strength: bool = False

@dataclass
class RobotControlConfig:
    enabled: bool = True
    robot_ip: str = "192.168.122.100"
    robot_port: int = 4242
    command_rate_hz: float = 100.0
    controller_frequency: float = 200.0
    command_latency: float = 0.0
    receive_latency: float = 0.0
    home_on_connect: bool = False
    home_joint_positions: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.0, -math.pi / 4.0, 0.0, -3.0 * math.pi / 4.0, 0.0, math.pi / 2.0, math.pi / 4.0],
            dtype=np.float64,
        )
    )
    home_move_duration: float = 5.0
    position_sensitivity_m_per_mm: float = 0.001
    axis_scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float64))
    tcp_relative_rpy_deg: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    tcp_relative_R: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    Kx_scale: float = 1.0
    Kxd_scale: np.ndarray = field(
        default_factory=lambda: np.array([2.0, 1.5, 2.0, 1.0, 1.0, 1.0], dtype=np.float64)
    )
    soft_real_time: bool = False
    orientation_control: bool = False
    debug: bool = True


@dataclass
class RobotControlStatus:
    enabled: bool
    active: bool
    waiting_for_reference: bool
    has_command: bool
    has_display_pose: bool
    debug: bool
    orientation_control: bool
    robot_bridge_connected: bool
    robot_bridge_error: Optional[str]
    last_command_pos_m: np.ndarray
    rel_command_pos_mm: np.ndarray
    rel_command_R: np.ndarray
    display_pose_pos_mm: np.ndarray
    display_pose_R: np.ndarray


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros(3, dtype=np.float64)
    return v / n


def quat_xyzw_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        return np.eye(3, dtype=np.float64)

    x, y, z, w = (q / n).tolist()
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))

    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        return _IDENTITY_QUAT_XYZW.copy()
    return q / n


def euler_rpy_deg_to_rotmat(rpy_deg: np.ndarray) -> np.ndarray:
    roll_deg, pitch_deg, yaw_deg = np.asarray(rpy_deg, dtype=np.float64).reshape(3)
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=np.float64,
    )
    Ry = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=np.float64,
    )
    Rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return Rz @ Ry @ Rx


def rotmat_to_euler_rpy_deg(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=np.float64).reshape(3, 3)
    sy = float(np.sqrt(m[0, 0] * m[0, 0] + m[1, 0] * m[1, 0]))
    singular = sy < 1e-9

    if not singular:
        roll = np.arctan2(m[2, 1], m[2, 2])
        pitch = np.arctan2(-m[2, 0], sy)
        yaw = np.arctan2(m[1, 0], m[0, 0])
    else:
        roll = np.arctan2(-m[1, 2], m[1, 1])
        pitch = np.arctan2(-m[2, 0], sy)
        yaw = 0.0

    return np.rad2deg(np.array([roll, pitch, yaw], dtype=np.float64))


def rotvec_to_rotmat(rotvec: np.ndarray) -> np.ndarray:
    return st.Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64).reshape(3)).as_matrix()


def rotmat_to_rotvec(R: np.ndarray) -> np.ndarray:
    return st.Rotation.from_matrix(np.asarray(R, dtype=np.float64).reshape(3, 3)).as_rotvec()


def basis_from_normal_direction(normal: np.ndarray, direction: np.ndarray) -> np.ndarray:
    y = _normalize(np.asarray(normal, dtype=np.float64))
    z_raw = _normalize(np.asarray(direction, dtype=np.float64))
    x = _normalize(np.cross(y, z_raw))
    if np.linalg.norm(x) < 1e-9:
        return np.eye(3, dtype=np.float64)
    z = _normalize(np.cross(x, y))
    if np.linalg.norm(z) < 1e-9:
        return np.eye(3, dtype=np.float64)
    return np.column_stack((x, y, z))


def hand_position_mm(hand) -> np.ndarray:
    if hasattr(hand, "palm") and hasattr(hand.palm, "position"):
        return np.array([hand.palm.position.x, hand.palm.position.y, hand.palm.position.z], dtype=np.float64)
    if hasattr(hand, "palm_position"):
        return np.array([hand.palm_position.x, hand.palm_position.y, hand.palm_position.z], dtype=np.float64)
    raise AttributeError("No palm position on hand object")


def hand_basis_matrix(hand) -> np.ndarray:
    if hasattr(hand, "basis"):
        b = hand.basis
        if hasattr(b, "x_basis") and hasattr(b, "y_basis") and hasattr(b, "z_basis"):
            x = np.array([b.x_basis.x, b.x_basis.y, b.x_basis.z], dtype=np.float64)
            y = np.array([b.y_basis.x, b.y_basis.y, b.y_basis.z], dtype=np.float64)
            z = np.array([b.z_basis.x, b.z_basis.y, b.z_basis.z], dtype=np.float64)
            return np.column_stack((_normalize(x), _normalize(y), _normalize(z)))

    if hasattr(hand, "palm") and hasattr(hand.palm, "orientation"):
        q = hand.palm.orientation
        return quat_xyzw_to_rotmat(np.array([q.x, q.y, q.z, q.w], dtype=np.float64))

    if hasattr(hand, "palm"):
        n = np.array([hand.palm.normal.x, hand.palm.normal.y, hand.palm.normal.z], dtype=np.float64)
        d = np.array([hand.palm.direction.x, hand.palm.direction.y, hand.palm.direction.z], dtype=np.float64)
        return basis_from_normal_direction(n, d)
    if hasattr(hand, "palm_normal") and hasattr(hand, "direction"):
        n = np.array([hand.palm_normal.x, hand.palm_normal.y, hand.palm_normal.z], dtype=np.float64)
        d = np.array([hand.direction.x, hand.direction.y, hand.direction.z], dtype=np.float64)
        return basis_from_normal_direction(n, d)

    raise AttributeError("No orientation source available on hand object")


def to_plot_frame(pos: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    C = _PLOT_FRAME_ROT
    return C @ pos, C @ R @ C.T


def _parse_axis_scale(value) -> np.ndarray:
    if value is None:
        return np.ones(3, dtype=np.float64)
    if np.isscalar(value):
        return np.full(3, float(value), dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError("robot_control.axis_scale must be a scalar or a 3-element list")
    return arr


def _parse_rpy_deg(value) -> np.ndarray:
    if value is None:
        return np.zeros(3, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError("robot_control.tcp_relative_rpy_deg must be a 3-element list")
    return arr


def _parse_gain_scale(value, *, length: int, name: str) -> np.ndarray:
    if value is None:
        return np.ones(length, dtype=np.float64)
    if np.isscalar(value):
        return np.full(length, float(value), dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != length:
        raise ValueError(f"{name} must be a scalar or a {length}-element list")
    return arr


def _parse_joint_positions(value, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 7:
        raise ValueError(f"{name} must be a 7-element list")
    return arr


def load_configs(config_path: Path) -> Tuple[RobotControlConfig, WSGControlConfig]:
    robot_cfg = RobotControlConfig()
    wsg_cfg = WSGControlConfig()
    if not config_path.exists():
        print(f"Config not found at {config_path}; using defaults.")
        return robot_cfg, wsg_cfg

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    robot_data = data.get("robot_control", data)
    robot_cfg.enabled = bool(robot_data.get("enabled", robot_cfg.enabled))
    robot_cfg.robot_ip = str(robot_data.get("robot_ip", robot_cfg.robot_ip))
    robot_cfg.robot_port = int(robot_data.get("robot_port", robot_cfg.robot_port))
    robot_cfg.command_rate_hz = max(1.0, float(robot_data.get("command_rate_hz", robot_cfg.command_rate_hz)))
    robot_cfg.controller_frequency = max(
        1.0, float(robot_data.get("controller_frequency", robot_cfg.controller_frequency))
    )
    robot_cfg.command_latency = float(robot_data.get("command_latency", robot_cfg.command_latency))
    robot_cfg.receive_latency = float(robot_data.get("receive_latency", robot_cfg.receive_latency))
    robot_cfg.home_on_connect = bool(robot_data.get("home_on_connect", robot_cfg.home_on_connect))
    robot_cfg.home_joint_positions = _parse_joint_positions(
        robot_data.get("home_joint_positions", robot_cfg.home_joint_positions.tolist()),
        name="robot_control.home_joint_positions",
    )
    robot_cfg.home_move_duration = max(
        0.0, float(robot_data.get("home_move_duration", robot_cfg.home_move_duration))
    )
    robot_cfg.position_sensitivity_m_per_mm = float(
        robot_data.get("position_sensitivity_m_per_mm", robot_cfg.position_sensitivity_m_per_mm)
    )
    robot_cfg.axis_scale = _parse_axis_scale(robot_data.get("axis_scale", robot_cfg.axis_scale.tolist()))
    robot_cfg.tcp_relative_rpy_deg = _parse_rpy_deg(
        robot_data.get("tcp_relative_rpy_deg", robot_cfg.tcp_relative_rpy_deg.tolist())
    )
    robot_cfg.tcp_relative_R = euler_rpy_deg_to_rotmat(robot_cfg.tcp_relative_rpy_deg)
    robot_cfg.Kx_scale = float(robot_data.get("Kx_scale", robot_cfg.Kx_scale))
    robot_cfg.Kxd_scale = _parse_gain_scale(
        robot_data.get("Kxd_scale", robot_cfg.Kxd_scale.tolist()),
        length=6,
        name="robot_control.Kxd_scale",
    )
    robot_cfg.soft_real_time = bool(robot_data.get("soft_real_time", robot_cfg.soft_real_time))
    robot_cfg.orientation_control = bool(robot_data.get("orientation_control", robot_cfg.orientation_control))
    robot_cfg.debug = bool(robot_data.get("debug", robot_cfg.debug))

    wsg_data = data.get("wsg_control", {})
    wsg_cfg.enabled = bool(wsg_data.get("enabled", wsg_cfg.enabled))
    wsg_cfg.hostname = str(wsg_data.get("hostname", wsg_cfg.hostname))
    wsg_cfg.port = int(wsg_data.get("port", wsg_cfg.port))
    wsg_cfg.frequency = float(wsg_data.get("frequency", wsg_cfg.frequency))
    wsg_cfg.home_to_open = bool(wsg_data.get("home_to_open", wsg_cfg.home_to_open))
    wsg_cfg.max_speed = float(wsg_data.get("max_speed", wsg_cfg.max_speed))
    wsg_cfg.max_pos = float(wsg_data.get("max_pos", wsg_cfg.max_pos))
    wsg_cfg.gesture = str(wsg_data.get("gesture", wsg_cfg.gesture))
    wsg_cfg.filter_time_constant = float(wsg_data.get("filter_time_constant", wsg_cfg.filter_time_constant))
    wsg_cfg.command_latency = float(wsg_data.get("command_latency", wsg_cfg.command_latency))
    wsg_cfg.receive_latency = float(wsg_data.get("receive_latency", wsg_cfg.receive_latency))
    wsg_cfg.use_meters = bool(wsg_data.get("use_meters", wsg_cfg.use_meters))
    wsg_cfg.direct_strength = bool(wsg_data.get("direct_strength", wsg_cfg.direct_strength))

    return robot_cfg, wsg_cfg


class DashboardSnapshot:
    def __init__(self) -> None:
        self.frame_id = 0
        self.visible = False
        self.hand_label = "none"

        self.confidence = 0.0
        self.grab_strength = 0.0
        self.pinch_strength = 0.0
        self.sphere_center: Optional[Tuple[float, float, float]] = None

        self.right_visible = False
        self.pos_mm = np.zeros(3, dtype=np.float64)
        self.R = np.eye(3, dtype=np.float64)

        self.active = False
        self.pending_zero = False
        self.ref_pos_mm: Optional[np.ndarray] = None
        self.ref_R: Optional[np.ndarray] = None


class DashboardListener(leap.Listener):
    def __init__(self, robot_controller: Optional["UltraGraspController"] = None) -> None:
        self._lock = threading.Lock()
        self.snapshot = DashboardSnapshot()
        self.robot_controller = robot_controller

    def toggle_tracking(self) -> None:
        with self._lock:
            if self.snapshot.active:
                self.snapshot.active = False
                self.snapshot.pending_zero = False
                self.snapshot.ref_pos_mm = None
                self.snapshot.ref_R = None
                print("Pose preview OFF")
            else:
                self.snapshot.active = True
                self.snapshot.pending_zero = True
                print("Pose preview ON (zero on next visible right hand)")

    def on_tracking_event(self, event) -> None:
        right = None
        for h in event.hands:
            if is_right_hand(h):
                right = h
                break
        gesture_hand = right if right is not None else (event.hands[0] if len(event.hands) > 0 else None)
        right_visible = False
        right_pos_mm = np.zeros(3, dtype=np.float64)
        right_R = np.eye(3, dtype=np.float64)

        with self._lock:
            self.snapshot.frame_id = int(getattr(event, "tracking_frame_id", 0))

            if gesture_hand is None:
                self.snapshot.visible = False
                self.snapshot.hand_label = "none"
                self.snapshot.confidence = 0.0
                self.snapshot.grab_strength = 0.0
                self.snapshot.pinch_strength = 0.0
                self.snapshot.sphere_center = None
            else:
                self.snapshot.visible = True
                self.snapshot.hand_label = "right" if is_right_hand(gesture_hand) else "left/first"
                self.snapshot.confidence = float(getattr(gesture_hand, "confidence", 0.0))
                self.snapshot.grab_strength = float(getattr(gesture_hand, "grab_strength", 0.0))
                self.snapshot.pinch_strength = float(getattr(gesture_hand, "pinch_strength", 0.0))

                sc = getattr(gesture_hand, "sphere_center", None)
                if sc is not None and all(hasattr(sc, axis) for axis in ("x", "y", "z")):
                    self.snapshot.sphere_center = (float(sc.x), float(sc.y), float(sc.z))
                else:
                    self.snapshot.sphere_center = None

            if right is None:
                self.snapshot.right_visible = False
            else:
                self.snapshot.right_visible = True
                self.snapshot.pos_mm = hand_position_mm(right)
                self.snapshot.R = hand_basis_matrix(right)
                right_visible = True
                right_pos_mm = self.snapshot.pos_mm.copy()
                right_R = self.snapshot.R.copy()

        if self.robot_controller is not None:
            self.robot_controller.update_from_hand(right_visible, right_pos_mm, right_R)

    def get_snapshot(self) -> Tuple[DashboardSnapshot, bool, bool, np.ndarray, np.ndarray]:
        with self._lock:
            snap = DashboardSnapshot()
            snap.frame_id = self.snapshot.frame_id
            snap.visible = self.snapshot.visible
            snap.hand_label = self.snapshot.hand_label
            snap.confidence = self.snapshot.confidence
            snap.grab_strength = self.snapshot.grab_strength
            snap.pinch_strength = self.snapshot.pinch_strength
            snap.sphere_center = self.snapshot.sphere_center
            snap.right_visible = self.snapshot.right_visible
            snap.pos_mm = self.snapshot.pos_mm.copy()
            snap.R = self.snapshot.R.copy()

            active = self.snapshot.active
            has_pose = False
            p_rel = np.zeros(3, dtype=np.float64)
            R_rel = np.eye(3, dtype=np.float64)

            if active and self.snapshot.right_visible:
                p = self.snapshot.pos_mm.copy()
                R = self.snapshot.R.copy()
                if (
                    self.snapshot.pending_zero
                    or self.snapshot.ref_pos_mm is None
                    or self.snapshot.ref_R is None
                ):
                    self.snapshot.ref_pos_mm = p
                    self.snapshot.ref_R = R
                    self.snapshot.pending_zero = False
                    print("Zero pose captured")

                has_pose = True
                p_rel = self.snapshot.ref_R.T @ (p - self.snapshot.ref_pos_mm)
                R_rel = self.snapshot.ref_R.T @ R

            return snap, active, has_pose, p_rel, R_rel


class UltraGraspController:
    def __init__(self, config: RobotControlConfig, shm_manager: Optional[SharedMemoryManager] = None) -> None:
        self.config = config
        self._shm_manager = shm_manager
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._command_loop, daemon=True)
        self._franka: Optional[FrankaInterpolationController] = None

        self._active = False
        self._waiting_for_reference = False
        self._has_command = False
        self._hand_tracking_lost = False

        self._hand_ref_pos_mm: Optional[np.ndarray] = None
        self._hand_ref_R: Optional[np.ndarray] = None

        self._session_anchor_command_pose = np.zeros(6, dtype=np.float64)

        self._current_command_pose = np.zeros(6, dtype=np.float64)

        self._last_command_pose = np.zeros(6, dtype=np.float64)
        self._display_pose_pos_mm = np.zeros(3, dtype=np.float64)
        self._display_pose_R = np.eye(3, dtype=np.float64)

        self._debug_send_count = 0
        self._debug_last_log_time = time.time()
        self._bridge_connect_error: Optional[str] = None

        if self.config.enabled and not self.config.debug:
            if shm_manager is None:
                raise ValueError("SharedMemoryManager is required for live Franka control.")
            self._start_franka_bridge()

        self._initial_tcp_pose = self._session_anchor_command_pose.copy()

        self._thread.start()

    def _reset_hand_session(self) -> None:
        self._active = False
        self._waiting_for_reference = False
        self._hand_tracking_lost = False
        self._hand_ref_pos_mm = None
        self._hand_ref_R = None
        self._display_pose_pos_mm = np.zeros(3, dtype=np.float64)
        self._display_pose_R = np.eye(3, dtype=np.float64)

    def _start_franka_bridge(self) -> bool:
        if self.config.debug or not self.config.enabled:
            return True
        if self._franka is not None:
            if self._franka.is_alive():
                return True
            self._franka = None
        if self._shm_manager is None:
            self._bridge_connect_error = "SharedMemoryManager unavailable"
            return False

        franka: Optional[FrankaInterpolationController] = None
        try:
            franka = FrankaInterpolationController(
                shm_manager=self._shm_manager,
                robot_ip=self.config.robot_ip,
                robot_port=self.config.robot_port,
                frequency=int(round(self.config.controller_frequency)),
                Kx_scale=self.config.Kx_scale,
                Kxd_scale=self.config.Kxd_scale,
                joints_init=self.config.home_joint_positions if self.config.home_on_connect else None,
                joints_init_duration=self.config.home_move_duration if self.config.home_on_connect else None,
                soft_real_time=self.config.soft_real_time,
                verbose=False,
                receive_latency=self.config.receive_latency,
            )
            if self.config.home_on_connect:
                print(
                    "Franka bridge connected; moving to startup home joint position "
                    f"over {self.config.home_move_duration:.1f}s."
                )
            franka.start(wait=True)
            self._franka = franka
            self._bridge_connect_error = None
            current_pose = self._get_live_robot_pose()
            with self._lock:
                self._session_anchor_command_pose = current_pose.copy()
                self._current_command_pose = current_pose.copy()
                self._last_command_pose = current_pose.copy()
                self._has_command = False
            self._initial_tcp_pose = current_pose.copy()
            print(
                f"Robot bridge connected to {self.config.robot_ip}:{self.config.robot_port}."
            )
            return True
        except Exception as exc:
            self._bridge_connect_error = str(exc)
            if franka is not None:
                try:
                    franka.stop(wait=True)
                except Exception:
                    pass
            self._franka = None
            print(f"WARNING: {ROBOT_BRIDGE_README_HINT}")
            print(f"  ({type(exc).__name__}: {exc})")
            return False

    def _robot_bridge_ok(self) -> bool:
        if not self.config.enabled or self.config.debug:
            return True
        return self._franka is not None and self._franka.is_alive()

    def _handle_franka_runtime_error(self, exc: Exception) -> None:
        error = str(exc) or type(exc).__name__
        with self._lock:
            franka = self._franka
            self._franka = None
            self._reset_hand_session()
            self._has_command = False
            self._bridge_connect_error = error

        if franka is not None:
            try:
                franka.stop(wait=False)
            except Exception:
                pass

        print(f"Robot control stopped after bridge/controller failure: {error}")

    def _get_live_robot_pose(self) -> np.ndarray:
        if self._franka is None:
            return self._last_command_pose.copy()
        state = self._franka.get_state()
        return np.asarray(state["ActualTCPPose"], dtype=np.float64).reshape(6)

    def _get_session_anchor_pose(self) -> np.ndarray:
        if self._franka is not None:
            return self._get_live_robot_pose()
        return self._last_command_pose.copy()

    def toggle_active(self) -> None:
        if not self.config.enabled:
            print("Robot control disabled in config; set robot_control.enabled: true to allow commands.")
            return
        if not self._robot_bridge_ok() and not self._start_franka_bridge():
            print(f"Robot control unavailable: {ROBOT_BRIDGE_README_HINT}")
            return

        with self._lock:
            if self._active:
                self._reset_hand_session()
                print("Robot control OFF (holding last commanded target)")
            else:
                anchor_pose = self._get_session_anchor_pose()
                self._active = True
                self._waiting_for_reference = True
                self._hand_tracking_lost = False
                self._hand_ref_pos_mm = None
                self._hand_ref_R = None
                self._display_pose_pos_mm = np.zeros(3, dtype=np.float64)
                self._display_pose_R = np.eye(3, dtype=np.float64)
                self._session_anchor_command_pose = anchor_pose.copy()
                self._current_command_pose = anchor_pose.copy()
                self._last_command_pose = anchor_pose.copy()
                print("Robot control ON (waiting for visible right hand to capture reference)")

    def request_home(self) -> None:
        """Move arm to configured home joint positions. Does not require live mode."""
        if not self.config.enabled:
            print("Robot control disabled in config; cannot home.")
            return
        if not self._robot_bridge_ok() and not self._start_franka_bridge():
            print(f"Homing unavailable: {ROBOT_BRIDGE_README_HINT}")
            return

        duration = float(self.config.home_move_duration)
        target_joints = self.config.home_joint_positions.copy()
        if self._franka is not None:
            min_dur = 1.0 / float(self._franka.frequency) + 1e-6
            duration = max(duration, min_dur)

        with self._lock:
            self._reset_hand_session()
            self._has_command = False

        if self._franka is not None:
            try:
                self._franka.move_to_joint_positions(target_joints, time_to_go=duration)
                print(
                    "Homing to configured joint target "
                    f"over {duration:.1f}s (live mode off; no 'r' needed)."
                )
            except Exception as exc:
                print(f"Homing failed: {exc}")
        else:
            print(
                "[DEBUG] Homing: would move to configured home_joint_positions "
                f"over {duration:.1f}s."
            )

    def update_from_hand(self, right_visible: bool, pos_mm: np.ndarray, R: np.ndarray) -> None:
        if not self.config.enabled:
            return

        p = np.asarray(pos_mm, dtype=np.float64).reshape(3)
        hand_R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        paused_due_to_loss = False
        resumed_after_loss = False

        with self._lock:
            if not self._active:
                return

            if not right_visible:
                if self._hand_ref_pos_mm is not None and not self._waiting_for_reference and not self._hand_tracking_lost:
                    self._hand_tracking_lost = True
                    paused_due_to_loss = True
                else:
                    return
            else:
                if self._hand_tracking_lost:
                    self._hand_tracking_lost = False
                    resumed_after_loss = True

                waiting_for_reference = self._waiting_for_reference
                hand_ref_pos_mm = None if self._hand_ref_pos_mm is None else self._hand_ref_pos_mm.copy()
                hand_ref_R = None if self._hand_ref_R is None else self._hand_ref_R.copy()
                session_anchor_command_pose = self._session_anchor_command_pose.copy()

        if paused_due_to_loss:
            print("Robot control paused: right hand lost, holding last target")
            return

        if resumed_after_loss:
            print("Robot control resumed with fixed session reference")

        need_reference = waiting_for_reference or hand_ref_pos_mm is None or hand_ref_R is None
        if need_reference:
            hand_ref_pos_mm = p.copy()
            hand_ref_R = hand_R.copy()
            session_anchor_command_pose = self._get_session_anchor_pose()

        # 1. Delta in the recorded hand's reference frame (mm)
        hand_delta_mm = hand_ref_R.T @ (p - hand_ref_pos_mm)

        # 2. Display variables
        hand_rot_rel = hand_ref_R.T @ hand_R
        display_pose_pos_mm = _PLOT_FRAME_ROT @ hand_delta_mm
        display_pose_R = _PLOT_FRAME_ROT @ hand_rot_rel @ _PLOT_FRAME_ROT.T

        # 3. Convert mm to meters and apply basic Leap-to-Robot mapping.
        hand_delta_m = hand_delta_mm * self.config.position_sensitivity_m_per_mm
        hand_delta_cmd = self.config.axis_scale * (_PLOT_FRAME_ROT @ hand_delta_m)

        # 4. Apply the TCP relative rotation to align "move hand right/forward/up"
        # with the robot's base frame directions.
        anchor_pos = session_anchor_command_pose[:3]
        anchor_R = rotvec_to_rotmat(session_anchor_command_pose[3:])
        command_pos = anchor_pos + (self.config.tcp_relative_R @ hand_delta_cmd)

        if self.config.orientation_control:
            command_rot_rel = _PLOT_FRAME_ROT @ hand_rot_rel @ _PLOT_FRAME_ROT.T
            command_R = self.config.tcp_relative_R @ command_rot_rel @ self.config.tcp_relative_R.T
            command_R = command_R @ anchor_R
        else:
            command_R = anchor_R

        command_pose = np.concatenate([command_pos, rotmat_to_rotvec(command_R)])

        reference_captured = False
        with self._lock:
            if not self._active:
                return

            if need_reference:
                self._hand_ref_pos_mm = hand_ref_pos_mm.copy()
                self._hand_ref_R = hand_ref_R.copy()
                self._session_anchor_command_pose = session_anchor_command_pose.copy()
                self._waiting_for_reference = False
                reference_captured = True

            self._display_pose_pos_mm = display_pose_pos_mm
            self._display_pose_R = display_pose_R
            self._current_command_pose = command_pose
            self._last_command_pose = command_pose.copy()
            self._has_command = True

        if reference_captured:
            print("Robot control reference captured")

    def get_status(self) -> RobotControlStatus:
        with self._lock:
            active = self._active
            waiting_for_reference = self._waiting_for_reference
            has_command = self._has_command
            has_display_pose = self._hand_ref_pos_mm is not None and self._hand_ref_R is not None
            bridge_ok = (not self.config.enabled) or self.config.debug or (self._franka is not None)
            bridge_error = self._bridge_connect_error
            last_command_pose = self._last_command_pose.copy()
            session_anchor_command_pose = self._session_anchor_command_pose.copy()
            display_pose_pos_mm = self._display_pose_pos_mm.copy()
            display_pose_R = self._display_pose_R.copy()

        rel_command_pos_mm = np.zeros(3, dtype=np.float64)
        rel_command_R = np.eye(3, dtype=np.float64)
        if has_command:
            anchor_R = rotvec_to_rotmat(session_anchor_command_pose[3:])
            last_R = rotvec_to_rotmat(last_command_pose[3:])
            rel_command_pos_m = anchor_R.T @ (
                last_command_pose[:3] - session_anchor_command_pose[:3]
            )
            rel_command_pos_mm = 1000.0 * rel_command_pos_m
            rel_command_R = anchor_R.T @ last_R

        return RobotControlStatus(
            enabled=self.config.enabled,
            active=active,
            waiting_for_reference=waiting_for_reference,
            has_command=has_command,
            has_display_pose=has_display_pose,
            debug=self.config.debug,
            orientation_control=self.config.orientation_control,
            robot_bridge_connected=bridge_ok,
            robot_bridge_error=bridge_error,
            last_command_pos_m=last_command_pose[:3].copy(),
            rel_command_pos_mm=rel_command_pos_mm,
            rel_command_R=rel_command_R,
            display_pose_pos_mm=display_pose_pos_mm,
            display_pose_R=display_pose_R,
        )

    def button_label(self, status: Optional[RobotControlStatus] = None) -> str:
        if status is None:
            status = self.get_status()
        if not status.enabled:
            return "Robot Disabled"
        if status.enabled and not status.debug and not status.robot_bridge_connected:
            return "Robot: No bridge"
        if status.active and status.waiting_for_reference:
            return "Robot: Waiting"
        if status.active:
            return "Robot: Stop"
        return "Robot: Start"

    def gripper_gesture_live(self) -> bool:
        if not self.config.enabled:
            return True
        with self._lock:
            return self._active and not self._waiting_for_reference and self._has_command

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        franka = self._franka
        self._franka = None
        if franka is not None:
            try:
                franka.stop(wait=True)
            except Exception as exc:
                print(f"Robot bridge shutdown warning: {str(exc) or type(exc).__name__}")

    def _dispatch_command(self, pose: np.ndarray, target_time: float) -> None:
        if self._franka is not None:
            self._franka.schedule_waypoint(pose=pose, target_time=target_time)
        self._debug_send_count += 1
        now = time.time()
        if self.config.debug and now - self._debug_last_log_time >= 1.0:
            cmd_rpy_deg = rotmat_to_euler_rpy_deg(rotvec_to_rotmat(pose[3:]))
            orientation_mode = "tracked" if self.config.orientation_control else "locked"
            print(
                "[DEBUG] Robot command stream: "
                f"{self._debug_send_count} msgs/s "
                f"pos=({pose[0]:+.3f}, {pose[1]:+.3f}, {pose[2]:+.3f})m "
                f"rotvec=(rx:{pose[3]:+.3f}, ry:{pose[4]:+.3f}, rz:{pose[5]:+.3f}) "
                f"rpy_deg=(r:{cmd_rpy_deg[0]:+.1f}, p:{cmd_rpy_deg[1]:+.1f}, y:{cmd_rpy_deg[2]:+.1f}) "
                f"orientation_mode={orientation_mode}"
            )
            self._debug_send_count = 0
            self._debug_last_log_time = now

    def _command_loop(self) -> None:
        dt = 1.0 / self.config.command_rate_hz
        t_start = time.monotonic()
        iter_idx = 0
        while self._running:
            t_cycle_end = t_start + (iter_idx + 1) * dt
            t_sample = t_cycle_end - self.config.command_latency
            if t_sample > time.monotonic():
                precise_wait(t_sample)

            send_pose = None

            with self._lock:
                if self.config.enabled and self._active and self._has_command:
                    send_pose = self._current_command_pose.copy()

            if send_pose is not None:
                try:
                    t_command_target = t_cycle_end + dt
                    wall_clock_target = t_command_target - time.monotonic() + time.time()
                    self._dispatch_command(send_pose, target_time=wall_clock_target)
                except Exception as exc:
                    self._handle_franka_runtime_error(exc)

            precise_wait(t_cycle_end)
            iter_idx += 1


class WSGControllerManager:
    def __init__(
        self,
        config: WSGControlConfig,
        listener: DashboardListener,
        shm_manager: SharedMemoryManager,
        robot_controller: Optional["UltraGraspController"] = None,
    ) -> None:
        self.config = config
        self.listener = listener
        self.robot_controller = robot_controller
        self._running = False
        self._thread = None
        self.gripper = None

        if not self.config.enabled:
            return

        self.gripper = WSGController(
            shm_manager=shm_manager,
            hostname=config.hostname,
            port=config.port,
            frequency=config.frequency,
            home_to_open=config.home_to_open,
            move_max_speed=config.max_speed,
            receive_latency=config.receive_latency,
            use_meters=config.use_meters,
            verbose=False,
        )
        self.gripper.start()
        
        self._running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()

    def gesture_control_live(self) -> bool:
        """When robot teleop is enabled, WSG follows hand only when robot live mode is on (r) and tracking."""
        rc = self.robot_controller
        if rc is None:
            return True
        return rc.gripper_gesture_live()

    def close(self) -> None:
        if self._running:
            self._running = False
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            if self.gripper is not None:
                self.gripper.stop()

    def _control_loop(self) -> None:
        dt = 1.0 / self.config.frequency
        filter_alpha = (
            1.0 if self.config.filter_time_constant <= 0.0 
            else 1.0 - math.exp(-dt / self.config.filter_time_constant)
        )
        
        state = self.gripper.get_state()
        target_pos = float(state["gripper_position"])
        opening_signal = _clip01(target_pos / self.config.max_pos)
        filtered_opening = opening_signal
        
        t_start = time.monotonic()
        self.gripper.restart_put(time.time())
        iter_idx = 0
        
        while self._running:
            t_cycle_end = t_start + (iter_idx + 1) * dt
            t_sample = t_cycle_end - self.config.command_latency
            t_command_target = t_cycle_end + dt
            
            precise_wait(t_sample)
            if not self._running:
                break
                
            snap, _, _, _, _ = self.listener.get_snapshot()
            live = self.gesture_control_live()

            if snap.visible and live:
                strength = snap.pinch_strength if self.config.gesture == "pinch" else snap.grab_strength
                opening_signal = _clip01(strength if self.config.direct_strength else 1.0 - strength)
                filtered_opening += filter_alpha * (opening_signal - filtered_opening)

            desired_pos = self.config.max_pos * filtered_opening
            if not live:
                desired_pos = target_pos
            max_step = self.config.max_speed / self.config.frequency
            dpos = max(-max_step, min(max_step, desired_pos - target_pos))
            target_pos = max(0.0, min(self.config.max_pos, target_pos + dpos))
            
            target_timestamp = t_command_target - time.monotonic() + time.time()
            self.gripper.schedule_waypoint(target_pos, target_timestamp)
            
            precise_wait(t_cycle_end)
            iter_idx += 1

def select_gui_backend() -> None:
    import matplotlib

    current = matplotlib.get_backend().lower()
    if current not in {"agg", "template"}:
        return

    for candidate in ("QtAgg", "TkAgg"):
        try:
            matplotlib.use(candidate, force=True)
            return
        except Exception:
            continue


def _configure_dashboard_window(fig) -> None:
    """Best-effort: normal stacking (no always-on-top) on Qt backends."""
    mgr = getattr(fig.canvas, "manager", None)
    if mgr is None:
        return
    win = getattr(mgr, "window", None)
    if win is None:
        return
    try:
        try:
            from PyQt5.QtCore import Qt
        except ImportError:
            try:
                from PySide6.QtCore import Qt
            except ImportError:
                try:
                    from PyQt6.QtCore import Qt
                except ImportError:
                    return
        flags = int(win.windowFlags())
        try:
            top_hint = Qt.WindowType.WindowStaysOnTopHint
        except AttributeError:
            top_hint = Qt.WindowStaysOnTopHint
        top = int(top_hint)
        if flags & top:
            win.setWindowFlags(flags & ~top)
            win.show()
    except Exception:
        pass


def _dashboard_gui_idle(fig, interval: float) -> None:
    """Sleep while servicing the GUI; avoids plt.pause(), which often steals focus every frame on Qt."""
    fig.canvas.flush_events()
    time.sleep(interval)


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


ROBOT_BRIDGE_README_HINT = (
    "No active robot bridge. Start the Polymetis Franka interface server (see README.md) "
    "and match robot_control.robot_ip / robot_port in your config."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="UltraGrasp dashboard with optional robot teleoperation")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/ultragrasp.yaml"),
        help="Path to UltraGrasp dashboard config YAML",
    )
    args = parser.parse_args()

    robot_config, wsg_config = load_configs(args.config)

    select_gui_backend()
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    backend = plt.get_backend().lower()
    if backend in {"agg", "template"}:
        raise RuntimeError("Matplotlib is using non-GUI backend Agg. Install/use Qt or Tk backend.")

    need_shm = (robot_config.enabled and not robot_config.debug) or wsg_config.enabled
    shm_manager = None
    robot_controller = None
    wsg_manager = None
    listener = None
    connection = None

    if need_shm:
        shm_manager = SharedMemoryManager()
        shm_manager.start()

    try:
        fig = None
        robot_controller = UltraGraspController(robot_config, shm_manager=shm_manager)
        listener = DashboardListener(robot_controller=robot_controller)
        connection = leap.Connection()
        connection.add_listener(listener)

        stop_event = threading.Event()

        def on_sigint(_sig, _frm):
            stop_event.set()

        signal.signal(signal.SIGINT, on_sigint)

        plt.ion()
        fig = plt.figure(
            "Hand Dashboard + Right Pose (Space=preview, r/button=robot, h=home, q=quit, Ctrl+C=quit)",
            figsize=(11.2, 6.8),
        )
        _configure_dashboard_window(fig)
        fig.subplots_adjust(bottom=0.12)
        gs = fig.add_gridspec(
            3,
            2,
            width_ratios=[1.05, 1.35],
            height_ratios=[0.95, 1.55, 0.42],
            wspace=0.25,
            hspace=0.16,
        )
        ax_bar = fig.add_subplot(gs[0, 0])
        ax_txt = fig.add_subplot(gs[1, 0])
        status_ax = fig.add_subplot(gs[2, 0])
        ax_3d = fig.add_subplot(gs[:, 1], projection="3d")

        labels = ["confidence", "grab", "pinch"]
        bars = ax_bar.bar(labels, [0.0, 0.0, 0.0], color=["#4C78A8", "#F58518", "#54A24B"])
        ax_bar.set_ylim(0.0, 1.0)
        ax_bar.set_ylabel("value")
        ax_bar.set_title("Gesture Strengths")
        ax_bar.grid(axis="y", alpha=0.25)

        ax_txt.axis("off")
        info_text = ax_txt.text(
            0.0,
            0.98,
            "",
            va="top",
            family="monospace",
            fontsize=9.5,
            transform=ax_txt.transAxes,
            clip_on=True,
        )

        ax_3d.set_title("Right-Hand Pose (zeroed)")
        ax_3d.set_xlabel("X (mm)")
        ax_3d.set_ylabel("Y (mm)")
        ax_3d.set_zlabel("Z (mm)")
        ax_3d.set_xlim(-250, 250)
        ax_3d.set_ylim(-250, 250)
        ax_3d.set_zlim(-250, 250)
        ax_3d.view_init(elev=20, azim=-65)
        ax_3d.plot([0, 120], [0, 0], [0, 0], "r--", alpha=0.35)
        ax_3d.plot([0, 0], [0, 120], [0, 0], "g--", alpha=0.35)
        ax_3d.plot([0, 0], [0, 0], [0, 120], "b--", alpha=0.35)

        axis_len = 90.0
        (x_line,) = ax_3d.plot([0, axis_len], [0, 0], [0, 0], "r-", linewidth=3)
        (y_line,) = ax_3d.plot([0, 0], [0, axis_len], [0, 0], "g-", linewidth=3)
        (z_line,) = ax_3d.plot([0, 0], [0, 0], [0, axis_len], "b-", linewidth=3)
        (p_dot,) = ax_3d.plot([0], [0], [0], "ko", markersize=6)

        status_ax.axis("off")
        status_text = status_ax.text(
            0.0,
            0.92,
            "",
            va="top",
            fontsize=8.5,
            family="monospace",
            transform=status_ax.transAxes,
            clip_on=True,
        )
        home_button_ax = fig.add_axes([0.62, 0.028, 0.14, 0.05])
        button_ax = fig.add_axes([0.79, 0.028, 0.17, 0.05])
        home_button = Button(home_button_ax, "Home")
        button = Button(button_ax, robot_controller.button_label())

        def refresh_button(status: Optional[RobotControlStatus] = None) -> None:
            if status is None:
                status = robot_controller.get_status()
            button.label.set_text(robot_controller.button_label(status))
            if not status.enabled:
                button_ax.set_facecolor("#808080")
                home_button_ax.set_facecolor("#808080")
            elif status.enabled and not status.debug and not status.robot_bridge_connected:
                button_ax.set_facecolor("#d62728")
                home_button_ax.set_facecolor("#d62728")
            elif status.active:
                button_ax.set_facecolor("#d95f02" if status.waiting_for_reference else "#1b9e77")
                home_button_ax.set_facecolor("#7570b3")
            else:
                button_ax.set_facecolor("#4c78a8")
                home_button_ax.set_facecolor("#7570b3")

        def toggle_robot_control(_event=None):
            robot_controller.toggle_active()
            refresh_button()

        def request_home(_event=None):
            robot_controller.request_home()
            refresh_button()

        def on_key(event):
            if event.key == " ":
                listener.toggle_tracking()
            elif event.key == "r":
                toggle_robot_control()
            elif event.key == "h":
                request_home()
            elif event.key in ("q", "Q"):
                stop_event.set()
                plt.close(fig)

        def on_figure_close(_event):
            stop_event.set()

        home_button.on_clicked(request_home)
        button.on_clicked(toggle_robot_control)
        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("close_event", on_figure_close)

        print(f"Loaded config from {args.config}")
        print(
            "Robot control config: "
            f"enabled={robot_config.enabled} debug={robot_config.debug} "
            f"command_rate_hz={robot_config.command_rate_hz:.1f} "
            f"controller_frequency={robot_config.controller_frequency:.1f} "
            f"sensitivity={robot_config.position_sensitivity_m_per_mm:.6f}m/mm "
            f"axis_scale={robot_config.axis_scale.tolist()} "
            f"tcp_relative_rpy_deg={robot_config.tcp_relative_rpy_deg.tolist()} "
            f"Kx_scale={robot_config.Kx_scale:.3f} "
            f"Kxd_scale={robot_config.Kxd_scale.tolist()} "
            f"orientation_control={robot_config.orientation_control}"
        )
        if wsg_config.enabled:
            wsg_manager = WSGControllerManager(wsg_config, listener, shm_manager, robot_controller)
            print(
                "WSG Gripper control enabled (hand gestures apply after robot live mode is on; "
                "robot_control.enabled=false uses gestures immediately)."
            )
        else:
            print("WSG Gripper control disabled in config.")
        if robot_config.enabled:
            if robot_config.debug:
                print("Robot control is in DEBUG mode; Franka targets are visualized but not sent.")
            else:
                print(
                    f"Robot control will use FrankaInterpolationController via {robot_config.robot_ip}:{robot_config.robot_port}."
                )
        print(
            "Dashboard opened. Space toggles zeroed pose preview; 'r' / Robot toggles live control; "
            "'h' / Home moves to configured home joint target; 'q' or close window to quit."
        )

        with connection.open():
            connection.set_tracking_mode(TrackingMode.Desktop)
            last_log = 0.0

            while not stop_event.is_set() and plt.fignum_exists(fig.number):
                snap, active, has_pose, pose_pos, pose_rot = listener.get_snapshot()
                robot_status = robot_controller.get_status()

                conf = _clip01(snap.confidence)
                grab = _clip01(snap.grab_strength)
                pinch = _clip01(snap.pinch_strength)
                for bar, val in zip(bars, (conf, grab, pinch)):
                    bar.set_height(val)

                if snap.sphere_center is None:
                    sphere_line = "sphere_center: not exposed in current binding"
                else:
                    sx, sy, sz = snap.sphere_center
                    sphere_line = f"sphere_center(mm): [{sx:7.1f}, {sy:7.1f}, {sz:7.1f}]"

                if not robot_status.enabled:
                    robot_line = "robot: disabled in config"
                elif robot_status.enabled and not robot_status.debug and not robot_status.robot_bridge_connected:
                    err = robot_status.robot_bridge_error or "?"
                    robot_line = f"robot: NO BRIDGE ({err[:48]}{'…' if len(err) > 48 else ''})"
                elif robot_status.active and robot_status.waiting_for_reference:
                    robot_line = "robot: active, waiting for right-hand reference"
                elif robot_status.active:
                    robot_line = "robot: active"
                else:
                    robot_line = "robot: inactive (last target held on server)"

                wsg_line = "wsg_control: disabled"
                if wsg_manager is not None and wsg_manager._running:
                    if wsg_manager.gesture_control_live():
                        wsg_line = f"wsg_control: active ({wsg_config.gesture})"
                    else:
                        wsg_line = "wsg_control: idle (enable robot live mode for gripper)"

                info_text.set_text(
                    f"frame:      {snap.frame_id}\n"
                    f"visible:    {snap.visible}\n"
                    f"hand:       {snap.hand_label}\n\n"
                    f"confidence: {conf:0.3f}\n"
                    f"grab:       {grab:0.3f}\n"
                    f"pinch:      {pinch:0.3f}\n\n"
                    f"{sphere_line}\n\n"
                    f"pose_preview_active: {active}\n"
                    f"right_visible:       {snap.right_visible}\n"
                    f"pose_valid:          {has_pose}\n"
                    f"{robot_line}\n"
                    f"robot_debug:         {robot_status.debug}\n"
                    f"robot_bridge:        {robot_status.robot_bridge_connected}\n"
                    f"{wsg_line}\n"
                )

                plot_source = "idle"
                if active and has_pose:
                    plot_pos, plot_rot = to_plot_frame(pose_pos, pose_rot)
                    ex, ey, ez = plot_rot[:, 0], plot_rot[:, 1], plot_rot[:, 2]
                    plot_source = "preview"
                elif robot_status.active and robot_status.has_display_pose and not robot_status.waiting_for_reference:
                    plot_pos = robot_status.display_pose_pos_mm
                    plot_rot = robot_status.display_pose_R
                    ex, ey, ez = plot_rot[:, 0], plot_rot[:, 1], plot_rot[:, 2]
                    plot_source = "robot_pre_tcp"
                else:
                    plot_pos = np.zeros(3, dtype=np.float64)
                    ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    ey = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                    ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)

                x_end = plot_pos + axis_len * ex
                y_end = plot_pos + axis_len * ey
                z_end = plot_pos + axis_len * ez

                x_line.set_data([plot_pos[0], x_end[0]], [plot_pos[1], x_end[1]])
                x_line.set_3d_properties([plot_pos[2], x_end[2]])
                y_line.set_data([plot_pos[0], y_end[0]], [plot_pos[1], y_end[1]])
                y_line.set_3d_properties([plot_pos[2], y_end[2]])
                z_line.set_data([plot_pos[0], z_end[0]], [plot_pos[1], z_end[1]])
                z_line.set_3d_properties([plot_pos[2], z_end[2]])
                p_dot.set_data([plot_pos[0]], [plot_pos[1]])
                p_dot.set_3d_properties([plot_pos[2]])

                refresh_button(robot_status)
                last_cmd = robot_status.last_command_pos_m
                ax_3d.set_title(
                    "Right-Hand Pose (zeroed)" if plot_source == "preview"
                    else "Right-Hand Pose (pre-TCP transform)" if plot_source == "robot_pre_tcp"
                    else "Right-Hand Pose / Robot Command"
                )
                bridge_warn = ""
                if (
                    robot_status.enabled
                    and not robot_status.debug
                    and not robot_status.robot_bridge_connected
                ):
                    bridge_warn = " | NO ROBOT BRIDGE — see README.md (Polymetis server)"
                status_text.set_text(
                    f"plot={plot_source}  frame={snap.frame_id}  right_visible={snap.right_visible}  "
                    f"active={active}  pose_valid={has_pose}{bridge_warn}\n"
                    f"display p(mm)=[{plot_pos[0]:+.1f}, {plot_pos[1]:+.1f}, {plot_pos[2]:+.1f}]  "
                    f"robot_cmd(m)=[{last_cmd[0]:+.3f}, {last_cmd[1]:+.3f}, {last_cmd[2]:+.3f}]"
                )

                now = time.time()
                if now - last_log >= 1.0:
                    last_log = now
                    print(
                        f"frame={snap.frame_id} visible={snap.visible} hand={snap.hand_label} "
                        f"conf={conf:.3f} grab={grab:.3f} pinch={pinch:.3f} "
                        f"right_visible={snap.right_visible} preview_active={active} pose_valid={has_pose} "
                        f"robot_active={robot_status.active} robot_waiting={robot_status.waiting_for_reference}"
                    )

                fig.canvas.draw_idle()
                _dashboard_gui_idle(fig, 1.0 / 30.0)
    finally:
        try:
            if fig is not None:
                plt.close(fig)
        except Exception:
            pass
        if robot_controller is not None:
            robot_controller.close()
        if wsg_manager is not None:
            wsg_manager.close()
        if shm_manager is not None:
            shm_manager.shutdown()

    print("Exiting.")


if __name__ == "__main__":
    main()
