#!/usr/bin/env python3
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import math
from dataclasses import dataclass
from multiprocessing.managers import SharedMemoryManager


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
os.chdir(ROOT_DIR)

UMI_DIR = os.path.join(ROOT_DIR, "dependencies", "universal_manipulation_interface")
LEAP_API_DIR = os.path.join(
    ROOT_DIR, "dependencies", "leapc-python-bindings", "leapc-python-api", "src"
)
LEAP_CFFI_DIR = os.path.join(
    ROOT_DIR, "dependencies", "leapc-python-bindings", "leapc-cffi", "src"
)
sys.path.extend([ROOT_DIR, UMI_DIR, LEAP_API_DIR, LEAP_CFFI_DIR])

import leap  # noqa: E402
from leap.enums import TrackingMode  # noqa: E402

from umi.common.precise_sleep import precise_wait  # noqa: E402
from umi.real_world.wsg_controller import WSGController  # noqa: E402


def is_right_hand(hand) -> bool:
    if hasattr(hand, "is_right"):
        try:
            return bool(hand.is_right)
        except Exception:
            pass
    hand_type = getattr(hand, "type", None)
    return hand_type is not None and str(hand_type) in {
        "HandType.Right",
        "eLeapHandType_Right",
    }


def clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


@dataclass
class GestureSnapshot:
    frame_id: int = 0
    visible: bool = False
    hand_label: str = "none"
    confidence: float = 0.0
    grab_strength: float = 0.0
    pinch_strength: float = 0.0


class GestureListener(leap.Listener):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = GestureSnapshot()

    def on_tracking_event(self, event) -> None:
        right = None
        for hand in event.hands:
            if is_right_hand(hand):
                right = hand
                break
        gesture_hand = right if right is not None else (event.hands[0] if len(event.hands) > 0 else None)

        with self._lock:
            self._snapshot.frame_id = int(getattr(event, "tracking_frame_id", 0))
            if gesture_hand is None:
                self._snapshot.visible = False
                self._snapshot.hand_label = "none"
                self._snapshot.confidence = 0.0
                self._snapshot.grab_strength = 0.0
                self._snapshot.pinch_strength = 0.0
            else:
                self._snapshot.visible = True
                self._snapshot.hand_label = "right" if is_right_hand(gesture_hand) else "left/first"
                self._snapshot.confidence = float(getattr(gesture_hand, "confidence", 0.0))
                self._snapshot.grab_strength = float(getattr(gesture_hand, "grab_strength", 0.0))
                self._snapshot.pinch_strength = float(getattr(gesture_hand, "pinch_strength", 0.0))

    def get_snapshot(self) -> GestureSnapshot:
        with self._lock:
            return GestureSnapshot(
                frame_id=self._snapshot.frame_id,
                visible=self._snapshot.visible,
                hand_label=self._snapshot.hand_label,
                confidence=self._snapshot.confidence,
                grab_strength=self._snapshot.grab_strength,
                pinch_strength=self._snapshot.pinch_strength,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test WSG gripper opening control from Leap pinch or grasp strength."
    )
    parser.add_argument("--hostname", default="172.16.0.4")
    parser.add_argument("--port", type=int, default=1000)
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--home-to-open", dest="home_to_open", action="store_true", default=True)
    parser.add_argument("--home-to-close", dest="home_to_open", action="store_false")
    parser.add_argument("--max-speed", type=float, default=400.0)
    parser.add_argument("--max-pos", type=float, default=110.0)
    parser.add_argument(
        "--gesture",
        choices=("pinch", "grasp"),
        default="pinch",
        help="Leap gesture source used to control gripper opening.",
    )
    parser.add_argument(
        "--filter-time-constant",
        type=float,
        default=0.05,
        help="Low-pass time constant in seconds for the opening signal.",
    )
    parser.add_argument("--command-latency", type=float, default=0.0)
    parser.add_argument("--receive-latency", type=float, default=0.0)
    parser.add_argument("--use-meters", action="store_true")
    parser.add_argument(
        "--direct-strength",
        action="store_true",
        help="Use gesture strength directly as opening. By default, opening = 1 - strength.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.frequency <= 0:
        raise ValueError("--frequency must be > 0")
    if args.max_speed <= 0:
        raise ValueError("--max-speed must be > 0")
    if args.max_pos <= 0:
        raise ValueError("--max-pos must be > 0")
    if args.filter_time_constant < 0:
        raise ValueError("--filter-time-constant must be >= 0")


def main() -> None:
    args = parse_args()
    validate_args(args)

    dt = 1.0 / args.frequency
    filter_alpha = (
        1.0
        if args.filter_time_constant <= 0.0
        else 1.0 - math.exp(-dt / args.filter_time_constant)
    )

    listener = GestureListener()
    connection = leap.Connection()
    connection.add_listener(listener)

    with SharedMemoryManager() as shm_manager:
        with WSGController(
            shm_manager=shm_manager,
            hostname=args.hostname,
            port=args.port,
            frequency=args.frequency,
            home_to_open=args.home_to_open,
            move_max_speed=args.max_speed,
            receive_latency=args.receive_latency,
            use_meters=args.use_meters,
            verbose=True,
        ) as gripper:
            print(
                f"UltraGrasp WSG control ready. gesture={args.gesture} "
                f"mapping={'strength' if args.direct_strength else '1-strength'} -> opening_signal -> width"
            )
            print("Press Ctrl+C to stop.")

            with connection.open():
                connection.set_tracking_mode(TrackingMode.Desktop)

                state = gripper.get_state()
                target_pos = float(state["gripper_position"])
                opening_signal = clip01(target_pos / args.max_pos)
                filtered_opening = opening_signal

                t_start = time.monotonic()
                gripper.restart_put(time.time())

                iter_idx = 0
                last_log = 0.0

                while True:
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - args.command_latency
                    t_command_target = t_cycle_end + dt

                    precise_wait(t_sample)

                    snap = listener.get_snapshot()
                    if snap.visible:
                        strength = snap.pinch_strength if args.gesture == "pinch" else snap.grab_strength
                        opening_signal = clip01(strength if args.direct_strength else 1.0 - strength)
                        filtered_opening += filter_alpha * (opening_signal - filtered_opening)

                    desired_pos = args.max_pos * filtered_opening
                    max_step = args.max_speed / args.frequency
                    dpos = max(-max_step, min(max_step, desired_pos - target_pos))
                    target_pos = max(0.0, min(args.max_pos, target_pos + dpos))

                    target_timestamp = t_command_target - time.monotonic() + time.time()
                    gripper.schedule_waypoint(target_pos, target_timestamp)

                    precise_wait(t_cycle_end)

                    now = time.time()
                    if now - last_log >= 0.2:
                        last_log = now
                        state = gripper.get_state()
                        strength = snap.pinch_strength if args.gesture == "pinch" else snap.grab_strength
                        print(
                            f"frame={snap.frame_id} visible={snap.visible} hand={snap.hand_label} "
                            f"conf={snap.confidence:.2f} pinch={snap.pinch_strength:.2f} grab={snap.grab_strength:.2f} "
                            f"gesture_strength={strength:.2f} opening_signal={opening_signal:.2f} "
                            f"filtered_opening={filtered_opening:.2f} target_mm={target_pos:.1f} "
                            f"actual_mm={float(state['gripper_position']):.1f}",
                            flush=True,
                        )

                    iter_idx += 1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
