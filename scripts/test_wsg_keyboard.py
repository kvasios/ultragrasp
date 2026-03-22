#!/usr/bin/env python3
# pyright: reportMissingImports=false
import os
import sys
import time
import curses
import math
from dataclasses import dataclass
from multiprocessing.managers import SharedMemoryManager

import click


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
os.chdir(ROOT_DIR)

UMI_DIR = os.path.join(ROOT_DIR, "dependencies", "universal_manipulation_interface")
sys.path.extend([ROOT_DIR, UMI_DIR])


from umi.real_world.wsg_controller import WSGController  # noqa: E402
from umi.common.precise_sleep import precise_wait  # noqa: E402


@dataclass
class KeyEvents:
    input_delta: float = 0.0
    quit: bool = False
    refresh_actual: bool = False


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _read_key_events(stdscr, step: float) -> KeyEvents:
    events = KeyEvents()
    while True:
        ch = stdscr.getch()
        if ch == -1:
            break
        if ch == curses.KEY_LEFT:
            events.input_delta -= step
        elif ch == curses.KEY_RIGHT:
            events.input_delta += step
        elif ch in (ord("q"), 27):  # q or ESC
            events.quit = True
        elif ch in (ord("r"), ord("R")):
            events.refresh_actual = True
    return events


def _run_curses(
    stdscr,
    *,
    gripper: WSGController,
    frequency: float,
    max_speed: float,
    step: float,
    max_pos: float,
    key_decay: float,
    filter_time_constant: float,
    command_latency: float,
):
    dt = 1.0 / frequency
    raw_decay = 0.0 if key_decay <= 0.0 else math.exp(-dt / key_decay)
    filter_alpha = 1.0 if filter_time_constant <= 0.0 else 1.0 - math.exp(-dt / filter_time_constant)

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    state = gripper.get_state()
    target_pos = float(state["gripper_position"])

    t_start = time.monotonic()
    gripper.restart_put(time.time())

    iter_idx = 0
    raw_cmd = 0.0
    filtered_cmd = 0.0
    last_cmd_pos = None

    while True:
        t_cycle_end = t_start + (iter_idx + 1) * dt
        t_sample = t_cycle_end - command_latency
        t_command_target = t_cycle_end + dt

        precise_wait(t_sample)
        events = _read_key_events(stdscr, step=step)
        if events.quit:
            return

        if events.refresh_actual:
            state = gripper.get_state()
            target_pos = float(state["gripper_position"])
            raw_cmd = 0.0
            filtered_cmd = 0.0

        raw_cmd = _clamp(raw_cmd * raw_decay + events.input_delta, -1.0, 1.0)
        filtered_cmd += filter_alpha * (raw_cmd - filtered_cmd)

        dpos = filtered_cmd * max_speed / frequency
        target_pos = _clamp(target_pos + dpos, 0.0, max_pos)
        target_timestamp = t_command_target - time.monotonic() + time.time()
        gripper.schedule_waypoint(target_pos, target_timestamp)
        last_cmd_pos = target_pos

        precise_wait(t_cycle_end)

        if iter_idx % max(1, int(frequency // 10)) == 0:
            state = gripper.get_state()
            actual_pos = float(state["gripper_position"])
            force = float(state["gripper_force"])
            vel = float(state["gripper_velocity"])

            stdscr.erase()
            stdscr.addstr(0, 0, "WSG keyboard test (relative steps -> absolute waypoints)")
            stdscr.addstr(2, 0, "Controls:")
            stdscr.addstr(3, 2, "LEFT:  close (decrease width)")
            stdscr.addstr(4, 2, "RIGHT: open  (increase width)")
            stdscr.addstr(5, 2, "r: sync target to actual")
            stdscr.addstr(6, 2, "q / ESC: quit")
            stdscr.addstr(8, 0, f"key_step: {step:.6g}   max_speed: {max_speed:.6g}   freq: {frequency:.6g} Hz")
            stdscr.addstr(9, 0, f"key_decay: {key_decay:.6g} s   filter_tau: {filter_time_constant:.6g} s")
            stdscr.addstr(10, 0, f"target_pos: {target_pos:.6g}")
            stdscr.addstr(11, 0, f"last_cmd_pos: {last_cmd_pos if last_cmd_pos is not None else '-'}")
            stdscr.addstr(12, 0, f"actual_pos: {actual_pos:.6g}")
            stdscr.addstr(13, 0, f"actual_vel: {vel:.6g}")
            stdscr.addstr(14, 0, f"force_motor: {force:.6g}")
            stdscr.addstr(15, 0, f"raw_cmd: {raw_cmd:.6g}   filtered_cmd: {filtered_cmd:.6g}   max_pos: {max_pos:.6g}")
            stdscr.refresh()

        iter_idx += 1


@click.command()
@click.option("-h", "--hostname", default="172.16.0.4", show_default=True, type=str)
@click.option("-p", "--port", type=int, default=1000, show_default=True)
@click.option("-f", "--frequency", type=float, default=30.0, show_default=True)
@click.option("--home-to-open/--home-to-close", default=True, show_default=True)
@click.option("-ms", "--max-speed", type=float, default=400.0, show_default=True)
@click.option("-mp", "--max-pos", type=float, default=110.0, show_default=True)
@click.option(
    "--step",
    type=float,
    default=1.0,
    show_default=True,
    help="Normalized command impulse added per key repeat event.",
)
@click.option("--key-decay", type=float, default=0.01, show_default=True, help="Decay time constant for raw keyboard command.")
@click.option("--filter-time-constant", type=float, default=0.01, show_default=True, help="Low-pass time constant for the smoothed command.")
@click.option("--command-latency", type=float, default=0.0, show_default=True)
@click.option("--receive-latency", type=float, default=0.0, show_default=True)
@click.option("--use-meters/--use-mm", default=False, show_default=True)
def main(
    hostname: str,
    port: int,
    frequency: float,
    home_to_open: bool,
    max_speed: float,
    max_pos: float,
    step: float,
    key_decay: float,
    filter_time_constant: float,
    command_latency: float,
    receive_latency: float,
    use_meters: bool,
):
    if frequency <= 0:
        raise click.ClickException("--frequency must be > 0")
    if max_pos <= 0:
        raise click.ClickException("--max-pos must be > 0")
    if step <= 0:
        raise click.ClickException("--step must be > 0")
    if max_speed <= 0:
        raise click.ClickException("--max-speed must be > 0")
    if key_decay < 0:
        raise click.ClickException("--key-decay must be >= 0")
    if filter_time_constant < 0:
        raise click.ClickException("--filter-time-constant must be >= 0")

    with SharedMemoryManager() as shm_manager:
        with WSGController(
            shm_manager=shm_manager,
            hostname=hostname,
            port=port,
            frequency=frequency,
            home_to_open=home_to_open,
            move_max_speed=max_speed,
            receive_latency=receive_latency,
            use_meters=use_meters,
            verbose=True,
        ) as gripper:
            curses.wrapper(
                _run_curses,
                gripper=gripper,
                frequency=frequency,
                max_speed=max_speed,
                step=step,
                max_pos=max_pos,
                key_decay=key_decay,
                filter_time_constant=filter_time_constant,
                command_latency=command_latency,
            )


if __name__ == "__main__":
    main()

