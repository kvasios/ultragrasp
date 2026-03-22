## ultragrasp

UltraGrasp uses Ultraleap hand tracking on **Ubuntu 24.04** with a working Python API (LeapC via `leapc-python-bindings`).

### Demo

[![Watch the demo on YouTube](https://img.youtube.com/vi/zAbXOstUxzs/hqdefault.jpg)](https://www.youtube.com/watch?v=zAbXOstUxzs)

### What this repo contains

- **Ultraleap runtime setup**: installs the tracking service and `libLeapC.so` on Ubuntu.
- **Python bindings (submodule)**: `dependencies/leapc-python-bindings` (LeapC → Python via CFFI).
- **Working examples**: e.g. pinch detection (`simple_pinching_example.py`).

### Requirements

- **Ubuntu 24.04** (this setup was tested and worked).
- **Ultraleap device** plugged in and recognized (e.g. Leap Motion Controller / Ultraleap Hand Tracking Camera).
- **Python 3** in an isolated environment (recommended: **micromamba**).
- **Build tools** (needed if you compile the CFFI module): `build-essential`, `python3-dev`.

### 1) Install Ultraleap Hand Tracking Software (LeapC runtime)

This installs the tracking service plus the LeapC SDK files on Linux (including `libLeapC.so`).

```bash
wget -qO - https://repo.ultraleap.com/keys/apt/gpg | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/ultraleap.gpg >/dev/null
echo 'deb [arch=amd64] https://repo.ultraleap.com/apt stable main' | sudo tee /etc/apt/sources.list.d/ultraleap.list >/dev/null
sudo apt update
sudo apt install -y ultraleap-hand-tracking
leapctl eula
```

Sanity checks:

```bash
leapctl --help
ultraleap-hand-tracking-control-panel
```

### 2) Get the Python bindings (git submodule)

If you cloned this repo without submodules:

```bash
git submodule update --init --recursive
```

### 3) Create a Python environment

#### Recommended: micromamba

```bash
micromamba create -n ultragrasp -c conda-forge python=3.11 pip -y
micromamba activate ultragrasp
python -m pip install -U pip
```

Install build/runtime deps for the bindings:

```bash
python -m pip install -r dependencies/leapc-python-bindings/requirements.txt
```

#### Alternative: `venv`

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r dependencies/leapc-python-bindings/requirements.txt
```

### 4) Install the bindings

The bindings repo has two pieces:

- `leapc-cffi`: the compiled CFFI extension (`leapc_cffi`)
- `leapc-python-api`: the pure-Python API (`import leap`)

#### Option A (recommended): build and install `leapc-cffi` locally

This works on any Python version (as long as you have a compiler toolchain).

```bash
python -m build dependencies/leapc-python-bindings/leapc-cffi
python -m pip install dependencies/leapc-python-bindings/leapc-cffi/dist/leapc_cffi-*.whl
python -m pip install -e dependencies/leapc-python-bindings/leapc-python-api
```

#### Option B: use the precompiled module (only if available for your Python)

Ultraleap sometimes ships prebuilt `leapc_cffi` for specific Python versions. If `import leap` fails with a
missing `_leapc_cffi` module, use Option A above.

### 5) Run an example (pinch detection)

```bash
# activate whichever environment you created:
# micromamba activate ultragrasp
# OR: source .venv/bin/activate
python dependencies/leapc-python-bindings/examples/simple_pinching_example.py
```

You should see output like “Left/Right hand … pinching …”.

### Troubleshooting

- **Service not running / no hands in control panel**: fix the Ultraleap install first (until the control panel shows hands reliably).
- **`ModuleNotFoundError: No module named 'leapc_cffi._leapc_cffi'`**: build the module (Option A).
- **Non-default Leap SDK install location**: set `LEAPSDK_INSTALL_LOCATION` to the `LeapSDK` folder. On Linux x64 the default is:

```bash
export LEAPSDK_INSTALL_LOCATION="/usr/lib/ultraleap-hand-tracking-service"
```

### Robot / Franka bridge (Polymetis)

The UltraGrasp dashboard can send Cartesian commands to a Franka arm through the **UMI** stack: `FrankaInterpolationController` in `dependencies/universal_manipulation_interface` talks over **ZeroRPC** to a small server that wraps **Polymetis** (`RobotInterface`).

If you run into issues with the upstream UMI Franka API in your setup, this repo includes patched replacements in `src/umi-franka-patches`. Copy them over the corresponding UMI files before starting the bridge:

```bash
cp src/umi-franka-patches/franka_interpolation_controller.py \
  dependencies/universal_manipulation_interface/umi/real_world/franka_interpolation_controller.py

cp src/umi-franka-patches/launch_franka_interface_server.py \
  dependencies/universal_manipulation_interface/scripts_real/launch_franka_interface_server.py
```

These patched files are only needed if the stock UMI Franka scripts do not work correctly on your machine.

For a quick real-time server setup on the host PC, `ServoBox` is a convenient option for bringing up an RT VM and installing Polymetis with minimal manual setup:

```bash
servobox init
servobox pkg-install polymetis
servobox run polymetis
```

See [ServoBox](https://www.servobox.dev/) for installation and host-side RT setup details.

1. On the machine connected to the arm (often the Franka control PC / NUC), start the interface server (default listens on TCP port **4242**):

   ```bash
   python dependencies/universal_manipulation_interface/scripts_real/launch_franka_interface_server.py
   ```

2. In `config/ultragrasp.yaml`, set `robot_control.robot_ip` to that host’s address and `robot_control.robot_port` if you use a non-default port.

3. If no server is reachable when you launch `ultragrasp.py`, the GUI and terminal show a **no bridge** warning; live teleoperation and homing stay disabled until the server is up. For development without hardware, set `robot_control.debug: true` in the config.

### References

- Ultraleap Linux install: `https://docs.ultraleap.com/linux/`
- LeapC guide: `https://docs.ultraleap.com/api-reference/tracking-api/leapc-guide.html`
- LeapC Python bindings upstream: `https://github.com/ultraleap/leapc-python-bindings`

### Related projects / next steps

- **Socket streaming bridge** (send tracking data over WebSocket): `https://github.com/ultraleap/UltraleapTrackingWebSocket`
- **Raw camera frames (IR) for CV** (LeapUVC): `https://github.com/leapmotion/leapuvc`
- **Explore the LeapSDK contents** (headers/samples): set `LEAPSDK_INSTALL_LOCATION` (see troubleshooting) and inspect that folder.
