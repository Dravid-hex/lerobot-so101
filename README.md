# SO-101 Intelligent Robotic Arm — TE3002B

Implementation of imitation learning policies on a SO-101 6-DOF robotic arm using [LeRobot](https://github.com/huggingface/lerobot) (HuggingFace). Includes custom patches for 12V Feetech motors, ArUco-based orientation detection, and ACT/SmolVLA policy training.

**Course:** TE3002B — Implementación de Robótica Inteligente  
**Institution:** Tecnológico de Monterrey  
**HuggingFace:** [Dravid419](https://huggingface.co/Dravid419)

---

## Demos

| Task | Policy | Dataset |
|------|--------|---------|
| Eraser stacking | [act_eraserStacking](https://huggingface.co/Dravid419/act_eraserStacking) | [eraserStacking](https://huggingface.co/datasets/Dravid419/eraserStacking) |
| Caiman pick & place | [smolvla_caimanes](https://huggingface.co/Dravid419/smolvla_caimanes) | [caimanes](https://huggingface.co/datasets/Dravid419/caimanes) |
| ArUco cube → left zone | [act_aruco_left](https://huggingface.co/Dravid419/act_aruco_left) | [aruco_left](https://huggingface.co/datasets/Dravid419/aruco_left) |

---

## Hardware

- **Robot:** SO-101 leader-follower pair (6x STS3215 Feetech motors, 12V)
- **Cameras:** 2x USB 1280×720 @ 30fps MJPG (scene frontal + side lateral)
- **Training GPU:** NVIDIA RTX 6000
- **Inference GPU:** NVIDIA RTX 4070 Laptop

---

## Repository Structure

```
so101-project/
├── aruco_cube_controller.py     # ArUco orientation detector + lerobot policy launcher
├── calibrate_camera.py          # Camera calibration with checkerboard
├── define_zones.py              # Interactive zone definition tool
├── calibrate_eraser_reward.py   # Vision-based reward detector (for DPPO)
├── camera_calibration.json      # Intrinsic camera parameters
├── orientation_calibration.json # ArUco orientation reference vectors
├── zones.json                   # Target zone pixel coordinates
├── follower_calibration.json    # SO-101 follower arm motor calibration
├── leader_calibration.json      # SO-101 leader arm motor calibration
└── patches/                     # Modified lerobot source files
    ├── so_follower.py           # Per-motor read loop, gripper protection, wrist_roll wrap fix
    ├── feetech.py               # Voltage error ignore, skip wrist_roll homing write
    ├── camera_opencv.py         # MJPG validation fix, frame timeout, error recovery
    ├── motors_bus.py            # Safe disconnect on motor error
    └── context.py               # Load input_features from saved policy config
```

---

## Setup

### 1. Install LeRobot

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[smolvla,training]"
hf auth login
```

### 2. Apply patches

Copy the files from `patches/` into the lerobot source:

```bash
cp patches/so_follower.py src/lerobot/robots/so_follower/so_follower.py
cp patches/feetech.py     src/lerobot/motors/feetech/feetech.py
cp patches/camera_opencv.py src/lerobot/cameras/opencv/camera_opencv.py
cp patches/motors_bus.py  src/lerobot/motors/motors_bus.py
cp patches/context.py     src/lerobot/rollout/context.py
```

> **Why patches?** The SO-101 uses 12V Feetech STS3215 motors which always report a voltage error bit (error=1). Stock LeRobot rejects these packets. The patches ignore bit 0 of the error byte, fix sync_read failures with a per-motor fallback loop, and resolve camera validation issues with V4L2+MJPG.

### 3. Motor setup

```bash
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
lerobot-setup-motors --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1
```

### 4. Calibrate

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_awesome_follower_arm
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=my_awesome_leader_arm
```

---

## Recording Demos

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{scene: {type: opencv, index_or_path: /dev/video2, width: 1280, height: 720, fps: 30, fourcc: MJPG, backend: V4L2}, side: {type: opencv, index_or_path: /dev/video4, width: 1280, height: 720, fps: 30, fourcc: MJPG, backend: V4L2}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --dataset.repo_id=Dravid419/your_dataset \
    --dataset.num_episodes=50 \
    --dataset.single_task="Your task description" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=4 \
    --dataset.vcodec=h264_nvenc \
    --dataset.push_to_hub=false
```

> **Note:** Camera paths (`/dev/videoX`) change between sessions. Always run `lerobot-find-cameras opencv` first.

---

## Training

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True lerobot-train \
    --dataset.repo_id=Dravid419/your_dataset \
    --dataset.root=/path/to/local/dataset \
    --policy.type=act \
    --policy.repo_id=Dravid419/your_policy \
    --output_dir=outputs/train/your_policy \
    --policy.device=cuda \
    --wandb.enable=false \
    --steps=100000
```

> For 2-camera datasets with RTX 4070 (8GB), add `--batch_size=2 --policy.resize_shape="[240, 320]"`. Use RTX 6000 for full resolution training.

---

## ArUco Cube Controller

Detects the orientation of an ArUco cube and launches the corresponding ACT policy.

### Camera calibration
```bash
python calibrate_camera.py --device /dev/video2 --width 1280 --height 720
```

### Define target zones
```bash
python define_zones.py --device /dev/video2 --width 1280 --height 720
```

### Calibrate cube orientations
```bash
python aruco_cube_controller.py \
    --calib camera_calibration.json \
    --device /dev/video2 \
    --calibrate-orientations
```

### Run full pipeline
```bash
python aruco_cube_controller.py \
    --calib camera_calibration.json \
    --device /dev/video2 \
    --zones zones.json \
    --policy-left  /path/to/act_aruco_left/checkpoints/last/pretrained_model \
    --policy-right /path/to/act_aruco_right/checkpoints/last/pretrained_model \
    --policy-center /path/to/act_aruco_center/checkpoints/last/pretrained_model \
    --follower-port /dev/ttyACM0 \
    --scene-cam /dev/video2 \
    --side-cam /dev/video4
```

Press **SPACE** to launch the policy for the detected orientation.

---

## Key Patches Summary

| File | What changed | Why |
|------|-------------|-----|
| `feetech.py` | Ignore voltage error bit (bit 0) | 12V motors always set this bit |
| `feetech.py` | Skip homing_offset write for wrist_roll | Writing offset causes motor to move to extreme |
| `so_follower.py` | Per-motor read loop with fallback | sync_read fails on 12V motors |
| `so_follower.py` | Gripper torque limit 30%, current 25% | Prevent motor burnout |
| `so_follower.py` | wrist_roll wrap-around correction | Full-turn motor crosses boundary |
| `camera_opencv.py` | Skip resolution/fps validation | V4L2+MJPG reports wrong success flags |
| `camera_opencv.py` | Frame timeout 500ms → 2000ms | Inference loop slower than camera FPS |
| `camera_opencv.py` | Return last frame on thread failure | Prevent crash on camera disconnect |
| `motors_bus.py` | Try/except on disconnect torque disable | Prevent crash when motor errors on exit |
| `context.py` | Load input_features from saved policy | Policy config missing image features bug |

---

## References

- Zhao, T. Z. et al. (2023). *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.* RSS 2023.
- Chi, C. et al. (2023). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS 2023.
- Ren, A. et al. (2024). *DPPO: Diffusion Policy Policy Optimization.* arXiv:2409.00588.
- Brohan, A. et al. (2023). *RT-2: Vision-Language-Action Models.* arXiv:2307.15818.
- Garrido-Jurado, S. et al. (2014). *Automatic generation and detection of highly reliable fiducial markers.* Pattern Recognition.
- LeRobot. HuggingFace. https://github.com/huggingface/lerobot
- SO-ARM100. The Robot Studio. https://github.com/TheRobotStudio/SO-ARM100
