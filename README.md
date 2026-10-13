```markdown
# Real-Time Hand Gesture Recognition for Robotic Arm Control using GRU-KAN

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)]()
[![Platform](https://img.shields.io/badge/Platform-CPU--Only-blue?style=flat)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-90.33%25-orange?style=flat)]()

**Official implementation of the paper:**

*Real-Time Hand Gesture Recognition for Robotic Arm Control using a GRU-KAN Framework with Sliding-Window Inference*

Israa F. Fatih, Wisam S. Hacham, Muhannad K. Sabir

Al-Khwarizmi College of Engineering, University of Baghdad, Iraq

</div>

---

## 📖 Overview

This repository provides a complete, end-to-end pipeline for **real-time hand gesture recognition** and **touchless robotic arm control** using a lightweight hybrid deep learning architecture called **GRU-KAN** — combining a Gated Recurrent Unit (GRU) for temporal modeling with a Kolmogorov–Arnold Network (KAN) for nonlinear classification.

The system operates **entirely on CPU** and processes **raw 3D skeletal data** from an Ultraleap Stereo IR 170 camera at 90 fps — **without** Kalman filtering, FFT, or any manual signal processing.

> **Companion repository (ablation study):**  
> [GRU-KAN-Hand-Gesture](https://github.com/asraa11/GRU-KAN-Hand-Gesture)

---

## ✨ Key Features

- 🎯 **90.33% window-level accuracy** with only **27,615 parameters**
- ⚡ **90 predictions per second** on standard CPU hardware
- 🤖 **Real-time control** of a physical 3-DOF robotic arm
- 🧠 **Learnable B-spline activations** (KAN) replace fixed-activation MLPs
- 🔄 **Sliding-window inference** for continuous streaming classification
- 🛡️ **Semantic Guard + Hysteresis** for robust gesture confirmation
- 🎛️ **Gesture-conditioned axis filter** for decoupled control axes
- 🧊 **Three-stage joint-command smoothing** (EMA + velocity + acceleration)
- 💻 **CPU-only** — no GPU required

---

## 📊 Performance Summary

| Metric | Value |
|:---|:---:|
| **Test Accuracy (window-level)** | **90.33%** |
| **Macro F1-Score** | **0.904** |
| **Macro Precision** | 0.908 |
| **Macro Recall** | 0.903 |
| **Model Parameters** | **27,615** |
| **Prediction Rate** | 90 predictions/s |
| **Confirmation Delay** | ≈ 44 ms |
| **Arm-Motion Success Rate** | **88%** (±5°) |
| **Gripper Actuation Success** | **100%** |
| **Execution Platform** | CPU-only (Intel i7-12700H) |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Ultraleap Stereo IR 170                   │
│                         (90 fps, 27 joints)                  │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│              14 Kinematic Features (raw)                     │
│   palm (x,y,z) · velocity (vx,vy,vz) · grab · pinch ·        │
│   pinch_distance · 5 finger-extension flags                  │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│              Global Z-score Normalization                    │
│           (mean & std computed from training set)            │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│           GRU Encoder (1 layer, hidden size = 38)            │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│         Multi-Head Attention (2 heads, learnable)            │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│      Attention Pooling + Residual Projection                 │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│            KAN Classifier [38, 19, 9]                        │
│           (grid size G=3, spline order k=3)                  │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│     Sliding-Window Inference + Unanimous Agreement           │
│            (40-frame window, stride = 1)                     │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│              Gesture → Action Mapping                        │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│          Cylindrical Inverse Kinematics (3-DOF)              │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│       Three-Stage Smoothing (EMA + Velocity + Accel)         │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│              Arduino Mega 2560 + PCA9685                     │
│                   → 3-DOF Robotic Arm                        │
└──────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
GRU-KAN-RealTime-Robot/
│
├── training/                          # Model training code
│   ├── train_deployment.py            # Train 9-gesture model
│   └── README_training.md
│
├── realtime/                          # Real-time inference pipeline
│   ├── realtime_inference.py          # Main GUI + inference
│   └── README_realtime.md
│
├── arduino/                           # Embedded firmware
│   └── robot_arm_controller.ino       # Arduino Mega 2560
│
├── configs/                           # Configuration files
│   └── deployment_config.yaml
│
├── data/                              # Sample data (not full dataset)
│   ├── sample_open.csv
│   ├── sample_close.csv
│   └── README_data.md
│
├── docs/                              # Documentation & figures
│   ├── architecture.png
│   ├── sliding_window.png
│   ├── robotic_arm.jpg
│   └── setup_photo.jpg
│
├── results/                           # Experimental results
│   ├── confusion_matrix_9gestures.png
│   ├── training_curves_9gestures.png
│   ├── per_class_metrics_9gestures.png
│   └── lr_schedule.png
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- **OS:** Ubuntu 22.04 LTS (recommended) or Windows 10/11
- **Python:** 3.10+
- **Hardware:** Ultraleap Stereo IR 170 + Arduino Mega 2560 + 3-DOF robotic arm

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/asraa11/GRU-KAN-RealTime-Robot.git
cd GRU-KAN-RealTime-Robot
```

### 2️⃣ Install Dependencies

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3️⃣ Verify Hardware Connection

```bash
# Check Ultraleap camera
python -c "import leap; print('Ultraleap OK')"

# Check Arduino serial port
python -c "import serial.tools.list_ports; print([p.device for p in serial.tools.list_ports.comports()])"
```

### 4️⃣ Train the Model

```bash
cd training
python train_deployment.py
```

Expected output:
```
✅ Test Accuracy: 90.33%
🏆 Best Val Acc: 92.78%
📊 Model saved to: hand_gesture_fast_smart_9classes.pth
```

### 5️⃣ Run Real-Time Inference

```bash
cd realtime
python realtime_inference.py
```

The GUI provides:
- 📷 Real-time 3D hand tracking visualization
- 🎯 Live gesture recognition with confidence scores
- 🦾 Robotic arm command transmission
- 📊 Motion execution logging

---

## 🤖 Gesture Vocabulary

| # | Gesture | Robotic Arm Action | Description |
|:-:|:---:|:---:|:---|
| 1 | `open` | Gripper Open | Activate both gripper motors to open |
| 2 | `close` | Gripper Close | Activate both gripper motors to close |
| 3 | `grab` | Gripper Grab | Close with force ∝ `grab_strength` |
| 4 | `push` | End-Effector Forward | Move forward along +Z |
| 5 | `pull` | End-Effector Backward | Move backward along −Z |
| 6 | `raise` | End-Effector Up | Move upward along +Y |
| 7 | `lower` | End-Effector Down | Move downward along −Y |
| 8 | `left` | Base Rotate CCW | Rotate base counterclockwise |
| 9 | `right` | Base Rotate CW | Rotate base clockwise |

---

## 🔧 Hardware Requirements

| Component | Model | Purpose |
|:---|:---|:---|
| **Hand Tracker** | Ultraleap Stereo IR 170 | 90 fps, 170°×170° FOV, 27 joints |
| **Robotic Arm** | Custom 3-DOF articulated | Base + Shoulder + Elbow |
| **Gripper** | 2-motor (thumb + 4-finger) | Independent actuation |
| **High-Torque Servos** | 2× DS3218 | Shoulder, Elbow |
| **Standard Servos** | 4× MG996R | Base, Passive joint, Gripper |
| **Microcontroller** | Arduino Mega 2560 | Low-level control |
| **PWM Driver** | PCA9685 (16-ch, 12-bit) | Servo control |
| **Power Supply** | 6 V DC / 5 A | External servo power |
| **Computing Unit** | Intel i7-12700H, 32 GB RAM | CPU-only inference |

---

## 📊 Training Configuration

| Parameter | Value |
|:---|:---:|
| Optimizer | AdamW |
| Learning Rate | 4 × 10⁻⁴ |
| Weight Decay | 5 × 10⁻⁴ |
| Batch Size | 64 |
| Maximum Epochs | 60 |
| Early-Stopping Patience | 15 |
| LR Scheduler | Warmup + Cosine Annealing Warm Restarts |
| Warmup Epochs | 5 |
| Cosine T₀ / T_mult | 10 / 2 |
| Loss Function | Focal Loss (α=0.25, γ=2.0) |
| Mixup α | 0.15 |
| Dropout Rate | 0.4 |
| Gradient Clipping | 1.0 (L2 norm) |
| Random Seed | 42 |
| Compute | CPU-only, single-threaded |

---

## 🔬 Deployment Configuration

| Parameter | Value |
|:---|:---:|
| Sliding-Window Size | 40 frames |
| Training Stride | 15 frames (62.5% overlap) |
| Deployment Stride | 1 frame (97.5% overlap) |
| Prediction Rate | Up to 90 predictions/s |
| Unanimous-Agreement Window | Last 4 predictions |
| Confirmation Delay | ≈ 44 ms @ 90 fps |
| Effective Training FPS | 45 fps (odd-frame decimation) |
| Deployment FPS | 90 fps (full stream) |
| EMA Filter α | 0.45 |
| Control Frequency | 30 Hz |
| Serial Baud Rate | 9600 bps |

---

## 📁 Data Format

Each CSV file contains **one gesture sequence** = **360 frames × 14 features**:

| Column | Description | Unit |
|:---|:---|:---:|
| `palm_x`, `palm_y`, `palm_z` | Palm centroid | mm |
| `palm_velocity_x`, `palm_velocity_y`, `palm_velocity_z` | Palm velocity (from SDK) | mm/s |
| `grab_strength` | Grip strength | 0–1 |
| `pinch_strength` | Pinch strength | 0–1 |
| `pinch_distance` | Pinch distance | mm |
| `thumb_extended` | Thumb extension flag | 0/1 |
| `index_extended` | Index extension flag | 0/1 |
| `middle_extended` | Middle extension flag | 0/1 |
| `ring_extended` | Ring extension flag | 0/1 |
| `pinky_extended` | Pinky extension flag | 0/1 |

**File naming convention:**
```
open_001.csv
close_001.csv
grab_001.csv
push_001.csv
pull_001.csv
raise_001.csv
lower_001.csv
left_001.csv
right_001.csv
```

---

## 📈 Results

### Confusion Matrix (9 Gestures, 900 Held-Out Windows)

![Confusion Matrix](results/confusion_matrix_9gestures.png)

### Per-Class Metrics

| Gesture | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| `close` | 0.744 | 0.900 | 0.814 | 100 |
| `grab` | 0.788 | 0.820 | 0.804 | 100 |
| `left` | 0.969 | 0.930 | 0.949 | 100 |
| `lower` | 0.989 | 0.910 | 0.948 | 100 |
| `open` | 0.802 | 0.690 | 0.742 | 100 |
| `pull` | 0.943 | 1.000 | 0.971 | 100 |
| `push` | 0.943 | 0.990 | 0.966 | 100 |
| `raise` | 1.000 | 0.910 | 0.953 | 100 |
| `right` | 0.990 | 0.980 | 0.985 | 100 |
| **Macro Avg** | **0.908** | **0.903** | **0.904** | **900** |

### Training Dynamics

![Training Curves](results/training_curves_9gestures.png)

---

## 🛠️ Troubleshooting

<details>
<summary><b>Ultraleap camera not detected</b></summary>

1. Ensure the camera is connected via **USB 3.0** (blue port).
2. Install the [Ultraleap Gemini SDK](https://developer.ultraleap.com/).
3. Test with the official Ultraleap Tracking Visualizer.
4. On Linux, add udev rules:
   ```bash
   sudo cp 99-ultraleap.rules /etc/udev/rules.d/
   sudo udevadm control --reload-rules
   ```
</details>

<details>
<summary><b>Arduino serial port error</b></summary>

1. Check the port:
   ```bash
   # Linux
   ls /dev/ttyUSB* /dev/ttyACM*
   # Windows
   # Device Manager → Ports (COM & LPT)
   ```
2. Update `SERIAL_PORT` in `realtime/realtime_inference.py`.
3. On Linux, add user to `dialout` group:
   ```bash
   sudo usermod -aG dialout $USER
   ```
</details>

<details>
<summary><b>Servos jitter or lose torque</b></summary>

1. Ensure external **6 V / 5 A** supply is connected.
2. Common ground between Arduino, PCA9685, and power supply.
3. Add a **1000 µF capacitor** across the servo power rails.
4. Reduce simultaneous servo commands if needed.
</details>

<details>
<summary><b>Low recognition accuracy</b></summary>

1. Ensure the hand is **30–50 cm** from the camera.
2. Confirm **Neutral** is captured before each session.
3. Verify the correct scaler is loaded from the checkpoint.
4. Re-calibrate with the **🎯 Neutral** button in the GUI.
</details>

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{fatih2026realtime,
  title={Real-Time Hand Gesture Recognition for Robotic Arm Control 
         using a GRU-KAN Framework with Sliding-Window Inference},
  author={Fatih, Israa F. and Hacham, Wisam S. and Sabir, Muhannad K.},
  year={2026}
}
```

**Companion paper:**

```bibtex
@article{fatih2026grukan,
  title={GRU-KAN: A Hybrid Recurrent--Kolmogorov--Arnold Network 
         for Low-Latency 3D Hand Gesture Classification 
         using Direct Kinematic Features from Skeletal Data},
  author={Fatih, Israa F. and Hacham, Wisam S. and Sabir, Muhannad K.},
  journal={International Journal of Intelligent Engineering and Systems},
  year={2026}
}
```

---

## 👥 Authors

| Author | Affiliation | Role |
|:---|:---|:---|
| **Israa F. Fatih** | Mechatronics Eng., Al-Khwarizmi College | First Author |
| **Wisam S. Hacham** | Mechatronics Eng., Al-Khwarizmi College | Corresponding Author |
| **Muhannad K. Sabir** | Biomedical Eng., Al-Khwarizmi College | Co-Author |

📧 **Corresponding Author:** wisam@kecbu.uobaghdad.edu.iq

🏛️ **Institution:** University of Baghdad, Al-Khwarizmi College of Engineering, Iraq

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Ultraleap** for the Stereo IR 170 camera and SDK
- **PyTorch** team for the deep learning framework
- **efficient-kan** library for KAN implementation
- University of Baghdad for supporting this research

---

<div align="center">

**⭐ If you find this work useful, please consider giving it a star! ⭐**

Made with ❤️ at the University of Baghdad

</div>
```
