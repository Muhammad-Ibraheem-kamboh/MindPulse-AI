# 👁️ Blinkr: Autonomous Examination Proctoring & 3D Attention Studio

[![Live Demo](https://img.shields.io/badge/Demo-Live%20Web%20Studio-38BDF8?style=for-the-badge&logo=googlechrome&logoColor=white)](https://muhammad-ibraheem-kamboh.github.io/MindPulse-AI/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/AI-Google%20MediaPipe-00897B.svg?style=flat)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8.svg?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![Privacy: 100% On-Device](https://img.shields.io/badge/Privacy-100%25%20On--Device%20Airgap-10B981.svg?style=flat)](https://github.com/Muhammad-Ibraheem-kamboh/MindPulse-AI)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> 🌐 **Try the Live Web Studio & Proctor Portal**:  
> 👉 [**https://muhammad-ibraheem-kamboh.github.io/MindPulse-AI/**](https://muhammad-ibraheem-kamboh.github.io/MindPulse-AI/)

> **Blinkr** is an autonomous video proctoring, computer vision, and cognitive ergonomics platform designed for examiners, universities, and students. It combines **live WebRTC video proctoring**, **visual identity verification**, **6DoF 3D spatial head pose estimation**, **real-time biometrics (PERCLOS & MAR)**, **anti-cheat tab tracking**, and **certified printable audit certificates**.

---

## 🌟 Key Features

### 1. 🌐 6DoF 3D Spatial Head Pose & Live Wireframe Gizmo
* **Perspective-n-Point (`cv2.solvePnP`)**: Solves head rotation and translation against a calibrated 3D anthropometric skull model.
* **Euler Angles (Pitch, Yaw, Roll)**:
  * **Pitch (Nodding / Slouching)**: Detects poor desk ergonomics and downward head droop.
  * **Yaw (Turning Left/Right)**: Quantifies off-screen visual divergence.
  * **Roll (Lateral Tilt)**: Detects asymmetrical head resting.
* **Real-Time 3D Wireframe Head Gizmo**: An interactive 3D faceted head model rendered inside the dashboard viewport that rotates in 3D perspective mirroring your head movements.
* **Face-Projected 3D Coordinate Axes**: Direct facial projection of spatial axes ($X$-Red, $Y$-Green, $Z$-Blue) onto your nose tip.

### 2. ⚡ Live Animated ECG/EEG Focus Waveform
* Real-time rolling waveform graph (scrolling across 140 temporal data points).
* Captures instantaneous cognitive engagement fluctuations and micro-rhythms in a futuristic neon medical bio-monitor display.

### 3. 🥱 MAR (Mouth Aspect Ratio) Yawn Detection
* Mathematical mouth aperture analysis tracking inner/outer lips:
  $$\text{MAR} = \frac{\|p_{13} - p_{14}\| + \|p_{82} - p_{87}\| + \|p_{312} - p_{317}\|}{3 \cdot \|p_{61} - p_{291}\|}$$
* Triggers an **early proactive fatigue warning** upon sustained yawning ($\text{MAR} \ge 0.58$ for $>1.2\text{s}$) before eyes ever close.

### 4. 📊 60-Second Rolling PERCLOS Fatigue Index
* Implements the **NASA & FAA gold standard** for physiological drowsiness assessment:
  $$\text{PERCLOS} = \frac{\sum \text{Frames with closed / drooping eyes}}{\text{Total frames in rolling 60-second window}} \times 100\%$$
* Visualized as a live ergonomic safety gauge:
  * **$< 12\%$**: 🟢 Alert / Optimal Flow
  * **$12\% - 18\%$**: 🟡 Mild Fatigue / Eye Strain
  * **$> 18\%$**: 🔴 Critical Fatigue / Microsleep Hazard

### 5. 👁️ Iris Gaze Vector Tracking
* Bilateral eye pupil position tracking relative to medial and lateral canthi (eye corners).
* Intelligently distinguishes between direct on-screen attention and wandering gaze (e.g. glancing down at a smartphone or lap while keeping head upright).

### 6. ⏱️ Physiological Blink Rate Counter (BPM)
* Filters micro-saccades and validates natural physiological blinks ($80\text{ms} - 450\text{ms}$).
* Measures active **Blinks Per Minute (BPM)** over a rolling 60s window to prevent digital asthenopia and eye dryness.

### 7. 🎨 Obsidian Glassmorphic HUD & Mouse Clickable UI
* **High-DPI Anti-Aliased Typography**: Rendered via PIL RGBA subpixel compositing using Microsoft Segoe UI.
* **Clickable On-Screen Buttons**:
  * `[SOUND: MUTED / ON]`: Interactive sound toggle (silent by default).
  * `[CALIBRATE]`: One-click 3-second eye baseline calibration.
  * `[ZEN MODE]`: Minimalist full-screen video distraction-free view.
  * `[BLUR: OFF / ON]`: Real-time background privacy blur.
* **Zero-CMD Launch**: Starts silently with zero command prompt console window via Windows `pythonw.exe`.

---

## 🔒 Cybersecurity & Privacy-First Architecture

* **100% On-Device AI**: Zero cloud telemetry, zero external socket connections, zero video frame logging.
* **Cryptographic Model Pinning**: Official Google MediaPipe models are verified with SHA-256 hashes before execution:
  * `face_landmarker.task`: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`
  * `efficientdet_lite0.tflite`: `0720bf247bd76e6594ea28fa9c6f7c5242be774818997dbbeffc4da460c723bb`
* **Ephemeral Memory Buffer**: Video frames live exclusively in volatile RAM and are overwritten every frame.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10, 3.11, 3.12, or 3.13
* Webcam (Integrated or USB)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Muhammad-Ibraheem-kamboh/MindPulse-AI.git
   cd MindPulse-AI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Studio**:
   * **Silent Mode (No Console Window)**:
     ```bash
     pythonw student_monitor.pyw
     ```
   * **Console Mode (With Logs)**:
     ```bash
     python student_monitor.py
     ```

---

## ⌨️ Controls & Shortcuts

| Control | Action | Function |
| :---: | :---: | :--- |
| **`[SOUND]` Button / `M`** | **Audio Mute/Unmute** | Toggles gentle psychoacoustic chimes (Default: MUTED). |
| **`[CALIBRATE]` / `C`** | **Calibrate Baseline** | 3-second open-eye calibration to set personalized EAR threshold. |
| **`[ZEN MODE]` / `H`** | **View Toggle** | Switches between full analytics dashboard and distraction-free view. |
| **`[BLUR]` / `V`** | **Privacy Blur** | Blurs camera feed background for meeting/screen-share privacy. |
| **`P`** | **Pomodoro Sprint** | Pauses or resumes the 25-minute deep work / 5-minute break cycle. |
| **`Q` / `Esc`** | **Graceful Exit** | Releases hardware camera locks and cleanly exits. |

---

## 📦 Project Structure

```
MindPulse-AI/
├── models/
│   ├── face_landmarker.task      # MediaPipe Face Landmarker model
│   └── efficientdet_lite0.tflite # Mobile phone object detector
├── student_monitor.pyw           # Silent studio application (GUI)
├── student_monitor.py            # Console-enabled application runner
├── requirements.txt              # Production dependency manifest
├── LICENSE                       # MIT Open Source License
└── README.md                     # Documentation & Architecture Guide
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Developed with ❤️ by **[Muhammad Ibraheem](https://github.com/Muhammad-Ibraheem-kamboh)**.
