#!/usr/bin/env python3
"""
Human Following Robot using YOLOv8n + BotSort tracking and ODrive motors.
Follows the first detected person using differential drive.
Before following: user spins 360° to build a ReID gallery for robust re-identification.

Controls:
- 'q' to quit
- 'r' to reset target (lock onto new person)
- 's' to toggle following on/off (safety stop)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import odrive
from odrive.enums import AXIS_STATE_IDLE
from ultralytics import YOLO


# ────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────

@dataclass
class MotorConfig:
    left_direction: int = 1
    right_direction: int = -1
    left_serial: str = "325735623133"
    right_serial: str = "306F388B3533"
    max_velocity: float = 1.5
    min_velocity: float = 0.22


@dataclass
class SteeringConfig:
    turn_gain: float = 1.1
    speed_gain: float = 2.0
    steer_inner: float = 0.10
    steer_ramp_end: float = 0.28
    steer_brake_extra: float = 0.10
    small_turn_boost: float = 1.5
    large_turn_damp: float = 0.65
    small_turn_thresh: float = 0.16
    large_turn_thresh: float = 0.22
    max_turn_diff: float = 2.8
    forward_time_constant: float = 0.04  # seconds; lower = snappier forward (dt-aware)
    # Gradual ramp-up / early-stop (units per second)
    turn_ramp_up_rate: float = 10.0      # ramp-up (~0.3 s zero→full)
    turn_ramp_down_rate: float = 15.0    # fast brake (~0.2 s full→zero)
    approach_brake_gain: float = 2.0     # scale down turn when error is shrinking


@dataclass
class DistanceConfig:
    target_bbox_height_ratio: float = 0.9
    distance_deadband: float = 0.05
    too_close_ratio: float = 0.78
    target_depth_meters: float = 1.5
    depth_deadband_meters: float = 0.25
    too_close_depth_meters: float = 0.6
    depth_gain: float = 0.8


@dataclass
class CameraConfig:
    use_realsense: bool = True
    camera_index: int = 1
    frame_width: int = 640
    frame_height: int = 480
    backend: int = field(default_factory=lambda: cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)


@dataclass
class ReIDConfig:
    gallery_dir: str = "reid_gallery"
    min_samples: int = 24
    sample_interval_frames: int = 4
    match_threshold: float = 0.6
    input_size: int = 224


@dataclass
class TrackingConfig:
    lost_track_timeout: float = 1.5


@dataclass
class Config:
    motor: MotorConfig = field(default_factory=MotorConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


# ────────────────────────────────────────────
# Motor Controller
# ────────────────────────────────────────────

class MotorController:
    """Manages two ODrive motor controllers for a differential-drive robot."""

    WHEEL_DEADBAND = 0.10

    def __init__(self, cfg: MotorConfig, steering_cfg: SteeringConfig):
        self._cfg = cfg
        self._steer = steering_cfg
        self._odrv_left = None
        self._odrv_right = None
        self._last_time = time.time()
        self.current_left_vel = 0.0
        self.current_right_vel = 0.0

    @property
    def connected(self) -> bool:
        return self._odrv_left is not None and self._odrv_right is not None

    def connect(self):
        print("Finding ODrive controllers...")
        print("  Looking for left motor...")
        self._odrv_left = odrive.find_any(serial_number=self._cfg.left_serial)
        print("  Looking for right motor...")
        self._odrv_right = odrive.find_any(serial_number=self._cfg.right_serial)
        print("✓ ODrive controllers found!")

        time.sleep(1)

        print("Configuring motors for velocity control...")
        for odrv in (self._odrv_left, self._odrv_right):
            odrv.axis0.requested_state = 8  # CLOSED_LOOP_CONTROL
            time.sleep(0.25)
            odrv.axis0.controller.config.control_mode = 2  # VELOCITY_CONTROL
            odrv.axis0.controller.config.input_mode = 1    # DIRECT
            odrv.axis0.controller.input_vel = 0

        print("✓ Motors configured")

    def stop(self):
        if not self.connected:
            return
        self._odrv_left.axis0.controller.input_vel = 0
        self._odrv_right.axis0.controller.input_vel = 0
        self.current_left_vel = 0.0
        self.current_right_vel = 0.0

    def set_velocities(self, left_vel: float, right_vel: float) -> tuple[float, float]:
        """Smooth forward component (dt-aware), pass turn straight through, then send.

        Turn smoothing is handled upstream by SteeringController's ramp system.
        Returns the actual (left, right) velocities sent.
        """
        now = time.time()
        dt = max(0.001, now - self._last_time)
        self._last_time = now

        left_vel = np.clip(left_vel, -self._cfg.max_velocity, self._cfg.max_velocity)
        right_vel = np.clip(right_vel, -self._cfg.max_velocity, self._cfg.max_velocity)

        cmd_fwd = (left_vel + right_vel) * 0.5
        cmd_turn = (right_vel - left_vel) * 0.5
        cur_fwd = (self.current_left_vel + self.current_right_vel) * 0.5

        tau = self._steer.forward_time_constant
        alpha = 1.0 - np.exp(-dt / tau) if tau > 0 else 1.0
        new_fwd = cur_fwd + alpha * (cmd_fwd - cur_fwd)

        self.current_left_vel = new_fwd - cmd_turn
        self.current_right_vel = new_fwd + cmd_turn

        final_left = self._apply_deadband(self.current_left_vel)
        final_right = self._apply_deadband(self.current_right_vel)

        self._odrv_left.axis0.controller.input_vel = final_left * self._cfg.left_direction
        self._odrv_right.axis0.controller.input_vel = final_right * self._cfg.right_direction
        return final_left, final_right

    def _apply_deadband(self, vel: float) -> float:
        """Kill very small values, otherwise bump up to min_velocity for the motor."""
        if abs(vel) < self.WHEEL_DEADBAND:
            return 0.0
        if abs(vel) < self._cfg.min_velocity:
            return self._cfg.min_velocity * np.sign(vel)
        return vel

    def shutdown(self):
        if not self.connected:
            return
        self.stop()
        time.sleep(0.3)
        self._odrv_left.axis0.requested_state = AXIS_STATE_IDLE
        self._odrv_right.axis0.requested_state = AXIS_STATE_IDLE
        print("✓ Motors set to idle")


# ────────────────────────────────────────────
# Camera
# ────────────────────────────────────────────

class Camera:
    """Abstracts RealSense (color+depth) and plain webcam behind one interface."""

    def __init__(self, cfg: CameraConfig):
        self._cfg = cfg
        self._pipeline = None
        self._align = None
        self._cap = None

    def open(self):
        if self._cfg.use_realsense:
            self._open_realsense()
        else:
            self._open_webcam()

    def _open_realsense(self):
        import pyrealsense2 as rs

        print("Opening RealSense (color + depth)...")
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self._cfg.frame_width, self._cfg.frame_height, rs.format.rgb8, 30)
        cfg.enable_stream(rs.stream.depth, self._cfg.frame_width, self._cfg.frame_height, rs.format.z16, 30)
        self._pipeline = rs.pipeline()
        self._pipeline.start(cfg)
        self._align = rs.align(rs.stream.color)
        print("✓ RealSense initialized")

    def _open_webcam(self):
        idx = self._cfg.camera_index
        backend_name = "DirectShow" if self._cfg.backend == cv2.CAP_DSHOW else "default"
        print(f"Opening webcam (index={idx}, backend={backend_name})...")
        self._cap = cv2.VideoCapture(idx, self._cfg.backend)
        if not self._cap.isOpened():
            raise RuntimeError("Could not open webcam. Try another CAMERA_INDEX or check USB.")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.frame_height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("✓ Webcam initialized")

    def read_frame(self) -> tuple[bool, np.ndarray | None, object | None]:
        """Returns (success, bgr_frame, depth_frame_or_None)."""
        if self._pipeline is not None:
            return self._read_realsense()
        return self._read_webcam()

    def _read_realsense(self):
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            aligned = self._align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame:
                return False, None, None
            img = np.asanyarray(color_frame.get_data())
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return True, bgr, depth_frame
        except Exception as e:
            print(f"⚠ RealSense frame read error: {e}")
            return False, None, None

    def _read_webcam(self):
        ret, frame = self._cap.read()
        return ret, frame, None

    def close(self):
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        if self._cap is not None:
            self._cap.release()


# ────────────────────────────────────────────
# ReID Engine
# ────────────────────────────────────────────

class ReIDEngine:
    """Person re-identification via ResNet-18 embeddings and a gallery of crops."""

    def __init__(self, cfg: ReIDConfig):
        self._cfg = cfg
        self._model = None
        self._transform = None
        self.gallery_embeddings: np.ndarray | None = None
        self.gallery_mean: np.ndarray | None = None

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from torchvision import models, transforms

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self._model = torch.nn.Sequential(*(list(backbone.children())[:-1]))
        self._model.eval()
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self._cfg.input_size, self._cfg.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def embed_crop(self, crop_rgb: np.ndarray) -> np.ndarray | None:
        """Compute L2-normalised embedding for a person crop (RGB, any size)."""
        import torch

        self._ensure_model()
        if crop_rgb is None or crop_rgb.size == 0:
            return None
        x = self._transform(crop_rgb).unsqueeze(0)
        with torch.no_grad():
            emb = self._model(x)
        emb = emb.flatten().numpy().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb /= norm
        return emb

    def load_gallery(self):
        path = os.path.join(self._cfg.gallery_dir, "embeddings.npy")
        if not os.path.isfile(path):
            self.gallery_mean = None
            return
        embs = np.load(path)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        self.gallery_embeddings = embs
        self._compute_mean(embs)
        print(f"✓ ReID gallery loaded ({len(embs)} samples)")

    def save_gallery(self, embeddings: list[np.ndarray]):
        os.makedirs(self._cfg.gallery_dir, exist_ok=True)
        stacked = np.stack(embeddings)
        path = os.path.join(self._cfg.gallery_dir, "embeddings.npy")
        np.save(path, stacked)
        self.gallery_embeddings = stacked
        self._compute_mean(stacked)
        print(f"✓ ReID gallery saved ({len(stacked)} samples) to {path}")

    def best_match_index(self, boxes, frame: np.ndarray) -> tuple[int | None, float]:
        """Return (index_into_boxes, similarity) of the best gallery match, or (None, -1)."""
        if self.gallery_mean is None:
            return None, -1.0
        best_sim = -1.0
        best_idx = None
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            crop = crop_person_bbox(frame, xyxy)
            if crop is None:
                continue
            emb = self.embed_crop(crop)
            if emb is None:
                continue
            sim = float(np.dot(emb, self.gallery_mean))
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        return best_idx, best_sim

    @property
    def has_gallery(self) -> bool:
        return self.gallery_mean is not None

    def _compute_mean(self, embs: np.ndarray):
        mean = embs.mean(axis=0).astype(np.float32)
        norm = np.linalg.norm(mean)
        if norm > 1e-6:
            mean /= norm
        self.gallery_mean = mean

    def run_enrollment(self, camera: Camera, detector: YOLO):
        """Interactive 360-degree capture loop. Stores gallery on completion."""
        self._ensure_model()
        phase = "wait"
        gallery: list[np.ndarray] = []
        frame_count = 0
        cam_cfg = camera._cfg

        cv2.namedWindow("Human Following Robot", cv2.WINDOW_NORMAL)
        print("\n--- ReID enrollment: spin 360° ---")
        print("  Position yourself in frame. Press SPACE to start capture.")
        print(f"  Then spin slowly 360°. Press ENTER when done (or auto-stop at {self._cfg.min_samples} samples).\n")

        while True:
            ret, frame, _ = camera.read_frame()
            if not ret or frame is None:
                continue
            frame_count += 1

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (cam_cfg.frame_width, 120), (0, 0, 0), -1)

            if phase == "wait":
                cv2.putText(overlay, "Press SPACE to start 360 deg capture", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(overlay, "Q = quit without enrolling", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            elif phase == "capturing":
                cv2.putText(overlay, "Spin slowly 360 deg - Press ENTER when done", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(overlay, f"Samples: {len(gallery)} / {self._cfg.min_samples}", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            results = detector.predict(frame, verbose=False, classes=[0])
            if phase == "capturing" and results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0 and frame_count % self._cfg.sample_interval_frames == 0:
                    crop = crop_person_bbox(frame, boxes.xyxy[0].cpu().numpy())
                    if crop is not None:
                        emb = self.embed_crop(crop)
                        if emb is not None:
                            gallery.append(emb)
                            print(f"  Captured sample {len(gallery)}")

            cv2.imshow("Human Following Robot", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') and phase == "wait":
                print("Enrollment skipped.")
                return
            if key == ord(' ') and phase == "wait":
                phase = "capturing"
                frame_count = 0
            if key in (13, 10) and phase == "capturing":
                break
            if phase == "capturing" and len(gallery) >= self._cfg.min_samples:
                break

        if len(gallery) < 4:
            print("Not enough samples (need at least 4). ReID gallery not saved.")
            return
        self.save_gallery(gallery)


# ────────────────────────────────────────────
# Steering Controller
# ────────────────────────────────────────────

class SteeringController:
    """Translates target position + distance into (left_vel, right_vel).

    Maintains state across frames for gradual ramp-up and predictive braking.
    """

    def __init__(self, steer_cfg: SteeringConfig, dist_cfg: DistanceConfig, motor_cfg: MotorConfig):
        self._s = steer_cfg
        self._d = dist_cfg
        self._m = motor_cfg
        self._turn_effort = 0.0
        self._prev_error: float | None = None
        self._prev_time: float = time.time()

    def reset(self):
        """Clear accumulated turn state (call when target changes)."""
        self._turn_effort = 0.0
        self._prev_error = None
        self._prev_time = time.time()

    def compute(self, center_x: float, height_ratio: float, depth_m: float | None = None) -> tuple[float, float]:
        """Return (left_vel, right_vel) for the current target position/distance."""
        now = time.time()
        dt = max(0.001, now - self._prev_time)
        self._prev_time = now

        steering_error = center_x - 0.5
        abs_err = abs(steering_error)

        # --- proportional turn target (same ramp / boost / damp as before) ---
        ramp_frac = min(1.0, max(0, (abs_err - self._s.steer_inner) / (self._s.steer_ramp_end - self._s.steer_inner)))
        effective_inner = self._s.steer_inner + self._s.steer_brake_extra * ramp_frac
        effective_inner = min(effective_inner, self._s.steer_ramp_end - 0.06)

        if abs_err <= effective_inner:
            turn_scale = 0.0
        else:
            turn_scale = min(1.0, (abs_err - effective_inner) / (self._s.steer_ramp_end - effective_inner))

        if abs_err < self._s.small_turn_thresh and abs_err > effective_inner:
            turn_scale = min(1.0, turn_scale * self._s.small_turn_boost)
        elif abs_err > self._s.large_turn_thresh:
            turn_scale *= self._s.large_turn_damp

        desired_turn = steering_error * self._s.turn_gain * turn_scale
        desired_turn = np.clip(desired_turn, -self._s.max_turn_diff, self._s.max_turn_diff)

        # --- predictive braking: reduce desired turn when approaching center ---
        if self._prev_error is not None and dt > 0:
            d_error = (steering_error - self._prev_error) / dt
            approaching = steering_error * d_error < 0
            if approaching:
                brake = max(0.0, 1.0 - abs(d_error) * self._s.approach_brake_gain)
                desired_turn *= brake
        self._prev_error = steering_error

        # --- gradual ramp: slowly increase, quickly decrease ---
        diff = desired_turn - self._turn_effort
        ramping_up = (abs(desired_turn) > abs(self._turn_effort)
                      and desired_turn * self._turn_effort >= 0)
        max_step = (self._s.turn_ramp_up_rate if ramping_up else self._s.turn_ramp_down_rate) * dt
        if abs(diff) <= max_step:
            self._turn_effort = desired_turn
        else:
            self._turn_effort += np.sign(diff) * max_step

        turn_diff = self._turn_effort

        # --- forward / backward speed ---
        forward_speed = self._compute_forward(height_ratio, depth_m)
        if forward_speed is None:
            return 0.0, 0.0

        if forward_speed < 0:
            forward_speed = max(forward_speed, -self._m.max_velocity * 0.4)
        else:
            forward_speed = min(forward_speed, self._m.max_velocity)

        if abs(forward_speed) < 0.1 and abs(turn_diff) > 0.1:
            return -turn_diff * 0.8, turn_diff * 0.8

        return forward_speed - turn_diff, forward_speed + turn_diff

    def _compute_forward(self, height_ratio: float, depth_m: float | None) -> float | None:
        """Return forward speed component; None means emergency stop."""
        use_depth = depth_m is not None and 0.3 < depth_m < 10.0
        if use_depth:
            if depth_m < self._d.too_close_depth_meters:
                return None
            error = depth_m - self._d.target_depth_meters
            if abs(error) < self._d.depth_deadband_meters:
                error = 0.0
            return error * self._d.depth_gain

        if height_ratio > self._d.too_close_ratio:
            return None
        error = self._d.target_bbox_height_ratio - height_ratio
        if abs(error) < self._d.distance_deadband:
            error = 0.0
        return error * self._s.speed_gain

    def distance_status(self, height_ratio: float, depth_m: float | None) -> str:
        use_depth = depth_m is not None and 0.3 < depth_m < 10.0
        if use_depth:
            if depth_m < self._d.too_close_depth_meters:
                return "TOO CLOSE"
            if depth_m < self._d.target_depth_meters - self._d.depth_deadband_meters:
                return "CLOSE"
            if depth_m > self._d.target_depth_meters + self._d.depth_deadband_meters:
                return "TOO FAR"
            return "GOOD"
        if height_ratio > self._d.too_close_ratio:
            return "TOO CLOSE"
        if height_ratio > self._d.target_bbox_height_ratio + self._d.distance_deadband:
            return "CLOSE"
        if height_ratio < self._d.target_bbox_height_ratio - self._d.distance_deadband:
            return "TOO FAR"
        return "GOOD"


# ────────────────────────────────────────────
# UI Renderer
# ────────────────────────────────────────────

@dataclass
class StatusInfo:
    tracking: bool = False
    track_id: int | None = None
    left_vel: float = 0.0
    right_vel: float = 0.0
    distance_status: str = "NO TARGET"
    height_ratio: float = 0.0
    depth_m: float | None = None


class UIRenderer:
    """Draws HUD overlays on the camera frame."""

    def __init__(self, cam_cfg: CameraConfig, center_deadband: float):
        self._w = cam_cfg.frame_width
        self._h = cam_cfg.frame_height
        self._deadband = center_deadband

    def draw(self, frame: np.ndarray, results, status: StatusInfo, following: bool) -> np.ndarray:
        annotated = results[0].plot() if results and len(results) > 0 else frame.copy()

        cv2.rectangle(annotated, (0, 0), (self._w, 90), (0, 0, 0), -1)

        color = (0, 255, 0) if following else (0, 0, 255)
        label = "FOLLOWING" if following else "STOPPED"
        cv2.putText(annotated, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if status.tracking:
            cv2.putText(annotated, f"Target ID: {status.track_id or '?'}", (200, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(annotated, f"Motors L:{status.left_vel:+.2f} R:{status.right_vel:+.2f}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        dist_str = f"Distance: {status.distance_status}"
        if status.depth_m is not None:
            dist_str += f"  Depth: {status.depth_m:.2f}m"
        else:
            dist_str += f"  Height: {status.height_ratio:.2f}"
        dist_color = {"TOO CLOSE": (0, 0, 255), "TOO FAR": (0, 165, 255)}.get(status.distance_status, (0, 255, 255))
        cv2.putText(annotated, dist_str, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dist_color, 2)

        cx = self._w // 2
        cv2.line(annotated, (cx, 90), (cx, self._h), (100, 100, 100), 1)
        zl = int(self._w * (0.5 - self._deadband))
        zr = int(self._w * (0.5 + self._deadband))
        cv2.rectangle(annotated, (zl, 90), (zr, self._h), (0, 255, 0), 2)

        cv2.putText(annotated, "Q:quit R:reset S:stop", (self._w - 200, self._h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        return annotated


# ────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────

def crop_person_bbox(frame: np.ndarray, xyxy, padding: float = 0.1) -> np.ndarray | None:
    """Crop frame to bbox with optional padding. Returns RGB (H,W,3) for ReID."""
    x1, y1, x2, y2 = map(int, xyxy)
    h, w = frame.shape[:2]
    pad_w = int((x2 - x1) * padding)
    pad_h = int((y2 - y1) * padding)
    x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
    x2, y2 = min(w, x2 + pad_w), min(h, y2 + pad_h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def extract_box_info(box, frame_w: int, frame_h: int, depth_frame=None) -> tuple[float, float, int | None, float | None]:
    """Return (center_x_norm, height_ratio, track_id_or_None, depth_m_or_None)."""
    xyxy = box.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = xyxy
    center_x = ((x1 + x2) / 2) / frame_w
    height_ratio = (y2 - y1) / frame_h
    depth_m = None
    if depth_frame is not None:
        cx_px, cy_px = int((x1 + x2) / 2), int((y1 + y2) / 2)
        d = depth_frame.get_distance(cx_px, cy_px)
        if 0.3 < d < 10.0:
            depth_m = float(d)
    return center_x, height_ratio, None, depth_m


# ────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────

class HumanFollower:
    """Top-level orchestrator that wires together all subsystems."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.motors = MotorController(self.cfg.motor, self.cfg.steering)
        self.camera = Camera(self.cfg.camera)
        self.reid = ReIDEngine(self.cfg.reid)
        self.steering = SteeringController(self.cfg.steering, self.cfg.distance, self.cfg.motor)
        self.ui = UIRenderer(self.cfg.camera, self.cfg.steering.steer_inner)
        self.detector: YOLO | None = None
        self.target_id: int | None = None
        self.last_detection_time = time.time()
        self.following_enabled = True

    def _init_detector(self):
        print("Loading YOLOv8n model...")
        self.detector = YOLO("yolov8n.pt")

    def find_target(self, results, frame: np.ndarray | None = None,
                    depth_frame=None) -> tuple | None:
        """Locate the target person in detection results.

        Returns (center_x, height_ratio, track_id, depth_m) or None.
        """
        if not results or len(results) == 0:
            return None
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        fw = self.cfg.camera.frame_width
        fh = self.cfg.camera.frame_height

        if self.reid.has_gallery and frame is not None:
            idx, sim = self.reid.best_match_index(boxes, frame)
            if idx is not None and sim >= self.cfg.reid.match_threshold:
                tid = int(boxes.id[idx].cpu().numpy()) if boxes.id is not None else None
                self.target_id = tid
                cx, hr, _, dm = extract_box_info(boxes[idx], fw, fh, depth_frame)
                return cx, hr, tid, dm
            return None

        if self.target_id is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy()
            for i in range(len(boxes)):
                if int(ids[i]) == self.target_id:
                    cx, hr, _, dm = extract_box_info(boxes[i], fw, fh, depth_frame)
                    return cx, hr, self.target_id, dm

        if boxes.id is not None:
            tid = int(boxes.id.cpu().numpy()[0])
            self.target_id = tid
            print(f"🎯 Locked onto person ID: {tid}")
            cx, hr, _, dm = extract_box_info(boxes[0], fw, fh, depth_frame)
            return cx, hr, tid, dm

        cx, hr, _, dm = extract_box_info(boxes[0], fw, fh, depth_frame)
        return cx, hr, None, dm

    def run(self):
        """Main control loop."""
        try:
            self.motors.connect()
            self._init_detector()
            self.camera.open()

            self.reid.run_enrollment(self.camera, self.detector)
            if not self.reid.has_gallery:
                self.reid.load_gallery()

            self._print_banner()
            self._main_loop()

        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            raise
        finally:
            self._shutdown()

    def _print_banner(self):
        print("\n" + "=" * 50)
        print("  HUMAN FOLLOWING ROBOT")
        print("=" * 50)
        print("Controls:")
        print("  Q - Quit")
        print("  R - Reset target (lock onto new person)")
        print("  S - Toggle following on/off")
        print("=" * 50 + "\n")

    def _main_loop(self):
        while True:
            ret, frame, depth_frame = self.camera.read_frame()
            if not ret or frame is None:
                print("⚠ Failed to read frame")
                continue

            results = self.detector.track(frame, tracker="botsort.yaml", persist=True, verbose=False, classes=[0])
            status = StatusInfo()

            target = self.find_target(results, frame, depth_frame)

            if target and self.following_enabled:
                cx, hr, tid, dm = target
                self.last_detection_time = time.time()
                status.distance_status = self.steering.distance_status(hr, dm)
                left, right = self.steering.compute(cx, hr, dm)
                al, ar = self.motors.set_velocities(left, right)
                status.tracking = True
                status.track_id = tid
                status.left_vel = al
                status.right_vel = ar
                status.height_ratio = hr
                status.depth_m = dm
            else:
                elapsed = time.time() - self.last_detection_time
                if elapsed > self.cfg.tracking.lost_track_timeout:
                    self.motors.stop()
                    self.steering.reset()
                    if self.target_id is not None:
                        print(f"⚠ Lost track of person {self.target_id}")
                        self.target_id = None
                    status.distance_status = "LOST"
                else:
                    status.distance_status = "SEARCHING"
                    status.left_vel = self.motors.current_left_vel
                    status.right_vel = self.motors.current_right_vel

            if not self.following_enabled:
                self.motors.stop()
                status.left_vel = 0.0
                status.right_vel = 0.0

            display = self.ui.draw(frame, results, status, self.following_enabled)
            cv2.imshow("Human Following Robot", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('r'):
                self.target_id = None
                self.steering.reset()
                print("🔄 Target reset - will lock onto next person")
            elif key == ord('s'):
                self.following_enabled = not self.following_enabled
                if not self.following_enabled:
                    self.motors.stop()
                    self.steering.reset()
                print(f"Following {'ENABLED' if self.following_enabled else 'DISABLED'}")

    def _shutdown(self):
        print("\nShutting down safely...")
        self.motors.stop()
        time.sleep(0.3)
        self.motors.shutdown()
        self.camera.close()
        cv2.destroyAllWindows()
        print("✓ Shutdown complete")


# ────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Human Following Robot v2.0")
    print("  YOLOv8n + BotSort + ODrive Differential Drive")
    print("=" * 50 + "\n")
    HumanFollower().run()


if __name__ == "__main__":
    main()
