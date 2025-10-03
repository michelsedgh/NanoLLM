#!/usr/bin/env python3
"""Run MoViNet streaming action recognition on live webcam using TF Hub model."""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


KINETICS_600_LABELS_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/"
    "f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/"
    "official/projects/movinet/files/kinetics_600_labels.txt"
)


class MoViNetTFHubStreamer:
    """Stream MoViNet A5 predictions from a live webcam feed."""

    def __init__(self, model_id: str = "a5", resolution: int = 320, top_k: int = 5) -> None:
        self.model_id = model_id
        self.resolution = resolution
        self.top_k = top_k

        self.model_url = (
            f"https://tfhub.dev/tensorflow/movinet/{model_id}/stream/"
            "kinetics-600/classification/3"
        )

        print(f"📥 Loading MoViNet {model_id.upper()} streaming model from TF Hub…")
        self.model = hub.load(self.model_url)
        print("✅ Model loaded")

        print("📑 Downloading Kinetics-600 labels…")
        labels_path = tf.keras.utils.get_file(
            fname="kinetics_600_labels.txt",
            origin=KINETICS_600_LABELS_URL,
        )
        self.labels = self._load_labels(Path(labels_path))
        print(f"✅ Loaded {len(self.labels)} labels")

        # Pre-compute the input shape tensor for init_states
        self.input_shape_tensor = tf.constant(
            [1, 1, self.resolution, self.resolution, 3], dtype=tf.int32
        )

    @staticmethod
    def _load_labels(path: Path) -> List[str]:
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    def _open_camera(self, camera_id: int) -> cv2.VideoCapture:
        # Jetson Orin Nano: USB webcam via GStreamer with HW MJPEG decode and GPU resize.
        # No fallbacks: either this works or we fail fast.
        res = int(self.resolution)
        gst = (
            f"v4l2src device=/dev/video{camera_id} io-mode=2 ! "
            f"image/jpeg, framerate=(fraction)30/1 ! "
            f"jpegparse ! "
            f"nvv4l2decoder mjpeg=1 ! "
            f"nvvidconv ! video/x-raw, format=(string)BGRx, width=(int){res}, height=(int){res} ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )
        cam = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if not cam.isOpened():
            raise RuntimeError(
                "Failed to open USB webcam via GStreamer. Ensure MJPEG is supported and GStreamer/NVIDIA plugins are installed."
            )
        ok, frame = cam.read()
        if not ok or frame is None or frame.size == 0:
            cam.release()
            raise RuntimeError("GStreamer pipeline opened but did not deliver a valid frame.")
        print("✅ Opened USB camera via GStreamer (MJPEG→nvv4l2decoder, nvvidconv)")
        return cam

    def _preprocess(self, frame: np.ndarray) -> tf.Tensor:
        # Frames arrive already resized to (resolution x resolution) via GStreamer
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = tf.convert_to_tensor(tensor)
        tensor = tf.expand_dims(tf.expand_dims(tensor, 0), 0)  # [1, 1, H, W, 3]
        return tensor

    def _format_predictions(self, logits: tf.Tensor) -> List[Tuple[str, float]]:
        probs = tf.nn.softmax(logits[0], axis=-1).numpy()
        top_indices = probs.argsort()[::-1][: self.top_k]
        return [(self.labels[i], float(probs[i])) for i in top_indices]

    def run(self, camera_id: int = 0) -> None:
        cap = self._open_camera(camera_id)
        window_name = "MoViNet TF Hub Streaming (press q to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError("Failed to read initial frame from webcam.")

        frame_tensor = self._preprocess(frame)
        states = self.model.init_states(self.input_shape_tensor)

        # Warm-up run to ensure graph is traced
        warm_inputs: Dict[str, tf.Tensor] = dict(states)
        warm_inputs["image"] = frame_tensor
        warm_logits, states = self.model(warm_inputs)
        print("🔥 Warm-up completed")

        fps_history: List[float] = []
        frame_counter = 0
        start_time = time.time()

        try:
            while True:
                iter_start = time.time()
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("❌ Failed to read frame from webcam")
                    break

                tensor = self._preprocess(frame)
                inputs = dict(states)
                inputs["image"] = tensor
                logits, states = self.model(inputs)

                predictions = self._format_predictions(logits)

                elapsed = time.time() - iter_start
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                fps_history.append(fps)
                if len(fps_history) > 60:
                    fps_history.pop(0)
                avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0

                display = frame.copy()
                cv2.putText(
                    display,
                    f"FPS: {fps:.1f} (avg {avg_fps:.1f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                for idx, (label, prob) in enumerate(predictions):
                    text = f"#{idx + 1} {label}: {prob:.3f}"
                    y = 60 + idx * 25
                    cv2.putText(
                        display,
                        text,
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                cv2.imshow(window_name, display)

                frame_counter += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            total_time = time.time() - start_time
            avg = frame_counter / total_time if total_time > 0 else 0.0
            peak = max(fps_history) if fps_history else 0.0
            print("\n📊 Session summary")
            print(f"Frames processed: {frame_counter}")
            print(f"Elapsed time: {total_time:.2f}s")
            print(f"Average FPS: {avg:.2f}")
            print(f"Peak FPS: {peak:.2f}")


def sys_platform_is_mac() -> bool:
    import sys

    return sys.platform == "darwin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MoViNet streaming action recognition from TF Hub on a webcam"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to show")
    parser.add_argument(
        "--resolution",
        type=int,
        default=320,
        help="Square resolution used for model input (default: 320)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    streamer = MoViNetTFHubStreamer(
        model_id="a5", resolution=args.resolution, top_k=args.topk
    )
    streamer.run(camera_id=args.camera)


if __name__ == "__main__":
    main()

