import cv2
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import logging
import time
from PIL import Image
import sys
from threading import Thread, Lock
from queue import Queue
from collections import deque
import os
import tempfile
from datetime import datetime
from pathlib import Path

def setup_logging():
    """Configure logging with basic formatting"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class VideoRingBuffer:
    """Time-based ring buffer holding the last N seconds of frames."""
    def __init__(self, max_duration_sec=30.0, target_width=None, target_height=None):
        self.max_duration_sec = float(max_duration_sec)
        self.target_width = target_width
        self.target_height = target_height
        self.frames = deque()  # each item: (timestamp, frame_bgr_resized)
        self.lock = Lock()

    def _resize_frame(self, frame):
        if self.target_width is None and self.target_height is None:
            return frame
        h, w = frame.shape[:2]
        if self.target_width is not None and (self.target_height is None):
            scale = self.target_width / float(w)
            new_w = self.target_width
            new_h = int(round(h * scale))
        elif self.target_height is not None and (self.target_width is None):
            scale = self.target_height / float(h)
            new_h = self.target_height
            new_w = int(round(w * scale))
        else:
            new_w, new_h = self.target_width, self.target_height
        return cv2.resize(frame, (new_w, new_h))

    def append(self, frame_bgr, timestamp_s):
        resized = self._resize_frame(frame_bgr)
        with self.lock:
            self.frames.append((timestamp_s, resized))
            self._trim_locked(current_time_s=timestamp_s)

    def _trim_locked(self, current_time_s):
        cutoff = current_time_s - self.max_duration_sec
        while self.frames and self.frames[0][0] < cutoff:
            self.frames.popleft()

    def snapshot_frames(self):
        with self.lock:
            return [f for (_, f) in list(self.frames)]

    def approx_duration(self):
        with self.lock:
            if len(self.frames) < 2:
                return 0.0
            start_t = self.frames[0][0]
            end_t = self.frames[-1][0]
            return max(0.0, end_t - start_t)

    def dump_to_temp_mp4(self, out_dir: Path, fps=30):
        frames = self.snapshot_frames()
        if not frames:
            return None
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_dir.mkdir(parents=True, exist_ok=True)
        temp_path = out_dir / f"segment_{int(time.time())}.mp4"
        writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (w, h))
        try:
            for f in frames:
                writer.write(f)
        finally:
            writer.release()
        return str(temp_path)

class MobileVideoGPTWrapper:
    """Thin wrapper around Mobile-VideoGPT loading and inference."""
    def __init__(self, model_id: str, device: str = None):
        self.model_id = model_id
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()

    def _load_model(self):
        try:
            # Lazy import to avoid import error before repo is installed
            from mobilevideogpt.utils import preprocess_input  # type: ignore
            self.preprocess_input = preprocess_input
        except Exception as e:
            logging.error("Failed to import mobilevideogpt. Ensure the repo is installed and PYTHONPATH is set.")
            raise e

        config = AutoConfig.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=False)
        torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            config=config,
            torch_dtype=torch_dtype
        )
        if self.device == 'cuda':
            self.model = self.model.cuda()

    def infer(self, video_path: str, prompt: str) -> str:
        input_ids, video_frames, context_frames, stop_str = self.preprocess_input(
            self.model, self.tokenizer, video_path, prompt
        )

        images_tensor = torch.stack(video_frames, dim=0)
        context_tensor = torch.stack(context_frames, dim=0)

        if self.device == 'cuda':
            images_tensor = images_tensor.half().cuda()
            context_tensor = context_tensor.half().cuda()
        else:
            images_tensor = images_tensor.to(self.device)
            context_tensor = context_tensor.to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                context_images=context_tensor,
                do_sample=False,
                temperature=0,
                top_p=1,
                num_beams=1,
                max_new_tokens=1024,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)].strip()
        return outputs

class InferenceCoordinator:
    """Background worker that sequentially runs inference on the latest 30s window."""
    def __init__(self, ring: VideoRingBuffer, model_wrapper: MobileVideoGPTWrapper, runs_dir: Path, prompt: str, min_seconds_required: float = 30.0, file_fps: int = 12):
        self.ring = ring
        self.wrapper = model_wrapper
        self.runs_dir = runs_dir
        self.prompt = prompt
        self.min_seconds_required = float(min_seconds_required)
        self.file_fps = int(file_fps)
        self.running = True
        self.lock = Lock()
        self.current_caption = f"Initializing Mobile-VideoGPT... ({self.wrapper.device.upper()})"
        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:
            try:
                # Ensure we have enough window duration
                if self.ring.approx_duration() < self.min_seconds_required:
                    time.sleep(0.25)
                    continue

                # Snapshot last 30s to a temp mp4
                segments_dir = self.runs_dir / "segments"
                video_path = self.ring.dump_to_temp_mp4(segments_dir, fps=self.file_fps)
                if video_path is None:
                    time.sleep(0.25)
                    continue

                t0 = time.time()
                caption = self.wrapper.infer(video_path, self.prompt)
                dt = time.time() - t0

                with self.lock:
                    self.current_caption = f"{caption}"

                # Persist raw text alongside video
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_txt = self.runs_dir / f"caption_{ts}.txt"
                with open(out_txt, 'w', encoding='utf-8') as f:
                    f.write(caption + "\n")

                # Sequential policy: immediately move on to process the latest 30s next loop.
                # Backlog is implicitly discarded because we always snapshot the most recent window.
            except Exception as e:
                logging.error(f"Inference worker error: {str(e)}")
                time.sleep(0.5)

    def get_caption(self):
        with self.lock:
            return self.current_caption

    def stop(self):
        self.running = False
        self.thread.join()

def get_gpu_usage():
    """Get the GPU memory usage and approximate utilization"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # MB

        memory_used_percent = (memory_allocated / memory_total) * 100
        gpu_info = f"GPU Memory Usage: {memory_used_percent:.2f}% | Allocated: {memory_allocated:.2f} MB / {memory_total:.2f} MB"
        
        return gpu_info
    else:
        return "GPU not available"

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def live_stream_with_mobile_videogpt(model_id: str, prompt: str, display_width=1280, display_height=720, ring_seconds=30, record_width=None, file_fps=None):
    """Stream webcam feed, maintain a 30s ring buffer, and run sequential Mobile-VideoGPT inference."""
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to access webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Webcam feed started successfully using {device.upper()}.")

    # Prepare runs dir
    runs_dir = Path(os.path.abspath("runs")) / datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dir(runs_dir)
    logger.info(f"Saving outputs to {runs_dir}")

    # Initialize ring buffer and model
    ring = VideoRingBuffer(max_duration_sec=ring_seconds, target_width=record_width, target_height=None)
    try:
        wrapper = MobileVideoGPTWrapper(model_id=model_id, device=device)
    except Exception:
        logger.error("Failed to initialize Mobile-VideoGPT. Exiting.")
        return

    # Determine recording FPS from camera to match example behavior (preprocess handles actual sampling)
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    if cam_fps is None or cam_fps <= 1.0:
        cam_fps = 30.0
    eff_fps = int(round(file_fps if file_fps is not None else cam_fps))
    coordinator = InferenceCoordinator(ring, wrapper, runs_dir, prompt, min_seconds_required=float(ring_seconds), file_fps=eff_fps)

    prev_time = time.time()  # Track time to calculate FPS

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from webcam.")
                break

            # Append to ring buffer
            now_s = time.time()
            ring.append(frame, now_s)
            current_caption = coordinator.get_caption()

            # Get GPU memory usage
            gpu_info = get_gpu_usage()

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Break caption into lines if it overflows
            max_width = 40  # Adjust max width for caption as needed
            caption_lines = [current_caption[i:i + max_width] for i in range(0, len(current_caption), max_width)]

            y_offset = 40
            for line in caption_lines:
                cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30

            # Display GPU memory usage and FPS
            cv2.putText(frame, gpu_info, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            y_offset += 30
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

            # Display the video frame
            cv2.imshow("Mobile-VideoGPT: Video Understanding", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Stream interrupted by user.")
    finally:
        coordinator.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    logger = setup_logging()

    MODEL_ID = "Amshaker/Mobile-VideoGPT-0.5B"
    PROMPT = "Describe what the person is doing at home (e.g., working on a laptop, eating, relaxing)."

    logger.info(f"Using model: {MODEL_ID}")
    logger.info("Starting live stream with Mobile-VideoGPT sequential 30s inference...")
    live_stream_with_mobile_videogpt(MODEL_ID, PROMPT)
