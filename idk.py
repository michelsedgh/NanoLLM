import cv2
from transformers import AutoTokenizer, AutoModel
import torch
import logging
import time
import sys
import os
import tempfile
from threading import Thread, Lock
from collections import deque
import numpy as np
import traceback

# Try to ensure decord is importable; create a stub if missing (helps on unsupported architectures)
try:
    import decord  # type: ignore
except ImportError:
    import types, sys
    decord_stub = types.ModuleType("decord")
    sys.modules["decord"] = decord_stub
    # Provide minimal attributes that some repos expect
    setattr(decord_stub, "__version__", "0.0.0-stub")

def setup_logging():
    """Configure logging with basic formatting"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class CaptionGenerator:
    def __init__(self, tokenizer, model, device, window_seconds=5, interval_seconds=0.1, prompt=None, fps_hint=24):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.current_caption = f"Initializing VideoChat-Flash-Qwen2_5-2B... ({device.upper()})"
        self.lock = Lock()
        self.running = True

        # Rolling time-based frame buffer: deque[(timestamp, frame)]
        self.frame_buffer = deque()
        self.window_seconds = window_seconds  # Keep last 5 seconds
        self.interval_seconds = interval_seconds  # 0.1s for ~10fps
        self.prompt = prompt or (
            "Describe the main human activity in the video in one short phrase."
        )
        self.fps_hint = fps_hint if fps_hint and fps_hint > 0 else 24
        self._last_infer_time = 0.0

        # Model configuration
        self.max_num_frames = 64  # Reasonable limit for 5 seconds at 12-15 fps
        self.generation_config = dict(
            do_sample=False,
            temperature=0.0,
            max_new_tokens=128,
            top_p=0.1,
            num_beams=1
        )

        self.thread = Thread(target=self._caption_worker)
        self.thread.daemon = True
        self.thread.start()

    def _prune_buffer(self, now_ts):
        cutoff = now_ts - self.window_seconds
        while self.frame_buffer and self.frame_buffer[0][0] < cutoff:
            self.frame_buffer.popleft()

    def update_frame(self, frame):
        now_ts = time.time()
        # Copy to avoid mutation by caller
        self.frame_buffer.append((now_ts, frame.copy()))
        self._prune_buffer(now_ts)

    def _estimate_fps(self):
        if len(self.frame_buffer) < 2:
            return float(self.fps_hint)
        duration = self.frame_buffer[-1][0] - self.frame_buffer[0][0]
        if duration <= 0:
            return float(self.fps_hint)
        return max(1.0, float(len(self.frame_buffer)) / float(duration))

    def _run_videochat(self, frames, fps):
        if not frames:
            return f"VideoChat-Flash: No frames available ({self.device.upper()})"

        try:
            # Convert frames to the format expected by the model
            # VideoChat-Flash expects frames as numpy arrays in a specific format
            processed_frames = []
            for frame in frames[:self.max_num_frames]:
                # Convert BGR to RGB and ensure proper format
                if len(frame.shape) == 3 and frame.shape[2] == 3:  # BGR format
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = frame

                # Resize frame to model's expected resolution (448x448 as per model name)
                frame_resized = cv2.resize(frame_rgb, (448, 448))

                # Normalize to [0,1] range if needed (model might expect this)
                if frame_resized.dtype == np.uint8:
                    frame_resized = frame_resized.astype(np.float32) / 255.0

                processed_frames.append(frame_resized)

            # Convert to numpy array
            video_frames = np.array(processed_frames)

            # Run inference using the chat method
            with torch.inference_mode():
                output, _ = self.model.chat(
                    video_frames=video_frames,
                    tokenizer=self.tokenizer,
                    user_prompt=self.prompt,
                    return_history=False,
                    max_num_frames=self.max_num_frames,
                    generation_config=self.generation_config
                )

            return f"VideoChat-Flash: {output.strip()} ({self.device.upper()})"
        except Exception as e:
            logging.error(f"VideoChat-Flash inference error: {str(e)}")
            return f"VideoChat-Flash: Inference failed ({self.device.upper()})"

    def _caption_worker(self):
        while self.running:
            try:
                now_ts = time.time()
                # Trigger at interval regardless of buffer length; use whatever is available
                if (now_ts - self._last_infer_time) >= self.interval_seconds and len(self.frame_buffer) > 0:
                    # Snapshot current frames for thread safety
                    self._prune_buffer(now_ts)
                    frames = [f for _, f in list(self.frame_buffer)]
                    fps = self._estimate_fps()
                    caption = self._run_videochat(frames, fps)
                    with self.lock:
                        self.current_caption = caption
                    self._last_infer_time = now_ts
            except Exception as e:
                logging.error(f"Caption worker error: {str(e)}")
            time.sleep(0.01)  # Faster polling for higher fps

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

def load_models(model_id="OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448"):
    """Load VideoChat-Flash-Qwen2_5-2B model and tokenizer"""
    try:
        device = 'cuda' if torch.cuda.is_available() else (
            'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, force_download=True, resume_download=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, force_download=True, resume_download=True)

        # Move to device and set dtype
        if device == 'cuda':
            try:
                torch.cuda.set_per_process_memory_fraction(0.9)
            except Exception:
                pass
            model = model.to(torch.bfloat16).cuda()
        elif device == 'mps':
            model = model.to('mps')
        else:
            model = model.to(torch.float32)

        # Configure model settings
        model.config.mm_llm_compress = False

        logging.info(f"Successfully loaded {model_id}")
        return tokenizer, model, device
    except Exception as e:
        logging.error(f"Failed to load VideoChat-Flash-Qwen2_5-2B: {str(e)}")
        logging.error(traceback.format_exc())
        return None, None, None

def live_stream_with_caption(tokenizer, model, device, display_width=1280, display_height=720):
    """Stream webcam feed with live VideoChat-Flash-Qwen2_5-2B activity captions and FPS"""
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to access webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)

    logger.info(f"Webcam feed started successfully using {device.upper()}.")
    # fps hint from capture if available
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    fps_hint = int(cap_fps) if cap_fps and cap_fps > 0 else 24
    caption_generator = CaptionGenerator(tokenizer, model, device, window_seconds=5, interval_seconds=0.1, fps_hint=fps_hint)

    prev_time = time.time()  # Track time to calculate FPS

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from webcam.")
                break

            # Update caption and track FPS
            caption_generator.update_frame(frame)
            current_caption = caption_generator.get_caption()

            # Get GPU memory usage
            gpu_info = get_gpu_usage()

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Break caption into lines if it overflows
            max_width = 60  # Adjust max width for caption as needed
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
            cv2.imshow("VideoChat-Flash-Qwen2_5-2B: Live Activity Captioning", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Stream interrupted by user.")
    finally:
        caption_generator.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    logger = setup_logging()

    logger.info("Loading VideoChat-Flash-Qwen2_5-2B model...")
    tokenizer, videochat_model, device = load_models()
    if None in (tokenizer, videochat_model):
        logging.error("Failed to load the VideoChat-Flash-Qwen2_5-2B model. Exiting.")
        sys.exit(1)

    logger.info(f"Using {device.upper()} for inference.")
    logger.info("Starting live stream with VideoChat-Flash-Qwen2_5-2B captioning and FPS display...")
    live_stream_with_caption(tokenizer, videochat_model, device)
