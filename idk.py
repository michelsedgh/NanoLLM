import cv2
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import logging
import time
import sys
import os
import tempfile
from threading import Thread, Lock
from collections import deque

# Mobile-VideoGPT import (ensure Mobile-VideoGPT repo is on PYTHONPATH)
try:
    from mobilevideogpt.utils import preprocess_input
except Exception as _import_err:
    preprocess_input = None

def setup_logging():
    """Configure logging with basic formatting"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class CaptionGenerator:
    def __init__(self, tokenizer, model, device, window_seconds=8, interval_seconds=4, prompt=None, fps_hint=24):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.current_caption = f"Initializing Mobile-VideoGPT... ({device.upper()})"
        self.lock = Lock()
        self.running = True

        # Rolling time-based frame buffer: deque[(timestamp, frame)]
        self.frame_buffer = deque()
        self.window_seconds = max(1, int(window_seconds))
        self.interval_seconds = max(1, int(interval_seconds))
        self.prompt = prompt or (
            "Describe the main human activity in the video in one short phrase."
        )
        self.fps_hint = fps_hint if fps_hint and fps_hint > 0 else 24
        self._last_infer_time = 0.0

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

    def _write_temp_video(self, frames, fps):
        if not frames:
            return None
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.mp4', prefix='mvideogpt_')
        os.close(tmp_fd)
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
        try:
            for f in frames:
                writer.write(f)
        finally:
            writer.release()
        return tmp_path

    def _run_videogpt(self, video_path):
        if preprocess_input is None:
            logging.error("Mobile-VideoGPT not found. Ensure repo is cloned and PYTHONPATH is set.")
            return f"Mobile-VideoGPT: preprocess_input unavailable ({self.device.upper()})"

        try:
            config_device = self.device
            dtype = torch.float16 if config_device == 'cuda' else torch.float32

            input_ids, video_frames, context_frames, stop_str = preprocess_input(
                self.model, self.tokenizer, video_path, self.prompt
            )

            with torch.inference_mode():
                images = torch.stack(video_frames, dim=0).to(config_device, dtype=dtype)
                context_images = torch.stack(context_frames, dim=0).to(config_device, dtype=dtype)
                output_ids = self.model.generate(
                    input_ids,
                    images=images,
                    context_images=context_images,
                    do_sample=False,
                    temperature=0,
                    top_p=1,
                    num_beams=1,
                    max_new_tokens=128,
                    use_cache=True,
                )

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if stop_str and outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)].strip()

            return f"Mobile-VideoGPT: {outputs} ({self.device.upper()})"
        except Exception as e:
            logging.error(f"Mobile-VideoGPT inference error: {str(e)}")
            return f"Mobile-VideoGPT: Inference failed ({self.device.upper()})"

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
                    tmp_path = self._write_temp_video(frames, fps)
                    if tmp_path:
                        caption = self._run_videogpt(tmp_path)
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                        with self.lock:
                            self.current_caption = caption
                    self._last_infer_time = now_ts
            except Exception as e:
                logging.error(f"Caption worker error: {str(e)}")
            time.sleep(0.05)  # Prevent busy waiting

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

def load_models(model_id="Amshaker/Mobile-VideoGPT-0.5B"):
    """Load Mobile-VideoGPT model and tokenizer"""
    try:
        device = 'cuda' if torch.cuda.is_available() else (
            'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
        )

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=(torch.float16 if device == 'cuda' else torch.float32),
            trust_remote_code=True
        )
        if device == 'cuda':
            try:
                torch.cuda.set_per_process_memory_fraction(0.9)
            except Exception:
                pass
            model = model.to('cuda')
        elif device == 'mps':
            model = model.to('mps')

        if preprocess_input is None:
            logging.warning("mobilevideogpt.utils not importable. Set PYTHONPATH to Mobile-VideoGPT repo before running.")

        return tokenizer, model, device
    except Exception as e:
        logging.error(f"Failed to load Mobile-VideoGPT: {str(e)}")
        return None, None, None

def live_stream_with_caption(tokenizer, model, device, display_width=1280, display_height=720):
    """Stream webcam feed with live Mobile-VideoGPT activity captions and FPS"""
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
    caption_generator = CaptionGenerator(tokenizer, model, device, window_seconds=8, interval_seconds=4, fps_hint=fps_hint)

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
            cv2.imshow("Mobile-VideoGPT: Live Activity Captioning", frame)

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

    logger.info("Loading Mobile-VideoGPT model...")
    tokenizer, mvideogpt_model, device = load_models()
    if None in (tokenizer, mvideogpt_model):
        logging.error("Failed to load the Mobile-VideoGPT model. Exiting.")
        sys.exit(1)

    logger.info(f"Using {device.upper()} for inference.")
    logger.info("Starting live stream with Mobile-VideoGPT captioning and FPS display...")
    live_stream_with_caption(tokenizer, mvideogpt_model, device)
