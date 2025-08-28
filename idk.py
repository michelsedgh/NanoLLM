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


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class CaptionGenerator:
    def __init__(self, tokenizer, model, device, window_seconds=5, prompt=None, fps_hint=24):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.current_caption = f"Initializing VideoChat-Flash-Qwen2_5-2B... ({device.upper()})"
        self.lock = Lock()
        self.running = True

        self.frame_buffer = deque()
        self.window_seconds = window_seconds
        self.interval_seconds = 0.0
        self.prompt = prompt or (
            "Describe the main human activity in the video in one short phrase."
        )
        self.fps_hint = fps_hint if fps_hint and fps_hint > 0 else 24
        self._last_infer_time = 0.0

        self.max_num_frames = 64
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
        fd, path = tempfile.mkstemp(suffix='.mp4', prefix='vchat_')
        os.close(fd)
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        try:
            for fr in frames:
                writer.write(fr)
        finally:
            writer.release()
        return path

    def _run_videochat(self, video_path):
        try:
            with torch.inference_mode():
                resp = self.model.chat(
                    video_path=video_path,
                    tokenizer=self.tokenizer,
                    user_prompt=self.prompt,
                    return_history=False,
                    max_num_frames=self.max_num_frames,
                    generation_config=self.generation_config,
                )
            output = resp[0] if isinstance(resp, tuple) else resp
            return f"VideoChat-Flash: {output.strip()} ({self.device.upper()})"
        except Exception as e:
            logging.error(f"VideoChat-Flash inference error: {str(e)}")
            logging.error(traceback.format_exc())
            return f"VideoChat-Flash: Inference failed ({self.device.upper()})"

    def _caption_worker(self):
        processing = False
        while self.running:
            try:
                if processing:
                    time.sleep(0.05)
                    continue

                now_ts = time.time()
                if len(self.frame_buffer) == 0:
                    time.sleep(0.01)
                    continue

                self._prune_buffer(now_ts)
                frames = [f for _, f in list(self.frame_buffer)]
                fps = self._estimate_fps()

                tmp_path = self._write_temp_video(frames, fps)
                if tmp_path:
                    processing = True
                    caption = self._run_videochat(tmp_path)
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                    with self.lock:
                        self.current_caption = caption
                    processing = False
            except Exception as e:
                logging.error(f"Caption worker error: {str(e)}")
                logging.error(traceback.format_exc())
                processing = False
            time.sleep(0.01)

    def get_caption(self):
        with self.lock:
            return self.current_caption

    def stop(self):
        self.running = False
        self.thread.join()


def get_gpu_usage():
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        memory_used_percent = (memory_allocated / memory_total) * 100
        return f"GPU Memory Usage: {memory_used_percent:.2f}% | Allocated: {memory_allocated:.2f} MB / {memory_total:.2f} MB"
    else:
        return "GPU not available"


def load_models(model_id="OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        logging.info("Tokenizer loaded.")
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(torch.float16).cuda()
        return tokenizer, model, 'cuda'
    except Exception as e:
        logging.error(f"Failed to load VideoChat-Flash-Qwen2_5-2B: {str(e)}")
        logging.error(traceback.format_exc())
        return None, None, None


def live_stream_with_caption(tokenizer, model, device, display_width=1280, display_height=720):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to access webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)

    logger.info(f"Webcam feed started successfully using {device.upper()}.")
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    fps_hint = int(cap_fps) if cap_fps and cap_fps > 0 else 24
    caption_generator = CaptionGenerator(tokenizer, model, device, window_seconds=5, fps_hint=fps_hint)

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from webcam.")
                break

            caption_generator.update_frame(frame)
            current_caption = caption_generator.get_caption()

            gpu_info = get_gpu_usage()

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            max_width = 60
            caption_lines = [current_caption[i:i + max_width] for i in range(0, len(current_caption), max_width)]

            y_offset = 40
            for line in caption_lines:
                cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30

            cv2.putText(frame, gpu_info, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            y_offset += 30
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

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
    logger.info("Model loaded successfully. Exiting test.")
    # logger.info("Starting live stream with VideoChat-Flash-Qwen2_5-2B captioning and FPS display...")
    # live_stream_with_caption(tokenizer, videochat_model, device)
