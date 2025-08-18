import cv2
from transformers import AutoModel, AutoTokenizer
import torch
import logging
import time
import sys
from threading import Thread, Lock
from queue import Queue, Empty
import tempfile
import os

def setup_logging():
    """Configure logging with basic formatting"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class CaptionGenerator:
    def __init__(self, tokenizer, model, device, sample_stride=4, segment_size=30):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.sample_stride = max(1, sample_stride)
        self.segment_size = max(1, segment_size)
        self.current_caption = f"Initializing VideoChat-Flash... ({device.upper()})"
        self.lock = Lock()
        self.running = True
        self.thread = Thread(target=self._caption_worker)
        self.thread.daemon = True
        self.frame_queue = Queue(maxsize=120)
        self.collected_frames = []
        self.total_frame_idx = 0
        self.chat_history = None
        self.generation_config = dict(
            do_sample=False,
            temperature=0.0,
            max_new_tokens=256,
            top_p=0.1,
            num_beams=1
        )
        self.user_prompt = "Describe what is happening in these frames."
        self.thread.start()

    def _write_temp_video(self, frames, fps=7.5):
        h, w = frames[0].shape[:2]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_path = tmp.name
        tmp.close()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        return tmp_path

    def _run_inference_on_segment(self, frames):
        try:
            if not frames:
                return
            temp_path = self._write_temp_video(frames)
            with torch.no_grad():
                if self.chat_history is None:
                    output, self.chat_history = self.model.chat(
                        video_path=temp_path,
                        tokenizer=self.tokenizer,
                        user_prompt=self.user_prompt,
                        return_history=True,
                        max_num_frames=len(frames),
                        generation_config=self.generation_config
                    )
                else:
                    output, self.chat_history = self.model.chat(
                        video_path=temp_path,
                        tokenizer=self.tokenizer,
                        user_prompt=self.user_prompt,
                        chat_history=self.chat_history,
                        return_history=True,
                        max_num_frames=len(frames),
                        generation_config=self.generation_config
                    )
            with self.lock:
                self.current_caption = f"VideoChat-Flash: {output} ({self.device.upper()})"
        except Exception as e:
            logging.error(f"Video segment inference error: {str(e)}")
        finally:
            try:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

    def _caption_worker(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.05)
                self.total_frame_idx += 1
                if (self.total_frame_idx - 1) % self.sample_stride == 0:
                    self.collected_frames.append(frame)
                if len(self.collected_frames) >= self.segment_size:
                    frames_to_process = self.collected_frames[:self.segment_size]
                    self.collected_frames = []
                    self._run_inference_on_segment(frames_to_process)
            except Empty:
                time.sleep(0.02)
            except Exception as e:
                logging.error(f"Caption worker error: {str(e)}")
                time.sleep(0.05)

    def update_frame(self, frame):
        try:
            self.frame_queue.put_nowait(frame.copy())
        except Exception:
            pass

    def get_caption(self):
        with self.lock:
            return self.current_caption

    def stop(self):
        self.running = False
        if self.thread.is_alive():
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

def load_models():
    """Load VideoChat-Flash model"""
    try:
        model_path = 'OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448'
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=False,
            resume_download=True
        )
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=False,
            resume_download=True
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            torch.cuda.set_per_process_memory_fraction(0.9)
            model = model.to(torch.bfloat16).cuda()
        else:
            model = model.to(torch.float32)

        # Disable compress by default for clarity
        if hasattr(model, 'config'):
            try:
                model.config.mm_llm_compress = False
            except Exception:
                pass

        return tokenizer, model, device
    except Exception as e:
        logging.error(f"Failed to load VideoChat-Flash model: {str(e)}")
        return None, None, None

def live_stream_with_caption(tokenizer, model, device, logger, display_width=1280, display_height=720):
    """Stream webcam feed with live captions and FPS"""
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to access webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)

    logger.info(f"Webcam feed started successfully using {device.upper()}.")
    caption_generator = CaptionGenerator(tokenizer, model, device, sample_stride=4, segment_size=30)

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
            cv2.imshow("VideoChat-Flash: Live Captions", frame)

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

    logger.info("Loading VideoChat-Flash model...")
    tokenizer, vc_model, device = load_models()
    if None in (tokenizer, vc_model):
        logging.error("Failed to load the VideoChat-Flash model. Exiting.")
        sys.exit(1)

    logger.info(f"Using {device.upper()} for inference.")
    logger.info("Starting live stream with VideoChat-Flash captioning and FPS display...")
    live_stream_with_caption(tokenizer, vc_model, device, logger)