import cv2
from transformers import AutoModel, AutoTokenizer
import torch
import logging
import time
from PIL import Image
import sys
from threading import Thread, Lock
from queue import Queue
import numpy as np

def setup_logging():
    """Configure logging with basic formatting"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class CaptionGenerator:
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.current_caption = f"Initializing video caption... ({device.upper()})"
        self.frame_buffer = []
        self.frame_count = 0
        self.lock = Lock()
        self.running = True
        self.thread = Thread(target=self._caption_worker)
        self.thread.daemon = True
        self.thread.start()

    def _caption_worker(self):
        while self.running:
            try:
                with self.lock:
                    if len(self.frame_buffer) >= 16:  # Reduced for Jetson memory constraints
                        frames_to_process = self.frame_buffer[:16].copy()
                        self.frame_buffer = self.frame_buffer[16:]
                        
                        caption = self._generate_caption(frames_to_process)
                        self.current_caption = caption
            except Exception as e:
                logging.error(f"Caption worker error: {str(e)}")
            time.sleep(0.1)  # Prevent busy waiting

    def _generate_caption(self, frames):
        try:
            # Create a temporary video from frames
            import tempfile
            temp_video_path = tempfile.mktemp(suffix='.mp4')
            
            # Write frames to video file
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (width, height))
            
            for frame in frames:
                out.write(frame)
            out.release()
            
            # Process with VideoChat - optimized for Jetson
            generation_config = dict(
                do_sample=False,
                temperature=0.0,
                max_new_tokens=50,  # Reduced for memory efficiency
                top_p=0.1,
                num_beams=1
            )
            
            # Clear GPU cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            question = "Describe what is happening in this video."
            output, _ = self.model.chat(
                video_path=temp_video_path,
                tokenizer=self.tokenizer,
                user_prompt=question,
                return_history=False,
                max_num_frames=16,  # Reduced for memory efficiency
                generation_config=generation_config
            )
            
            # Clean up temp file
            import os
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
                
            return f"VideoChat: {output} ({self.device.upper()})"
        except torch.cuda.OutOfMemoryError as e:
            logging.error(f"CUDA out of memory: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return f"VideoChat: GPU memory error - try reducing frame count ({self.device.upper()})"
        except Exception as e:
            logging.error(f"Caption generation error: {str(e)}")
            return f"VideoChat: Video processing failed ({self.device.upper()})"

    def update_frame(self, frame):
        # Implement take 1, skip 3 pattern
        if self.frame_count % 4 == 0:  # Take every 4th frame (0, 4, 8, 12...)
            with self.lock:
                self.frame_buffer.append(frame.copy())
        self.frame_count += 1

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
        # Check for flash attention availability
        try:
            import flash_attn
            logging.info("Flash attention is available")
        except ImportError:
            logging.warning("Flash attention not found - model may run slower")
        
        model_path = 'OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448'
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            # More conservative memory allocation for Jetson
            torch.cuda.set_per_process_memory_fraction(0.7)
            # Use fp16 instead of bfloat16 for better Jetson compatibility
            model = model.to(torch.float16).cuda()
        else:
            model = model.to(torch.float16)

        # Enable compression to reduce memory usage
        model.config.mm_llm_compress = True
        model.config.llm_compress_type = "uniform0_attention"
        model.config.llm_compress_layer_list = [4, 18]
        model.config.llm_image_token_ratio_list = [1, 0.75, 0.25]

        return tokenizer, model, device
    except Exception as e:
        logging.error(f"Failed to load models: {str(e)}")
        return None, None, None

def live_stream_with_caption(tokenizer, model, device, logger, display_width=1280, display_height=720):
    """Stream webcam feed with live video captions and FPS"""
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to access webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)

    logger.info(f"Webcam feed started successfully using {device.upper()}.")
    caption_generator = CaptionGenerator(tokenizer, model, device)

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
            cv2.imshow("VideoChat-Flash: Video Understanding Demo", frame)

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
    tokenizer, model, device = load_models()
    if None in (tokenizer, model):
        logging.error("Failed to load the VideoChat model. Exiting.")
        sys.exit(1)

    logger.info(f"Using {device.upper()} for inference.")
    logger.info("Starting live stream with VideoChat video captioning...")
    logger.info("Frame pattern: Take 1 frame, skip 3, collect 16 frames for processing (Jetson optimized)")
    live_stream_with_caption(tokenizer, model, device, logger)