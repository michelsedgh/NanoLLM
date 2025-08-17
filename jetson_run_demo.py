#!/usr/bin/env python3
"""
Simple runner for VideoChat-Flash on Jetson Orin Nano
Assumes all dependencies are already installed
"""

import argparse
import os
import time
import tempfile
from datetime import datetime

import cv2
import torch
from transformers import AutoModel, AutoTokenizer


# Jetson-specific environment settings
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")


def open_camera(device_path: str, width: int, height: int, target_fps: float) -> cv2.VideoCapture:
    """Open camera using V4L2 backend for Jetson"""
    cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera at {device_path}")

    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if target_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(target_fps))
    
    return cap


def get_fps(cap: cv2.VideoCapture, default_fps: float = 30.0) -> float:
    """Get camera FPS with fallback"""
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or float(fps) < 5.0:
            return default_fps
        return float(fps)
    except Exception:
        return default_fps


def write_segment_avi(frames, fps: float, frame_size: tuple[int, int]) -> str:
    """Write frames to AVI file using MJPG codec"""
    tmp_dir = tempfile.gettempdir()
    avi_path = os.path.join(tmp_dir, f"jetson_segment_{int(time.time()*1000)}.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(avi_path, fourcc, fps, frame_size)
    
    if not writer.isOpened():
        raise RuntimeError("Failed to initialize MJPG video writer")
    
    for frame in frames:
        writer.write(frame)
    writer.release()
    return avi_path


def load_model(model_id: str):
    """Load VideoChat-Flash model on CUDA with fp16"""
    print(f"Loading {model_id} on CUDA with fp16...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model = model.to(dtype=torch.float16, device="cuda")
    model.eval()
    
    # Initialize vision tower
    try:
        _ = model.get_vision_tower().image_processor
    except Exception:
        pass
    
    # Enable compression for better performance on Jetson
    if hasattr(model, 'config'):
        model.config.mm_llm_compress = True
        model.config.llm_compress_type = "uniform0_attention"
        model.config.llm_compress_layer_list = [4, 18]
        model.config.llm_image_token_ratio_list = [1, 0.75, 0.25]
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Jetson Orin Nano VideoChat-Flash Demo")
    parser.add_argument("--prompt", type=str, default="Describe this video in detail.")
    parser.add_argument("--model", type=str, default="OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448")
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--camera", type=str, default="/dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--show-preview", action="store_true", help="Show OpenCV preview window")
    args = parser.parse_args()

    print("=== Jetson Orin Nano VideoChat-Flash Demo ===")
    print(f"Camera: {args.camera}")
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print()

    # Load model
    model, tokenizer = load_model(args.model)

    # Generation config optimized for Jetson
    generation_config = dict(
        do_sample=False,
        temperature=0.0,
        max_new_tokens=192,
        top_p=0.1,
        num_beams=1,
    )

    # Open camera
    cap = open_camera(args.camera, args.width, args.height, args.target_fps)
    camera_fps = get_fps(cap)
    fps = args.target_fps if camera_fps < 5.0 else min(camera_fps, args.target_fps)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height)

    print(f"Camera opened: {frame_width}x{frame_height} @ target ~{fps:.1f} FPS")
    print("Press Ctrl+C to stop. Running continuous segments...\n")

    try:
        while True:
            # Capture segment
            frames = []
            start_time = time.time()
            next_refresh = start_time
            
            while (time.time() - start_time) < args.segment_seconds:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                frames.append(frame)

                if args.show_preview:
                    preview = frame.copy()
                    cv2.putText(preview, "Recording...", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.imshow("Jetson VideoChat Demo", preview)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt

                # Frame pacing
                now = time.time()
                if now < next_refresh:
                    time.sleep(min(0.01, next_refresh - now))
                next_refresh = now + (1.0 / max(fps, 1.0))

            if len(frames) == 0:
                print("No frames captured; retrying...")
                continue

            # Limit frames for performance
            if len(frames) > args.max_frames:
                stride = max(1, len(frames) // args.max_frames)
                frames = frames[::stride][:args.max_frames]

            # Write video segment
            video_path = write_segment_avi(frames, fps=fps, frame_size=(frame_width, frame_height))

            # Run inference
            try:
                t0 = time.time()
                output, _ = model.chat(
                    video_path=video_path,
                    tokenizer=tokenizer,
                    user_prompt=args.prompt,
                    return_history=True,
                    max_num_frames=min(args.max_frames, len(frames)),
                    generation_config=generation_config,
                )
                dt = time.time() - t0
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] ({dt:.2f}s, {len(frames)} frames):")
                print(f"{output}\n")
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"Inference error: {e}")
            finally:
                # Clean up temp file
                try:
                    if os.path.exists(video_path):
                        os.remove(video_path)
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\nStopping demo...")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        if args.show_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
