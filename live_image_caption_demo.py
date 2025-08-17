import argparse
import os
import time
from datetime import datetime

import cv2
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


# Favor MPS on Apple Silicon, disable CUDA
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def select_device_and_dtype():
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def open_camera(camera_index: int, width: int, height: int, target_fps: float) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera. Allow camera access in System Settings → Privacy & Security → Camera for Terminal/Python.")
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if target_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(target_fps))
    return cap


def load_model(model_id: str, device: str, dtype: torch.dtype):
    print(f"Loading {model_id} on {device} ({dtype})...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": device},
    )
    model.eval()
    return model, processor


def main():
    parser = argparse.ArgumentParser(description="Continuous camera → image caption demo (Qwen2.5-VL)")
    parser.add_argument("--prompt", type=str, default="Describe what is happening.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-2B-Instruct")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between frames to caption.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--show-preview", action="store_true")
    args = parser.parse_args()

    device, dtype = select_device_and_dtype()
    model, processor = load_model(args.model, device, dtype)

    cap = open_camera(args.camera_index, args.width, args.height, target_fps=15.0)
    print("Press Ctrl+C to stop. Capturing frames continuously...\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            # BGR → RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": args.prompt},
                    ],
                }
            ]
            chat = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=[chat], images=[rgb], return_tensors="pt")
            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            if dtype == torch.float16:
                # Cast only floating tensors to fp16
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                        inputs[k] = v.to(dtype)

            try:
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
                text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] {text}\n")
            except Exception as e:
                print(f"Inference error: {e}")

            if args.show_preview:
                preview = frame.copy()
                cv2.putText(preview, "Capturing...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("Qwen2.5-VL Preview", preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            time.sleep(max(0.0, args.interval))

    except KeyboardInterrupt:
        pass
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


