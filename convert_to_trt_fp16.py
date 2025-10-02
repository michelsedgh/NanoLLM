#!/usr/bin/env python3
"""
Convert a SavedModel (streaming MoViNet) to TF-TRT FP16.

Requirements:
 - NVIDIA GPU with recent drivers
 - CUDA + cuDNN properly installed
 - TensorRT installed
 - TensorFlow built with TF-TRT support (typically Linux + NVIDIA GPU)

Notes:
 - This will NOT work on macOS without an NVIDIA GPU. Apple Silicon uses Metal, not TensorRT.
 - Run this on a Linux machine with an NVIDIA GPU (or cloud VM) for successful conversion.
"""

import argparse
import os
import sys
import tensorflow as tf

try:
    # In TF 2.14+, TF-TRT is under tf.experimental.tensorrt. Older versions use python.compiler.tensorrt
    from tensorflow.experimental.tensorrt import Converter as TrtConverter
    USE_EXPERIMENTAL_API = True
except Exception:  # pragma: no cover - compatibility path
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    USE_EXPERIMENTAL_API = False


def has_nvidia_gpu() -> bool:
    gpus = tf.config.list_physical_devices("GPU")
    return len(gpus) > 0 and any("NVIDIA" in (gpu.name or "") for gpu in gpus)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SavedModel to TF-TRT FP16")
    parser.add_argument(
        "--input_saved_model_dir",
        type=str,
        required=True,
        help="Path to input SavedModel directory",
    )
    parser.add_argument(
        "--output_saved_model_dir",
        type=str,
        default="./movinet_trt_fp16_saved_model",
        help="Output directory for TF-TRT optimized SavedModel",
    )
    parser.add_argument(
        "--max_workspace_size_bytes",
        type=int,
        default=1 << 30,
        help="TensorRT workspace size (default 1GiB)",
    )
    parser.add_argument(
        "--minimum_segment_size",
        type=int,
        default=3,
        help="Minimum graph segment size for TRT conversion",
    )
    parser.add_argument(
        "--allow_build_at_runtime",
        action="store_true",
        help="Allow TRT engine building at runtime (may add first-run latency)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if sys.platform == "darwin":
        print(
            "❌ TF-TRT conversion is not supported on macOS without an NVIDIA GPU. "
            "Run this on a Linux machine with NVIDIA GPU."
        )
        sys.exit(2)

    if not has_nvidia_gpu():
        print("❌ No NVIDIA GPU detected by TensorFlow. Aborting conversion.")
        sys.exit(2)

    input_dir = os.path.abspath(args.input_saved_model_dir)
    output_dir = os.path.abspath(args.output_saved_model_dir)

    print(f"📁 Input SavedModel: {input_dir}")
    print(f"📦 Output (TF-TRT FP16) will be written to: {output_dir}")

    if USE_EXPERIMENTAL_API:
        converter = TrtConverter(
            input_saved_model_dir=input_dir,
            conversion_params=tf.experimental.tensorrt.ConversionParams(
                precision_mode="FP16",
                max_workspace_size_bytes=int(args.max_workspace_size_bytes),
                minimum_segment_size=int(args.minimum_segment_size),
                allow_build_at_runtime=bool(args.allow_build_at_runtime),
            ),
        )
        print("🚀 Starting TF-TRT conversion (experimental API)…")
        converter.convert()
        converter.save(output_dir)
    else:
        conversion_params = (
            trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                precision_mode="FP16",
                max_workspace_size_bytes=int(args.max_workspace_size_bytes),
                minimum_segment_size=int(args.minimum_segment_size),
                # maximum_cached_engines may be available depending on TF version
            )
        )

        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=input_dir,
            conversion_params=conversion_params,
        )
        print("🚀 Starting TF-TRT conversion (legacy API)…")
        converter.convert()
        converter.save(output_dir)

    print("✅ TF-TRT FP16 conversion complete")


if __name__ == "__main__":
    main()


