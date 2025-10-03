#!/usr/bin/env python3
"""
Jetson camera and dependency checker.

- Verifies presence of: v4l2-ctl, gst-launch-1.0, gst-inspect-1.0
- Verifies NVIDIA GStreamer elements: nvv4l2decoder, nvvidconv, v4l2src, jpegparse
- Lists camera formats with v4l2-ctl if available
- Suggests the correct GStreamer pipeline for MJPEG or YUY2 at a given resolution

Usage:
  python3 check_camera_deps.py --device /dev/video0 --resolution 320

On Ubuntu/Jetson, install missing tools:
  sudo apt update && sudo apt install -y v4l-utils \
      gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import List, Optional


def find_command(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def run_cmd(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def check_gst_element(name: str) -> bool:
    tool = find_command("gst-inspect-1.0")
    if not tool:
        return False
    result = subprocess.run([tool, name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def has_opencv_gstreamer() -> bool:
    try:
        import cv2  # type: ignore

        info = cv2.getBuildInformation()
        return "GStreamer:                 YES" in info
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Jetson camera deps and suggest pipelines")
    parser.add_argument("--device", type=str, default="/dev/video0", help="V4L2 device path")
    parser.add_argument("--resolution", type=int, default=320, help="Square resize (width=height)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"OS: {platform.system()} {platform.release()} ({platform.platform()})")
    print(f"Python: {platform.python_version()}")
    print(f"Device: {args.device}")
    print()

    missing: List[str] = []

    v4l2_ctl = find_command("v4l2-ctl")
    gst_launch = find_command("gst-launch-1.0")
    gst_inspect = find_command("gst-inspect-1.0")

    print("Checking required tools…")
    print(f"  v4l2-ctl:        {'FOUND at ' + v4l2_ctl if v4l2_ctl else 'MISSING'}")
    print(f"  gst-launch-1.0:  {'FOUND at ' + gst_launch if gst_launch else 'MISSING'}")
    print(f"  gst-inspect-1.0: {'FOUND at ' + gst_inspect if gst_inspect else 'MISSING'}")
    if not v4l2_ctl:
        missing.append("v4l-utils")
    if not gst_launch or not gst_inspect:
        missing.extend([
            "gstreamer1.0-tools",
            "gstreamer1.0-plugins-base",
            "gstreamer1.0-plugins-good",
            "gstreamer1.0-plugins-bad",
            "gstreamer1.0-plugins-ugly",
            "gstreamer1.0-libav",
        ])

    print()
    print("Checking NVIDIA GStreamer elements…")
    print(f"  v4l2src:        {'OK' if check_gst_element('v4l2src') else 'MISSING'}")
    print(f"  jpegparse:      {'OK' if check_gst_element('jpegparse') else 'MISSING'}")
    print(f"  nvv4l2decoder:  {'OK' if check_gst_element('nvv4l2decoder') else 'MISSING'}")
    print(f"  nvvidconv:      {'OK' if check_gst_element('nvvidconv') else 'MISSING'}")
    print()

    print(f"OpenCV with GStreamer support: {'YES' if has_opencv_gstreamer() else 'NO'}")
    print()

    if missing:
        unique = sorted(set(missing))
        print("Install missing packages (Ubuntu/Jetson):")
        print(
            "  sudo apt update && sudo apt install -y " + " ".join(unique)
        )
        print()

    # Camera formats via v4l2-ctl
    mjpeg_supported = False
    yuy2_supported = False
    if v4l2_ctl:
        print("Listing camera formats via v4l2-ctl…")
        result = run_cmd([v4l2_ctl, f"--device={args.device}", "--list-formats-ext"])
        print(result.stdout)
        # Heuristic detection
        text = result.stdout.upper()
        if "MJPG" in text or "MJPEG" in text:
            mjpeg_supported = True
        if "YUYV" in text or "YUY2" in text:
            yuy2_supported = True
    else:
        print("v4l2-ctl not available; cannot list formats.")
        print("If your cam is typical, it supports YUY2 and possibly MJPEG.")

    res = int(args.resolution)
    device_num = re.sub(r"[^0-9]", "", args.device) or "0"

    print()
    print("Suggested GStreamer pipelines:")
    if mjpeg_supported:
        print("- MJPEG (hardware decode + GPU resize):")
        print(
            "  v4l2src device=/dev/video" + device_num + " io-mode=2 ! "
            + "image/jpeg, framerate=(fraction)30/1 ! "
            + "jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! "
            + f"video/x-raw, format=(string)BGRx, width=(int){res}, height=(int){res} ! "
            + "appsink drop=true max-buffers=1 sync=false"
        )
    if yuy2_supported or not mjpeg_supported:
        print("- YUY2 (GPU colorspace + resize):")
        print(
            "  v4l2src device=/dev/video" + device_num + " io-mode=2 ! "
            + "video/x-raw, format=(string)YUY2, framerate=(fraction)30/1 ! nvvidconv ! "
            + f"video/x-raw, format=(string)BGRx, width=(int){res}, height=(int){res} ! "
            + "appsink drop=true max-buffers=1 sync=false"
        )

    print()
    print("Test with gst-launch:")
    print("  gst-launch-1.0 -v v4l2src device=/dev/video" + device_num + " io-mode=2 ! image/jpeg, framerate=30/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx,width=" + str(res) + ",height=" + str(res) + " ! fakesink sync=false")
    print("  gst-launch-1.0 -v v4l2src device=/dev/video" + device_num + " io-mode=2 ! video/x-raw,format=YUY2,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx,width=" + str(res) + ",height=" + str(res) + " ! fakesink sync=false")


if __name__ == "__main__":
    main()


