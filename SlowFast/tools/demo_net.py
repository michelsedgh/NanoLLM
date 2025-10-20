class TerminalReporter:
    def __init__(self, cfg):
        self.top_k = cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS
        self.common_classes = set(cfg.DEMO.COMMON_CLASS_NAMES)
        self.common_thres = cfg.DEMO.COMMON_CLASS_THRES
        self.uncommon_thres = cfg.DEMO.UNCOMMON_CLASS_THRES
        class_names_path = cfg.DEMO.LABEL_FILE_PATH
        try:
            self.class_names, _, _ = get_class_names(class_names_path, None, None)
        except Exception:
            logger.warning("Unable to load class names from %s; falling back to indices.", class_names_path)
            self.class_names = None

    def _threshold(self, class_name):
        if not self.common_classes:
            return self.common_thres
        if class_name in self.common_classes:
            return self.common_thres
        return self.uncommon_thres

    def log_task(self, task):
        preds = task.action_preds
        boxes = task.bboxes
        num_boxes = 0 if preds is None else preds.shape[0]
        logger.info("Task %d: %d detections", task.id, num_boxes)

        if preds is None or num_boxes == 0:
            logger.info("  No detections above threshold")
            return

        preds_np = preds.numpy()
        boxes_np = boxes.numpy() if boxes is not None else None

        for idx in range(num_boxes):
            box_str = "(no box)"
            if boxes_np is not None and idx < boxes_np.shape[0]:
                x1, y1, x2, y2 = boxes_np[idx]
                box_str = f"[{int(round(x1))}, {int(round(y1))}, {int(round(x2))}, {int(round(y2))}]"

            scores = preds_np[idx]
            top_indices = scores.argsort()[::-1][: self.top_k]
            labels = []
            for cls_idx in top_indices:
                score = float(scores[cls_idx])
                class_name = (
                    self.class_names[cls_idx]
                    if self.class_names and cls_idx < len(self.class_names)
                    else f"class_{cls_idx}"
                )
                if score < self._threshold(class_name):
                    continue
                labels.append(f"{class_name} {score:.2f}")

            if not labels:
                cls_idx = int(top_indices[0])
                score = float(scores[cls_idx])
                class_name = (
                    self.class_names[cls_idx]
                    if self.class_names and cls_idx < len(self.class_names)
                    else f"class_{cls_idx}"
                )
                labels.append(f"{class_name} {score:.2f}")

            logger.info("  Box %d %s -> %s", idx, box_str, ", ".join(labels))
#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import time

import os

import numpy as np
import torch
import tqdm

from slowfast.utils import logging
from slowfast.visualization.async_predictor import AsyncDemo, AsyncVis
from slowfast.visualization.ava_demo_precomputed_boxes import (
    AVAVisualizerWithPrecomputedBox,
)
from slowfast.visualization.demo_loader import ThreadVideoManager, VideoManager
from slowfast.visualization.predictor import ActionPredictor
from slowfast.visualization.video_visualizer import VideoVisualizer
from slowfast.utils.misc import get_class_names

logger = logging.get_logger(__name__)


def run_demo(cfg, frame_provider):
    """
    Run demo visualization.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        frame_provider (iterator): Python iterator that return task objects that are filled
            with necessary information such as `frames`, `id` and `num_buffer_frames` for the
            prediction and visualization pipeline.
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    # Print config.
    logger.info("Run demo with config:")
    logger.info(cfg)
    common_classes = (
        cfg.DEMO.COMMON_CLASS_NAMES if len(cfg.DEMO.LABEL_FILE_PATH) != 0 else None
    )

    video_vis = None
    async_vis = None
    terminal_only = os.environ.get("SLOWFAST_TERMINAL_ONLY", "0") == "1"

    reporter = TerminalReporter(cfg) if terminal_only else None

    if not terminal_only:
        video_vis = VideoVisualizer(
            num_classes=cfg.MODEL.NUM_CLASSES,
            class_names_path=cfg.DEMO.LABEL_FILE_PATH,
            top_k=cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS,
            thres=cfg.DEMO.COMMON_CLASS_THRES,
            lower_thres=cfg.DEMO.UNCOMMON_CLASS_THRES,
            common_class_names=common_classes,
            colormap=cfg.TENSORBOARD.MODEL_VIS.COLORMAP,
            mode=cfg.DEMO.VIS_MODE,
        )

        async_vis = AsyncVis(video_vis, n_workers=cfg.DEMO.NUM_VIS_INSTANCES)

    if cfg.NUM_GPUS <= 1:
        model = ActionPredictor(cfg=cfg, async_vis=async_vis)
    else:
        model = AsyncDemo(cfg=cfg, async_vis=async_vis)

    seq_len = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE

    assert (
        cfg.DEMO.BUFFER_SIZE <= seq_len // 2
    ), "Buffer size cannot be greater than half of sequence length."
    num_task = 0
    # Start reading frames.
    frame_provider.start()
    for able_to_read, task in frame_provider:
        if not able_to_read:
            break
        if task is None:
            time.sleep(0.02)
            continue
        num_task += 1

        model.put(task)
        try:
            task = model.get()
            num_task -= 1
            if reporter:
                reporter.log_task(task)
                torch.cuda.empty_cache()
            yield task
        except IndexError:
            continue

    while num_task != 0:
        try:
            task = model.get()
            num_task -= 1
            if reporter:
                reporter.log_task(task)
                torch.cuda.empty_cache()
            yield task
        except IndexError:
            continue


def demo(cfg):
    """
    Run inference on an input video or stream from webcam.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # AVA format-specific visualization with precomputed boxes.
    if cfg.DETECTION.ENABLE and cfg.DEMO.PREDS_BOXES != "":
        precomputed_box_vis = AVAVisualizerWithPrecomputedBox(cfg)
        precomputed_box_vis()
    else:
        start = time.time()
        if cfg.DEMO.THREAD_ENABLE:
            frame_provider = ThreadVideoManager(cfg)
        else:
            frame_provider = VideoManager(cfg)

        for task in tqdm.tqdm(run_demo(cfg, frame_provider)):
            if os.environ.get("SLOWFAST_TERMINAL_ONLY", "0") == "1":
                continue
            frame_provider.display(task)

        frame_provider.join()
        frame_provider.clean()
        logger.info("Finish demo in: {}".format(time.time() - start))
