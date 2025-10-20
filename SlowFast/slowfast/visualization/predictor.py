#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import queue
import sys

import cv2
import numpy as np

import slowfast.utils.checkpoint as cu
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from slowfast.datasets import cv2_transform
from slowfast.models import build_model
from slowfast.utils import logging
from slowfast.visualization.utils import process_cv2_inputs

logger = logging.get_logger(__name__)


class Predictor:
    """
    Action Predictor for action recognition.
    """

    def __init__(self, cfg, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            gpu_id (Optional[int]): GPU id.
        """
        if cfg.NUM_GPUS:
            self.gpu_id = torch.cuda.current_device() if gpu_id is None else gpu_id

        # Build the video model and print model statistics.
        self.model = build_model(cfg, gpu_id=gpu_id)
        self.model.eval()
        self.cfg = cfg

        if cfg.DETECTION.ENABLE:
            self.object_detector = Detectron2Predictor(cfg, gpu_id=self.gpu_id)

        logger.info("Start loading model weights.")
        cu.load_test_checkpoint(cfg, self.model)
        logger.info("Finish loading model weights")

    def __call__(self, task):
        """
        Returns the prediction results for the current task.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes)
        Returns:
            task (TaskInfo object): the same task info object but filled with
                prediction values (a tensor) and the corresponding boxes for
                action detection task.
        """
        if self.cfg.DETECTION.ENABLE:
            task = self.object_detector(task)

        frames, bboxes = task.frames, task.bboxes
        if bboxes is not None:
            bboxes = cv2_transform.scale_boxes(
                self.cfg.DATA.TEST_CROP_SIZE,
                bboxes,
                task.img_height,
                task.img_width,
            )
        if self.cfg.DEMO.INPUT_FORMAT == "BGR":
            frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

        frames = [
            cv2_transform.scale(self.cfg.DATA.TEST_CROP_SIZE, frame) for frame in frames
        ]
        inputs = process_cv2_inputs(frames, self.cfg)
        if bboxes is not None:
            index_pad = torch.full(
                size=(bboxes.shape[0], 1),
                fill_value=float(0),
                device=bboxes.device,
            )

            # Pad frame index for each box.
            bboxes = torch.cat([index_pad, bboxes], axis=1)
        if self.cfg.NUM_GPUS > 0 and bboxes is not None:
            bboxes = bboxes.cuda(
                device=torch.device(self.gpu_id), non_blocking=True
            )
        if self.cfg.NUM_GPUS > 0:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(
                        device=torch.device(self.gpu_id), non_blocking=True
                    )
            else:
                inputs = inputs.cuda(
                    device=torch.device(self.gpu_id), non_blocking=True
                )
        if self.cfg.DETECTION.ENABLE and not bboxes.shape[0]:
            preds = torch.tensor([])
        else:
            preds = self.model(inputs, bboxes)

        if self.cfg.NUM_GPUS:
            preds = preds.cpu()
            if bboxes is not None:
                bboxes = bboxes.detach().cpu()

        preds = preds.detach()
        task.add_action_preds(preds)
        if bboxes is not None:
            task.add_bboxes(bboxes[:, 1:])

        return task


class ActionPredictor:
    """
    Synchronous Action Prediction and Visualization pipeline with AsyncVis.
    """

    def __init__(self, cfg, async_vis=None, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            async_vis (AsyncVis object): asynchronous visualizer.
            gpu_id (Optional[int]): GPU id.
        """
        self.predictor = Predictor(cfg=cfg, gpu_id=gpu_id)
        self.async_vis = async_vis
        self._async_enabled = async_vis is not None
        self._last_result = None

    def put(self, task):
        """
        Make prediction and put the results in `async_vis` task queue.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes)
        """
        task = self.predictor(task)
        if self._async_enabled:
            self.async_vis.get_indices_ls.append(task.id)
            self.async_vis.put(task)
        else:
            self._last_result = task

    def get(self):
        """
        Get the visualized clips if any.
        """
        if self._async_enabled:
            try:
                task = self.async_vis.get()
            except (queue.Empty, IndexError):
                raise IndexError("Results are not available yet.")
            return task

        if self._last_result is None:
            raise IndexError("Results are not available yet.")
        task = self._last_result
        self._last_result = None
        return task


class Detectron2Predictor:
    """
    Wrapper around Detectron2 to return the required predicted bounding boxes
    as a ndarray.
    """

    def __init__(self, cfg, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            gpu_id (Optional[int]): GPU id.
        """

        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(cfg.DEMO.DETECTRON2_CFG))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.DEMO.DETECTRON2_THRESH
        self.cfg.MODEL.WEIGHTS = cfg.DEMO.DETECTRON2_WEIGHTS
        self.cfg.INPUT.FORMAT = cfg.DEMO.INPUT_FORMAT
        if cfg.NUM_GPUS and gpu_id is None:
            gpu_id = torch.cuda.current_device()
        self.cfg.MODEL.DEVICE = "cuda:{}".format(gpu_id) if cfg.NUM_GPUS > 0 else "cpu"

        trt_engine_path = os.environ.get("SLOWFAST_TRT_ENGINE")
        if trt_engine_path:
            logger.info("Initialized TensorRT Object Detection Model from engine: %s", trt_engine_path)
            self.use_trt = True
            self.predictor = TensorRTDetector(
                cfg,
                trt_engine_path,
                gpu_id=gpu_id,
            )
        else:
            logger.info("Initialized Detectron2 Object Detection Model.")
            self.use_trt = False
            self.predictor = DefaultPredictor(self.cfg)

    def __call__(self, task):
        """
        Return bounding boxes predictions as a tensor.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames)
        Returns:
            task (TaskInfo object): the same task info object but filled with
                prediction values (a tensor) and the corresponding boxes for
                action detection task.
        """
        middle_frame = task.frames[len(task.frames) // 2]
        if self.use_trt:
            pred_boxes = self.predictor(middle_frame)
        else:
            outputs = self.predictor(middle_frame)
            mask = outputs["instances"].pred_classes == 0
            pred_boxes = outputs["instances"].pred_boxes.tensor[mask]
        task.add_bboxes(pred_boxes)

        return task


class TensorRTDetector:
    """TensorRT inference wrapper for Detectron-style bounding boxes."""

    def __init__(self, cfg, engine_path, gpu_id=None):
        import tensorrt as trt
        from pathlib import Path

        engine_path = Path(engine_path).resolve()
        helper_candidates = [
            engine_path.parent,
            engine_path.parent.parent,
            engine_path.parent.parent.parent,
            engine_path.parent.parent.parent.parent,
            engine_path.parent.parent.parent.parent.parent,
        ]

        extra_helper = os.environ.get("SLOWFAST_TRT_HELPERS")
        if extra_helper:
            helper_candidates.append(Path(extra_helper))

        for candidate in helper_candidates:
            if candidate and candidate.exists() and str(candidate) not in sys.path:
                sys.path.append(str(candidate))

        try:
            from common_runtime import allocate_buffers, do_inference
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "Unable to locate TensorRT helper module 'common_runtime'. "
                "Ensure the engine path resides inside the TensorRT samples directory "
                "or set SLOWFAST_TRT_HELPERS to that directory."
            ) from err

        self.trt = trt
        self.allocate_buffers = allocate_buffers
        self.do_inference = do_inference

        self.engine_path = str(engine_path)
        if not os.path.isfile(self.engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)

        input_name = self.engine.get_tensor_name(0)
        input_shape = self.engine.get_tensor_shape(input_name)
        self.batch_size = input_shape[0]
        self.input_channels = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        self.score_thresh = cfg.DEMO.DETECTRON2_THRESH

        self.det2_cfg = get_cfg()
        self.det2_cfg.merge_from_file(model_zoo.get_config_file(cfg.DEMO.DETECTRON2_CFG))
        self.min_size_test = self.det2_cfg.INPUT.MIN_SIZE_TEST
        self.max_size_test = self.det2_cfg.INPUT.MAX_SIZE_TEST

    def _resize_pad(self, frame):
        img = frame.astype(np.float32)
        height, width = img.shape[:2]

        target_size = float(self.min_size_test)
        scale = target_size / min(height, width)
        if height < width:
            new_h = target_size
            new_w = scale * width
        else:
            new_w = target_size
            new_h = scale * height

        if max(new_h, new_w) > self.max_size_test:
            scale = float(self.max_size_test) / max(new_h, new_w)
            new_h = new_h * scale
            new_w = new_w * scale

        new_w = int(round(new_w))
        new_h = int(round(new_h))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full(
            (self.input_height, self.input_width, 3),
            (124.0, 116.0, 104.0),
            dtype=np.float32,
        )
        canvas[0:new_h, 0:new_w, :] = resized

        scaling = max(new_h / height, new_w / width)
        tensor = canvas.transpose(2, 0, 1)
        batch = tensor[np.newaxis, :, :, :]
        return batch, scaling

    def __call__(self, frame):
        batch, scaling = self._resize_pad(frame)

        np.copyto(self.inputs[0].host, batch.ravel())
        outputs = self.do_inference(
            self.context,
            self.engine,
            self.bindings,
            self.inputs,
            self.outputs,
            self.stream,
        )

        num_detections = int(outputs[0][0])
        boxes = outputs[1].reshape(self.batch_size, -1, 4)[0]
        scores = outputs[2].reshape(self.batch_size, -1)[0]
        classes = outputs[3].reshape(self.batch_size, -1)[0]

        scale_factor = self.input_height / scaling

        detections = []
        for idx in range(num_detections):
            if scores[idx] < self.score_thresh:
                continue
            if int(classes[idx]) != 0:
                continue
            ymin, xmin, ymax, xmax = boxes[idx]
            detections.append(
                [
                    xmin * scale_factor,
                    ymin * scale_factor,
                    xmax * scale_factor,
                    ymax * scale_factor,
                ]
            )

        if not detections:
            return torch.empty((0, 4), dtype=torch.float32)

        return torch.tensor(detections, dtype=torch.float32)
