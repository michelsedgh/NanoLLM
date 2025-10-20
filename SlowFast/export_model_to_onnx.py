# -*- coding:utf-8 -*-
import os
import sys
from collections import OrderedDict
import torch
import torch.nn as nn
import argparse
work_root = os.path.split(os.path.realpath(__file__))[0]
from slowfast.config.defaults import get_cfg
import slowfast.utils.checkpoint as cu
from slowfast.models import build_model


class SlowFastExportWrapper(nn.Module):
    def __init__(self, model, enable_detection):
        super(SlowFastExportWrapper, self).__init__()
        self.model = model
        self.enable_detection = enable_detection

    def forward(self, slow_pathway, fast_pathway, bboxes=None):
        inputs = [slow_pathway, fast_pathway]
        if self.enable_detection:
            return self.model(inputs, bboxes)
        return self.model(inputs)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        type=str,
        default=os.path.join(
            work_root, "configs/SLOWFAST_4x16_R50_inference.yaml"),
        help="Path to the config file",
    )
    parser.add_argument(
        '--half',
        type=bool,
        default=False,
        help='use half mode',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=os.path.join(work_root,
                             "weights/checkpoint_epoch_00050.pyth"),
        help='test model file path',
    )
    parser.add_argument(
        '--save',
        type=str,
        default=os.path.join(work_root, "weights/checkpoint_epoch_00050.onnx"),
        help='save model file path',
    )
    return parser.parse_args()


def main():
    args = parser_args()
    print(args)
    cfg_file = args.cfg_file
    checkpoint_file = args.checkpoint
    save_checkpoint_file = args.save
    half_flag = args.half
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_file
    print(cfg.DATA)
    print("export pytorch model to onnx!\n")
    device = "cuda:0"
    with torch.no_grad():
        model = build_model(cfg)
        model = model.to(device)
        model.eval()
        cu.load_test_checkpoint(cfg, model)
        export_model = SlowFastExportWrapper(model, cfg.DETECTION.ENABLE).to(device)

        fast_frames = cfg.DATA.NUM_FRAMES
        slow_frames = max(1, fast_frames // cfg.SLOWFAST.ALPHA)
        crop_size = cfg.DATA.TEST_CROP_SIZE

        fast_pathway = torch.randn(1, 3, fast_frames, crop_size, crop_size, device=device)
        slow_pathway = torch.randn(1, 3, slow_frames, crop_size, crop_size, device=device)

        if half_flag:
            model.half()
            fast_pathway = fast_pathway.half()
            slow_pathway = slow_pathway.half()

        # with open(save_checkpoint_file, 'wb') as file:
        #     torch.save({"model_state": model.state_dict()}, file)

        inputs = [slow_pathway, fast_pathway]
        dummy_bboxes = None
        if cfg.DETECTION.ENABLE:
            dummy_bboxes = torch.tensor(
                [[0.0, 0.1 * crop_size, 0.1 * crop_size, 0.9 * crop_size, 0.9 * crop_size]],
                device=device,
            )
            if half_flag:
                dummy_bboxes = dummy_bboxes.half()

        for p in export_model.parameters():
            p.requires_grad = False
        dynamic_axes = {
            'slowpath': {0: 'batch'},
            'fastpath': {0: 'batch'},
            'output': {0: 'batch'},
        }
        input_names = ['slowpath', 'fastpath']
        export_inputs = (slow_pathway, fast_pathway)
        if dummy_bboxes is not None:
            input_names.append('bboxes')
            dynamic_axes['bboxes'] = {0: 'num_boxes'}
            export_inputs = (slow_pathway, fast_pathway, dummy_bboxes)

        torch.onnx.export(
            export_model,
            export_inputs,
            save_checkpoint_file,
            input_names=input_names,
            output_names=['output'],
            opset_version=12,
            dynamic_axes=dynamic_axes,
        )
        onnx_check()


def onnx_check():
    import onnx
    args = parser_args()
    print(args)
    onnx_model_path = args.save
    model = onnx.load(onnx_model_path)
    onnx.checker.check_model(model)


if __name__ == '__main__':
    main()
