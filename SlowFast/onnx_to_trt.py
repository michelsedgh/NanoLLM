import os
import argparse
import ctypes
import importlib.util
from onnx import ModelProto


def _parse_trt_version(ver_str):
    parts = []
    for token in ver_str.split('.'):
        digits = ''.join(ch for ch in token if ch.isdigit())
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts)
work_root = os.path.split(os.path.realpath(__file__))[0]

def _load_trt_plugins():
    candidates = []
    env_path = os.environ.get("TRT_PLUGIN_LIBRARY")
    if env_path:
        candidates.append(env_path)
    spec = importlib.util.find_spec("tensorrt")
    if spec and spec.origin:
        package_dir = os.path.dirname(spec.origin)
        candidates.append(os.path.join(package_dir, "libnvinfer_plugin.so"))
    candidates.append("/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so")
    candidates.append("/usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so")

    for path in candidates:
        if not path:
            continue
        if not os.path.exists(path):
            continue
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError:
            continue

    import tensorrt as trt_module

    logger = trt_module.Logger(trt_module.Logger.INFO)
    trt_module.init_libnvinfer_plugins(logger, "")
    return trt_module


trt = _load_trt_plugins()
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
"""SlowFast onnx模型转trt模型"""


def build_engine_trt8(onnx_path, shapes, precision_mode=True, max_batch_size=8):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        if precision_mode:
            config.set_flag(trt.BuilderFlag.FP16)
        workspace_size = 16 * (1 << 20)
        if _parse_trt_version(trt.__version__) >= (10, 0, 0):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
        else:
            config.max_workspace_size = workspace_size
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for idx in range(parser.num_errors):
                    err = parser.get_error(idx)
                    print(err)
                raise RuntimeError("Failed to parse ONNX model")
        for idx in range(len(shapes)):
            input_tensor = network.get_input(idx)
            min_shape = list(shapes[idx])
            if min_shape:
                min_shape[0] = 1
            opt_shape = tuple(min_shape)
            max_shape = list(min_shape)
            if max_shape:
                max_shape[0] = max_batch_size
            profile.set_shape(
                input_tensor.name,
                tuple(min_shape),
                opt_shape,
                tuple(max_shape))
        config.add_optimization_profile(profile)

        if _parse_trt_version(trt.__version__) >= (10, 0, 0):
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError("Failed to build serialized network")
            engine = trt_runtime.deserialize_cuda_engine(serialized)
            return engine
        else:
            engine = builder.build_engine(network, config)
            return engine


def build_engine_trt7(onnx_path, shapes, precision_mode=True):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        #if builder.platform_has_fast_fp16 and precision_mode == 16:
        if hasattr(builder, "fp16_mode"):
            builder.fp16_mode = precision_mode
        builder.max_workspace_size = 16 * (1 << 20)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        for idx in range(len(shapes)):
            network.get_input(idx).shape = shapes[idx]
        engine = builder.build_cuda_engine(network)
        return engine


def build_engine(onnx_path, shapes, precision_mode=True, max_batch_size=8):
    if _parse_trt_version(trt.__version__) >= (8, 0, 0):
        return build_engine_trt8(onnx_path, shapes, precision_mode, max_batch_size)
    else:
        return build_engine_trt7(onnx_path, shapes, precision_mode)


def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)


def load_engine(trt_runtime, plan_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


def parser_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--onnx_model',
        type=str,
        default=os.path.join(work_root,
                             "onnx_models/checkpoint_epoch_00050.onnx"),
        help='test model file path',
    )
    parser.add_argument(
        '--trt_model',
        type=str,
        default=os.path.join(work_root, "trt_models/checkpoint_epoch_00050.engine"),
        help='tensorrt model file path',
    )
    return parser.parse_args()


def main():
    args = parser_args()
    print(args)
    onnx_path = args.onnx_model
    engine_name = args.trt_model

    model = ModelProto()
    with open(onnx_path, "rb") as f:
        model.ParseFromString(f.read())
    onnx_input = model.graph.input
    # print(onnx_input)
    input_shapes = []
    for i in range(len(onnx_input)):
        input_shape = []
        input_dim = len(onnx_input[i].type.tensor_type.shape.dim)
        for j in range(input_dim):
            dim = onnx_input[i].type.tensor_type.shape.dim[j].dim_value
            input_shape.append(dim)
        if len(input_shape):
            input_shapes.append(input_shape)
    print(input_shapes)  # [[1, 3, 4, 256, 256], [1, 3, 32, 256, 256]]

    engine = build_engine(onnx_path, input_shapes)
    save_engine(engine, engine_name)


if __name__ == '__main__':
    main()
    
