#!/usr/bin/env python3
"""
Export a TF Hub MoViNet streaming model as a SavedModel with explicit state inputs/outputs.

This wrapper exposes two signatures:
 - init_states(input_shape: int32[5]) -> states dict
 - serving_default(image: float32[1,1,H,W,3], states: dict) -> { logits, updated states... }

Notes:
 - Works on CPU or GPU and on macOS (no NVIDIA required for export).
 - Choose H=W via --image_size. For MoViNet A0 streaming, use 172; for A5, use 320.
 - Uses TF Hub streaming model endpoint: movinet/{model_id}/stream/kinetics-600/classification/{version}
"""

import argparse
from pathlib import Path
from typing import Dict

import tensorflow as tf
import tensorflow_hub as hub


def build_model_url(model_id: str, version: int) -> str:
    return (
        f"https://tfhub.dev/tensorflow/movinet/{model_id}/stream/"
        f"kinetics-600/classification/{version}"
    )


class MovinetStreamingWrapper(tf.Module):
    def __init__(self, hub_model: tf.types.experimental.GenericFunction, image_size: int):
        super().__init__()
        self._model = hub_model
        self._image_size = int(image_size)

        # Prepare a concrete shape to derive state specs
        input_shape_tensor = tf.constant(
            [1, 1, self._image_size, self._image_size, 3], dtype=tf.int32
        )
        example_states: Dict[str, tf.Tensor] = self._model.init_states(input_shape_tensor)

        # Build TensorSpecs for the states so we can expose them in the SavedModel signature
        self._state_specs: Dict[str, tf.TensorSpec] = {
            name: tf.TensorSpec(shape=t.shape, dtype=t.dtype, name=name)
            for name, t in example_states.items()
        }

        # Pre-create concrete functions so saving does not retrace unexpectedly
        self._concrete_init = self.init_states.get_concrete_function()

        # Build concrete signature for inference using dynamic spatial dims
        image_spec = tf.TensorSpec(
            shape=[1, 1, None, None, 3], dtype=tf.float32, name="image"
        )
        states_spec = {k: v for k, v in sorted(self._state_specs.items())}
        self._concrete_infer = self.infer.get_concrete_function(image_spec, states_spec)

    @tf.function(input_signature=[tf.TensorSpec(shape=[5], dtype=tf.int32, name="input_shape")])
    def init_states(self, input_shape: tf.Tensor) -> Dict[str, tf.Tensor]:
        return self._model.init_states(input_shape)

    @tf.function
    def infer(self, image: tf.Tensor, states: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        inputs = dict(states)
        inputs["image"] = image
        logits, new_states = self._model(inputs)
        outputs: Dict[str, tf.Tensor] = {"logits": logits}
        outputs.update(new_states)
        return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export MoViNet streaming (TF Hub) to SavedModel with explicit state IO"
        )
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="a0",
        help="MoViNet model id (e.g., a0, a1, a2, a3, a4, a5)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=172,
        help="Square input size H=W for streaming (a0=172, a5=320, etc.)",
    )
    parser.add_argument(
        "--hub_version",
        type=int,
        default=3,
        help="TF Hub model version to load (default: 3)",
    )
    parser.add_argument(
        "--export_path",
        type=str,
        default="./movinet_stream_saved_model",
        help="Output directory for SavedModel",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_url = build_model_url(args.model_id, args.hub_version)
    print(f"📥 Loading MoViNet streaming model from TF Hub: {model_url}")
    hub_model = hub.load(model_url)
    print("✅ TF Hub model loaded")

    print("🔧 Wrapping model to expose explicit state inputs/outputs…")
    wrapper = MovinetStreamingWrapper(hub_model, image_size=args.image_size)

    export_dir = Path(args.export_path)
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving SavedModel to: {export_dir}")
    tf.saved_model.save(
        wrapper,
        str(export_dir),
        signatures={
            "init_states": wrapper._concrete_init,
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: wrapper._concrete_infer,
        },
    )
    print("✅ SavedModel export complete")

    # Optional: reload to show signatures
    reloaded = tf.saved_model.load(str(export_dir))
    available = list(reloaded.signatures.keys())
    print(f"🔎 Available signatures: {available}")


if __name__ == "__main__":
    main()


