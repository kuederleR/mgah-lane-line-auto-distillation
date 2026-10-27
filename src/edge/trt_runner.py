"""TensorRT FP16 harness for Jetson Orin Nano, with an ONNX Runtime fallback.

``trtexec`` builds the engine on device. This module does not download or
build an engine during import. ``infer`` uses TensorRT when an engine and
the TensorRT Python bindings are present, and otherwise runs ONNX Runtime
on the CPU (or whatever providers ORT was installed with).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class TensorRTRunner:
    """Execute the exported student. Input is RGB float NCHW in ``[0, 1]``."""

    def __init__(
        self,
        engine_path: str | Path | None = None,
        onnx_path: str | Path | None = None,
        precision: str = "fp16",
        workspace_mb: int = 1024,
    ) -> None:
        self.engine_path = Path(engine_path) if engine_path else None
        self.onnx_path = Path(onnx_path) if onnx_path else None
        self.precision = precision
        self.workspace_mb = workspace_mb
        self.backend: str | None = None
        self._session = None
        self._engine = None
        self._context = None

    def load(self) -> str:
        if self._tensorrt_ready():
            self._load_tensorrt()
            self.backend = "tensorrt"
            return self.backend
        if self.onnx_path is not None and self.onnx_path.is_file():
            self._load_onnxruntime()
            self.backend = "onnxruntime"
            return self.backend
        raise FileNotFoundError(
            "no TensorRT engine could be loaded and no ONNX file was provided"
        )

    def infer(self, images: np.ndarray) -> np.ndarray:
        """Return ``[N, 4, 31]`` float32 lane tensors."""
        if self.backend is None:
            self.load()
        array = np.ascontiguousarray(images, dtype=np.float32)
        if array.ndim != 4 or array.shape[1] != 3:
            raise ValueError(f"expected NCHW RGB, got {array.shape}")
        if self.backend == "onnxruntime":
            outputs = self._session.run(None, {"images": array})
            return np.asarray(outputs[0], dtype=np.float32)
        return self._infer_tensorrt(array)

    @staticmethod
    def trtexec_command(
        onnx_path: str | Path,
        engine_path: str | Path,
        precision: str = "fp16",
        workspace_mb: int = 1024,
    ) -> list[str]:
        """Command line for the Jetson ``trtexec`` build. Not executed here."""
        command = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            f"--memPoolSize=workspace:{workspace_mb}",
        ]
        if precision == "fp16":
            command.append("--fp16")
        elif precision == "int8":
            command.append("--int8")
        elif precision != "fp32":
            raise ValueError(f"unsupported precision {precision!r}")
        return command

    def _tensorrt_ready(self) -> bool:
        if self.engine_path is None or not self.engine_path.is_file():
            return False
        try:
            import tensorrt  # noqa: F401
        except ImportError:
            logger.info("tensorrt is not installed; using ONNX Runtime if an ONNX file exists")
            return False
        return True

    def _load_onnxruntime(self) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(self.onnx_path),
            providers=["CPUExecutionProvider"],
        )

    def _load_tensorrt(self) -> None:
        import tensorrt as trt

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"TensorRT could not deserialize {self.engine_path}")
        self._engine = engine
        self._context = engine.create_execution_context()

    def _infer_tensorrt(self, images: np.ndarray) -> np.ndarray:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT execution needs a CUDA device")
        context = self._context
        engine = self._engine
        inp = torch.from_numpy(images).cuda()
        input_name, output_name = _io_names(engine)
        if hasattr(context, "set_input_shape"):
            context.set_input_shape(input_name, tuple(inp.shape))
        out_shape = tuple(context.get_tensor_shape(output_name)) if hasattr(context, "get_tensor_shape") else None
        if not out_shape or any(dim < 0 for dim in out_shape):
            out_shape = (images.shape[0], 4, 31)
        out = torch.empty(out_shape, dtype=torch.float32, device="cuda")
        if hasattr(context, "set_tensor_address") and hasattr(context, "execute_async_v3"):
            context.set_tensor_address(input_name, inp.data_ptr())
            context.set_tensor_address(output_name, out.data_ptr())
            stream = torch.cuda.current_stream().cuda_stream
            ok = context.execute_async_v3(stream)
            if ok is False:
                raise RuntimeError("TensorRT execute_async_v3 returned false")
        else:
            raise RuntimeError(
                "this TensorRT build does not expose execute_async_v3; "
                "use the ONNX Runtime fallback or update the Jetson JetPack bindings"
            )
        torch.cuda.current_stream().synchronize()
        return out.detach().cpu().numpy()


def _io_names(engine) -> tuple[str, str]:
    inputs = []
    outputs = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        # TensorRT 10: TensorIOMode.INPUT / OUTPUT.
        if "INPUT" in str(mode):
            inputs.append(name)
        else:
            outputs.append(name)
    if not inputs or not outputs:
        raise RuntimeError("TensorRT engine is missing an input or output tensor")
    return inputs[0], outputs[0]


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run the lane student engine")
    parser.add_argument("--engine", type=Path, default=None)
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument("--print-trtexec", action="store_true")
    args = parser.parse_args(argv)
    runner = TensorRTRunner(engine_path=args.engine, onnx_path=args.onnx, precision=args.precision)
    if args.print_trtexec:
        if args.onnx is None or args.engine is None:
            raise SystemExit("--print-trtexec needs --onnx and --engine")
        print(" ".join(runner.trtexec_command(args.onnx, args.engine, args.precision)))
        return
    if args.onnx is None and args.engine is None:
        raise SystemExit("pass --onnx or --engine")
    runner.load()
    session_input = None
    if runner.backend == "onnxruntime":
        session_input = runner._session.get_inputs()[0].shape
    shape = _concrete_shape(session_input)
    dummy = np.zeros(shape, dtype=np.float32)
    lanes = runner.infer(dummy)
    logger.info("backend=%s output=%s", runner.backend, tuple(lanes.shape))


def _concrete_shape(shape: list | None) -> tuple[int, int, int, int]:
    if not shape or any(not isinstance(dim, int) or dim <= 0 for dim in shape):
        return (1, 3, 288, 512)
    return tuple(int(dim) for dim in shape)  # type: ignore[return-value]


if __name__ == "__main__":
    main()
