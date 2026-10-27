# Agent notes

Engineering rules for this repo live in [`.cursorrules`](.cursorrules).
The invariants below are repeated so a reviewer can check them without opening that file.

## Deadline

Meta AI Worldwide Developer Hackathon. Freeze the demo on **4 November 2026**.
Sprint checklist: [`ROADMAP.md`](ROADMAP.md).

## Non-negotiables

- Python 3.10+. Device order is MPS, then CUDA, then CPU (`src/common/device.py`).
- Camera optical frame: **+X right, +Y down, +Z forward**. Student knots are in this frame.
- Ego frame: **+X right, +Y forward, +Z up**. Origin on the ground under the camera.
- Spline head: **M = 4** lanes × (**1 existence logit + K = 10 XYZ knots**) = tensor `[B, 4, 31]`.
  Slot order is ego_left, ego_right, left_adjacent, right_adjacent.
- Meta SAM 3.1 Video, DINOv3, and Llama 3.2 Vision are integrated behind offline mocks.
  `META_API_MOCK` defaults to on. DINOv3 loads with `local_files_only=True` unless asked.
- Do not call external APIs, download weights, or require a display from tests or scaffolds.

## Where the math lives

| Concern | Module |
| --- | --- |
| Unprojection, ego extrinsics, knot resampling | `src/common/geometry.py` |
| `SplineKnot3D`, `Lane3D`, `Frame3DAnnotation` | `src/common/types.py` |
| SAM / Llama request payloads | `src/teacher/meta_client.py` |
| DINOv3 features and metric depth fallback | `src/teacher/dinov3_depth.py` |
| Mask + depth → 3D knots | `src/teacher/lifter.py` |
| Geometric and VLM audit | `src/teacher/auditor.py` |
| Student, masked Smooth L1, MPS loop | `src/student/` |
| ONNX and TensorRT FP16 | `src/export/`, `src/edge/trt_runner.py` |

Run `pytest` from the repo root after changing geometry, the lifter, the loss, or the tensor layout.
