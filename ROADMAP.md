# Roadmap — 26 October to 4 November 2026

Ten calendar days to a submission video. Dates are America/New_York calendar days.
The scaffold in this repo closes the Day 1 boxes. Later boxes are the sprint, not finished work.

Packed prediction layout used everywhere below: `[B, 4, 31]`
(4 lanes, existence logit + 10 camera-frame XYZ knots).

## Phase 1 — Ingestion and multi-model teacher

### Day 1 — Monday 26 October — scaffold

- [x] Repository layout, configs, offline Meta client, agent rules.
- [x] Camera / ego geometry and K = 10 spline resampling.
- [x] Synthetic clip, distillation smoke path, student step, ONNX wrapper.
- [ ] Confirm MPS training on the Apple Silicon laptop (this sandbox is CPU).

### Day 2 — Tuesday 27 October — real imagery

- [ ] Point `DATASET_ROOT` at nuScenes mini or a phone / dashcam clip.
- [ ] Hand prompts for frame 0 (two positive points per lane) and record them in `prompts.json`.
- [ ] Turn on SAM 3.1 Video with `META_API_MOCK=0` for one 10-second clip.
- [ ] Check temporal IDs: ego_left stays ego_left through a lane change and a occlusion.

### Day 3 — Wednesday 28 October — metric lift and audit

- [ ] Cache DINOv3 (`facebook/dinov3-vits16-pretrain-lvd1689m`) with `HUGGINGFACE_TOKEN`.
- [ ] Fit `PatchDepthHead` on the ground-plane prior, then spot-check against nuScenes LiDAR where it exists.
- [ ] Run Llama 3.2 Vision as the auditor (`META_API_MOCK=0`) and compare it with the geometric score.
- [ ] Raise `audit.min_score` until obvious z-spikes and crossed ego lanes drop out.
- [ ] Freeze a distilled JSON/NPZ split under `data/processed/`.

## Phase 2 — Student distillation and training

### Day 4 — Thursday 29 October — overfit the contract

- [ ] Train `resnet18` and a timm MobileNetV4 candidate on the synthetic clip until existence is correct.
- [ ] Log lateral error (meters) of ego knots at 10 m, 20 m, and 40 m.
- [ ] Confirm the masked Smooth L1 ignores absent adjacent lanes.

### Day 5 — Friday 30 October — real labels, MPS

- [ ] Train on the distilled split with `device=auto` on Apple Silicon.
- [ ] Hold out one sequence. Report ego-lane Chamfer distance and existence F1.
- [ ] Save `artifacts/student.pt` with the backbone name and input size inside the checkpoint.

### Day 6 — Saturday 31 October — feature distillation

- [ ] Enable `distill_dim` and MSE against frozen DINOv3 patch tokens.
- [ ] Keep the distillation loss only if ego lateral error improves on the holdout.
- [ ] Pick the backbone that still exports cleanly (ResNet18 is the safe default).

## Phase 3 — TensorRT export and Orin Nano

### Day 7 — Sunday 1 November — ONNX parity

- [ ] `python -m src.export.export_onnx --checkpoint artifacts/student.pt`.
- [ ] ONNX Runtime vs PyTorch max abs error under 1e-3 on a fixed batch.
- [ ] Lock input to `1×3×288×512` RGB in `[0, 1]` (normalization stays in the graph).

### Day 8 — Monday 2 November — Jetson Orin Nano

- [ ] `trtexec --onnx=artifacts/student.onnx --saveEngine=artifacts/student_fp16.engine --fp16`.
- [ ] `src/edge/trt_runner.py` on device. Record p50 / p95 latency at batch 1.
- [ ] If FP16 drifts, fall back to the ONNX Runtime path for the demo and note it.

## Phase 4 — Video demo

### Day 9 — Tuesday 3 November — dual view

- [ ] Record a front-camera sequence through the visualizer (overlay + ego BEV).
- [ ] Cut in one failure case the auditor rejects (occlusion or a z-spike).
- [ ] Show the teacher label and the student prediction on the same frame.

### Day 10 — Wednesday 4 November — submission

- [ ] Freeze weights and the engine hash in the README.
- [ ] One-page architecture recap: SAM 3.1 + DINOv3 + Llama teacher, student, Orin.
- [ ] Rehearse the CPU / ONNX Runtime backup in case the Orin misbehaves on stage.
- [ ] Submit before the deadline. Do not retrain that afternoon.

## Exit criteria

- A single monocular frame goes in. Up to four 3D lane splines come out.
- Ego left is to the left of ego right in the camera frame, on the ground plane, with Z increasing.
- The Orin demo or the documented ORT fallback renders the dual view at interactive rate.
