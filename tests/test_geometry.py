"""Coordinate-frame contract. These numbers are derived by hand, not from the helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.common.geometry import (
    LANE_OUTPUT_DIM,
    camera_to_ego,
    ego_to_camera,
    ground_plane_depth,
    intrinsics_matrix,
    pack_lane_array,
    project_points,
    resample_knots,
    split_lane_array,
    unproject_pixels,
)


def test_unproject_matches_the_pinhole_equation():
    k = intrinsics_matrix(500.0, 500.0, 320.0, 180.0)
    uv = np.array([[276.25, 217.5]])
    xyz = unproject_pixels(uv, np.array([20.0]), k)
    np.testing.assert_allclose(xyz[0], [-1.75, 1.5, 20.0], atol=1e-6)


def test_project_unproject_roundtrip():
    k = intrinsics_matrix(700.0, 650.0, 320.0, 180.0)
    xyz = np.array([[-2.0, 1.4, 15.0], [3.0, 1.2, 40.0]])
    pixels = project_points(xyz, k)
    recovered = unproject_pixels(pixels, xyz[:, 2], k)
    np.testing.assert_allclose(recovered, xyz, atol=1e-6)


def test_pitch_zero_mount_and_positive_pitch_looks_down():
    height = 1.5
    ahead = np.array([[0.0, 20.0, height]])
    level = ego_to_camera(ahead, camera_height_m=height, pitch_rad=0.0)
    np.testing.assert_allclose(level[0], [0.0, 0.0, 20.0], atol=1e-8)
    pitched = ego_to_camera(ahead, camera_height_m=height, pitch_rad=0.2)
    assert pitched[0, 1] < 0.0

    ground = np.array([[2.0, 20.0, 0.0], [-4.0, 8.0, 0.0]])
    camera = ego_to_camera(ground, camera_height_m=height, pitch_rad=0.15)
    back = camera_to_ego(camera, camera_height_m=height, pitch_rad=0.15)
    np.testing.assert_allclose(back, ground, atol=1e-8)


def test_ground_plane_depth_at_zero_pitch():
    k = intrinsics_matrix(500.0, 500.0, 320.0, 180.0)
    depth = ground_plane_depth(360, 640, k, camera_height_m=1.5, pitch_rad=0.0)
    # Z = height * fy / (v - cy) for a level camera.
    assert depth[180, 320] == 0.0
    np.testing.assert_allclose(depth[230, 320], 1.5 * 500.0 / 50.0, atol=1e-5)
    assert depth[200, 100] > 0.0


def test_resample_is_uniform_in_z_and_preserves_a_straight_lane():
    z = np.linspace(6.0, 40.0, 30)
    xyz = np.stack([np.full_like(z, -1.75), np.full_like(z, 1.5), z], axis=1)
    knots = resample_knots(xyz, num_knots=10)
    assert knots.shape == (10, 3)
    np.testing.assert_allclose(np.diff(knots[:, 2]), np.full(9, np.diff(knots[:, 2])[0]), atol=1e-6)
    np.testing.assert_allclose(knots[:, 0], -1.75, atol=1e-6)
    np.testing.assert_allclose(knots[:, 1], 1.5, atol=1e-6)
    assert not np.isfinite(resample_knots(xyz[:1])).any()


def test_pack_roundtrip():
    existence = np.array([1, 0, 1, 0], dtype=np.float32)
    knots = np.zeros((4, 10, 3), dtype=np.float32)
    knots[0, :, 2] = np.linspace(1, 10, 10)
    packed = pack_lane_array(existence, knots)
    assert packed.shape == (4, LANE_OUTPUT_DIM)
    exist_out, knots_out = split_lane_array(packed)
    np.testing.assert_allclose(exist_out, existence)
    np.testing.assert_allclose(knots_out, knots)


def test_bad_intrinsics_raise():
    singular = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 10.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        unproject_pixels(np.zeros((1, 2)), np.array([1.0]), singular)
