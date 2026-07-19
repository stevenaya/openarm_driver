# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Safety checkers for OpenArm robot control."""

import numpy as np
from numpy.typing import ArrayLike

from .base_safety import Checker, CheckResult


def _validate_joint_positions(
    joint_positions: ArrayLike,
    expected_shape: tuple[int, ...],
    *,
    label: str,
) -> np.ndarray:
    positions = np.asarray(joint_positions, dtype=float)
    if positions.shape != expected_shape:
        raise ValueError(
            f"{label} must have shape {expected_shape}, got {positions.shape}."
        )
    if not np.all(np.isfinite(positions)):
        raise ValueError(f"{label} must contain only finite values.")
    return positions


class JointPosChecker(Checker):
    """Check that joint positions are within limits."""

    def __init__(self, joint_limits: ArrayLike):
        """Initialize joint position checker.

        Args:
            joint_limits: (8, 2) array of [min, max] limits.

        """
        self.joint_limits = np.asarray(joint_limits, dtype=float)
        if (
            self.joint_limits.ndim != 2
            or self.joint_limits.shape[1] != 2
            or not np.all(np.isfinite(self.joint_limits))
        ):
            raise ValueError("Joint position limits must be finite [min, max] pairs.")
        if np.any(self.joint_limits[:, 0] > self.joint_limits[:, 1]):
            raise ValueError("Joint position minimums must not exceed maximums.")

    def check(self, joint_positions: ArrayLike, **kwargs) -> CheckResult:
        """Run check."""
        expected_shape = (self.joint_limits.shape[0],)
        positions = _validate_joint_positions(
            joint_positions,
            expected_shape,
            label="Joint positions",
        )
        low = self.joint_limits[:, 0]
        high = self.joint_limits[:, 1]

        position_limited = np.clip(positions, low, high)
        violations = (positions < low) | (positions > high)

        if np.any(violations):
            violated_joints = np.where(violations)[0].tolist()
            return CheckResult(
                is_safe=False,
                force_stop=False,
                fixed_joint_positions=position_limited,
                message=f"Joint positions over limits at joints: {violated_joints}",
                check_type="joint_limits",
                details={"violated_joints": violated_joints},
            )

        return CheckResult(
            is_safe=True,
            message="All joint positions within limits.",
            check_type="joint_limits",
        )


class JointDeltaPosChecker(Checker):
    """Check that joint position changes don't exceed velocity limits."""

    def __init__(self, delta_limits: ArrayLike):
        """Initialize delta position checker.

        Args:
            delta_limits: Maximum allowed change per step for each joint.

        """
        self.delta_limits = np.asarray(delta_limits, dtype=float)
        if (
            self.delta_limits.ndim != 1
            or not np.all(np.isfinite(self.delta_limits))
            or np.any(self.delta_limits <= 0.0)
        ):
            raise ValueError("Joint delta limits must be finite and positive.")

    def check(self, joint_positions: ArrayLike, **kwargs) -> CheckResult:
        """Run check."""
        positions = _validate_joint_positions(
            joint_positions,
            self.delta_limits.shape,
            label="Joint positions",
        )

        driver = kwargs.get("driver")
        if driver is None or not hasattr(driver, "last_command"):
            return CheckResult(
                is_safe=True,
                message="No previous command to compare against.",
                check_type="joint_delta",
            )

        previous = _validate_joint_positions(
            driver.last_command,
            self.delta_limits.shape,
            label="Previous joint positions",
        )

        delta = positions - previous

        for i, d in enumerate(delta):
            if abs(d) > self.delta_limits[i]:
                return CheckResult(
                    is_safe=False,
                    force_stop=True,
                    message=(
                        f"Joint {i} delta {d:.4f} exceeds limit {self.delta_limits[i]:.4f}"
                    ),
                    check_type="joint_delta",
                    details={
                        "joint": i,
                        "delta": float(d),
                        "limit": float(self.delta_limits[i]),
                    },
                )

        return CheckResult(
            is_safe=True,
            message="All joint deltas within limits.",
            check_type="joint_delta",
        )
