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

import numpy as np
import pytest

from openarm_driver.safety import (
    JointDeltaPosChecker,
    JointPosChecker,
    JointVelocityChecker,
)


@pytest.mark.parametrize(
    ("checker_type", "limits"),
    [
        (JointPosChecker, [[-1.0, 1.0], [-1.0, np.nan]]),
        (JointDeltaPosChecker, [1.0, np.nan]),
        (JointVelocityChecker, [1.0, np.inf]),
    ],
)
def test_joint_checkers_reject_non_finite_limits(checker_type, limits):
    with pytest.raises(ValueError, match="finite"):
        checker_type(limits)


@pytest.mark.parametrize(
    ("checker_type", "limits"),
    [
        (JointPosChecker, [-1.0, 1.0]),
        (JointDeltaPosChecker, [[1.0, 1.0]]),
        (JointVelocityChecker, [[1.0, 1.0]]),
    ],
)
def test_joint_checkers_reject_invalid_limit_shapes(checker_type, limits):
    with pytest.raises(ValueError):
        checker_type(limits)


def test_joint_position_checker_rejects_reversed_limits():
    with pytest.raises(ValueError, match="minimums"):
        JointPosChecker([[1.0, -1.0]])
