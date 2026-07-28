import numpy as np
import pytest

from vehicle_state_estimation.models.state import VehicleState
from vehicle_state_estimation.models.tire import fiala_lateral_force
from vehicle_state_estimation.models.vehicle import DynamicBicycleModel


def test_vehicle_state_rejects_wrong_shape():
    with pytest.raises(ValueError):
        VehicleState.from_array(np.zeros(3))


def test_zero_input_keeps_straight_line_state():
    model = DynamicBicycleModel.default()
    state = np.array([10.0, 0.0, 0.0, 0.0])
    assert np.allclose(model.derivative(state, steering=0.0, acceleration=0.0), 0.0)


def test_fiala_force_is_bounded_by_friction():
    force = fiala_lateral_force(1.2, 80000.0, 0.8, 3500.0)
    assert abs(force) <= 0.8 * 3500.0 + 1e-9
