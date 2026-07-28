from __future__ import annotations

import math


def fiala_lateral_force(
    slip_angle: float,
    cornering_stiffness: float,
    friction: float,
    normal_load: float,
) -> float:
    """Return a bounded Fiala lateral force for one tire."""
    if cornering_stiffness <= 0 or friction <= 0 or normal_load <= 0:
        raise ValueError("tire parameters must be positive")
    limit = friction * normal_load
    tan_alpha = math.tan(float(slip_angle))
    alpha_sl = math.atan(3.0 * limit / cornering_stiffness)
    if abs(slip_angle) >= alpha_sl:
        return -math.copysign(limit, slip_angle)
    ca = cornering_stiffness
    term = -ca * tan_alpha
    term += (ca**2 / (3.0 * limit)) * abs(tan_alpha) * tan_alpha
    term -= (ca**3 / (27.0 * limit**2)) * tan_alpha**3
    return float(max(-limit, min(limit, term)))
