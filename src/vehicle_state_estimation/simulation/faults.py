from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FaultManager:
    sensors: list[str]
    _health: dict[str, bool] = field(init=False)

    def __post_init__(self) -> None:
        if not self.sensors or len(set(self.sensors)) != len(self.sensors):
            raise ValueError("sensors must be a non-empty list of unique names")
        self._health = dict.fromkeys(self.sensors, True)

    def inject(self, sensor: str, fault: str) -> None:
        if sensor not in self._health or fault not in {"dropout", "bias", "delay"}:
            raise ValueError("unknown sensor or fault type")
        self._health[sensor] = False

    def recover(self, sensor: str) -> None:
        if sensor not in self._health:
            raise ValueError("unknown sensor")
        self._health[sensor] = True

    def is_healthy(self, sensor: str) -> bool:
        if sensor not in self._health:
            raise ValueError("unknown sensor")
        return self._health[sensor]

    def health_report(self) -> dict[str, bool]:
        return self._health.copy()
