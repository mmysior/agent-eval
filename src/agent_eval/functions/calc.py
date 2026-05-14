import math


def rpm_to_rad_s(rpm: float) -> float:
    """Convert rotational speed from RPM to rad/s."""
    return rpm * math.pi / 30


def rad_s_to_rpm(rad_s: float) -> float:
    """Convert rotational speed from rad/s to RPM."""
    return rad_s * 30 / math.pi


def power_from_torque(torque_nm: float, rpm: float) -> float:
    """Calculate mechanical power [W] from torque [N·m] and speed [RPM]."""
    return torque_nm * rpm_to_rad_s(rpm)


def gear_ratio(teeth_driven: int, teeth_driver: int) -> float:
    """Calculate gear ratio from tooth counts (driven / driver)."""
    return teeth_driven / teeth_driver


def output_torque(input_torque_nm: float, ratio: float, efficiency: float = 1.0) -> float:
    """Calculate output torque [N·m] given input torque, gear ratio, and mesh efficiency."""
    return input_torque_nm * ratio * efficiency


def output_speed(input_rpm: float, ratio: float) -> float:
    """Calculate output shaft speed [RPM] given input speed and gear ratio."""
    return input_rpm / ratio
