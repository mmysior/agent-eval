from typing import Literal

import llm_agent_template.functions.calc as calc_functions


async def convert_speed(value: float, from_unit: Literal["rpm", "rad_s"]) -> str:
    """Convert rotational speed between RPM and rad/s.

    Args:
        value: The speed value to convert.
        from_unit: The unit of the input value. Use "rpm" to convert to rad/s,
            or "rad_s" to convert to RPM.

    Returns:
        A formatted string with the input and converted value including units.
    """
    if from_unit == "rpm":
        result = calc_functions.rpm_to_rad_s(value)
        return f"{value} RPM = {result:.4f} rad/s"
    else:
        result = calc_functions.rad_s_to_rpm(value)
        return f"{value} rad/s = {result:.4f} RPM"


async def calculate_power(torque_nm: float, rpm: float) -> str:
    """Calculate mechanical power from torque and rotational speed.

    Args:
        torque_nm: Shaft torque in Newton-metres [N·m].
        rpm: Rotational speed in revolutions per minute [RPM].

    Returns:
        A formatted string with the resulting power in both watts and kilowatts.
    """
    result = calc_functions.power_from_torque(torque_nm, rpm)
    return f"Power: {result:.2f} W ({result / 1000:.4f} kW)"


async def calculate_transmission(
    teeth_driven: int,
    teeth_driver: int,
    input_torque_nm: float,
    input_rpm: float,
    efficiency: float = 1.0,
) -> str:
    """Calculate gear stage output: ratio, torque, and speed.

    Args:
        teeth_driven: Number of teeth on the driven (output) gear.
        teeth_driver: Number of teeth on the driver (input) gear.
        input_torque_nm: Input shaft torque in Newton-metres [N·m].
        input_rpm: Input shaft speed in revolutions per minute [RPM].
        efficiency: Gear mesh efficiency as a fraction between 0 and 1. Defaults to 1.0.

    Returns:
        A formatted string with gear ratio, output torque [N·m], and output speed [RPM].
    """
    ratio = calc_functions.gear_ratio(teeth_driven, teeth_driver)
    torque_out = calc_functions.output_torque(input_torque_nm, ratio, efficiency)
    speed_out = calc_functions.output_speed(input_rpm, ratio)
    return f"Gear ratio: {ratio:.4f}\nOutput torque: {torque_out:.2f} N·m\nOutput speed: {speed_out:.2f} RPM"
