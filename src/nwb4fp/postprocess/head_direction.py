import numpy as np


def calc_head_direction(positions):
    """
    Calculate head direction.

    Calculates the head direction for each position sample pair. Direction
    is defined as east = 0 degrees, north = 90 degrees, west = 180 degrees,
    south = 270 degrees. Direction is set to NaN for missing samples.
    Position matrix contains information about snout and neck. Head
    direction is the counter-clockwise direction from back LED to the front.

    Parameters:
    positions (np.array): Animal's position data, Nx5. Position data should
                          contain timestamps (1 column), X/Y coordinates of
                          left ear (2 and 3 columns correspondingly), X/Y
                          coordinates of the right ear (4 and 5 columns
                          correspondingly).
                          it is assumed that positions[:, 1:2] correspond to
                          front LED, and positions[:, 3:4] to the back LED.
                          The resulting hd is the direction from back LED to
                          the front LED.

    Returns:
    np.array: Vector of head directions in degrees.
    """

    if positions.shape[1] < 5:
        raise ValueError('Position data should be 2D (type ''help calc_head_direction'' for details).')

    x1 = positions[:, 1]
    y1 = positions[:, 2]
    x2 = positions[:, 3]
    y2 = positions[:, 4]

    hd = np.remainder(np.arctan2(y2-y1, x2-x1) * 180 / np.pi + 180, 360)
    return degrees_to_pi_range(hd)


def degrees_to_pi_range(degrees):
    radians = np.radians(degrees)
    return radians
