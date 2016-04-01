#!/usr/bin/env python
import math
import argparse

import numpy as np
import audio


def hrtf_file(audio_path, azimuth, elevation=0, output=None):
    """
    Read mono audio file and write binaural wav file to output
    """
    y, sr = audio.load(audio_path)
    y = hrtf(y, sr, azimuth, elevation)
    if output:
        audio.write_wav(output, y, sr, norm=False)
    return y


def hrtf(y, sr, azimuth, elevation=0):
    """
    Take a mono signal and azimuth angle and return a stereo binaural signal

    Args:
        y: mono signal
        azimuth: angle in degrees
        sr: sample rate
    Returns:
        Binaural stereo signal (2 row np array)
    """
    ITD = compute_itd(azimuth, elevation)
    print(ITD)
    y = apply_itd(y, sr, ITD)
    return y


def compute_itd(azimuth, elevation=0, distance=1, ear_distance=0.215):
    """
    Compute the Interaural Time Difference given the azimuth angle
    and distance between ears.

    Args:
        azimuth: Angle in degrees (-180 < θ < 180)
        elevation: Angle in degrees (-90 < θ < 90)
        distance: Distance of source from listener in meters
        ear_distance: distance between ears in meters
    Returns:
        Interaural Time Difference (ITD)
    """
    c = 343.2
    theta = math.radians(azimuth)
    phi = math.radians(elevation)
    radius = ear_distance/2
    # Woodworth's formula
    # ITD = (radius/c) * (math.sin(theta) + theta)
    # Woodworth's formula with elevation
    # ITD = (radius/c) * (math.sin(theta) + theta) * math.cos(phi)
    # Larcher and Jot equation
    # ITD = (radius/c) * (math.asin(math.cos(phi)*math.sin(theta)) + math.cos(phi)*math.sin(theta))
    # Colin's Formula
    distance_r = math.sqrt(distance**2 + radius**2 - 2*distance*radius*math.sin(-theta))
    distance_l = math.sqrt(distance**2 + radius**2 - 2*distance*radius*math.sin(theta))
    IDD = distance_r - distance_l
    ITD = IDD / c
    return ITD


def apply_itd(y, sr, ITD):
    left = y
    right = y
    if ITD > 0:
        left = delay(y, ITD, sr)
    if ITD < 0:
        right = delay(y, abs(ITD), sr)
    if len(left) > len(right):
        right = np.zeros(len(left))
        right[0:len(y)] = y
    if len(right) > len(left):
        left = np.zeros(len(right))
        left[0:len(y)] = y
    y = np.vstack([left, right])
    return y


def delay(y, time, sr):
    """
    Prepend zeros to delay signal by time seconds
    """
    y = np.pad(
        y,
        pad_width=[round(time*sr), 0],
        mode='constant',
        constant_values=0
    )
    return y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a binaural stereo wav file from a mono audio file."
    )
    parser.add_argument('audio_path', type=str,
                        help='Path to input audio file')
    parser.add_argument('azimuth', type=float,
                        help='Azimuth angle in degrees')
    parser.add_argument('elevation', type=float,
                        help='Elevation angle in degrees')
    parser.add_argument('output', type=str,
                        help='Output file')
    args = parser.parse_args()

    hrtf_file(**vars(args))
