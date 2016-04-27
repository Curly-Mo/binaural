#!/usr/bin/env python
import math
import argparse

import audio


def hrtf_file(audio_path, azimuth, elevation=0, distance=1, output=None):
    """
    Read mono audio file and write binaural wav file to output
    """
    y, sr = audio.load(audio_path)
    y = hrtf(y, sr, azimuth, elevation, distance)
    if output:
        audio.write_wav(output, y, sr, norm=False)
    return y


def hrtf(y, sr, azimuth, elevation=0, distance=1):
    """
    Take a mono signal and azimuth angle and return a stereo binaural signal

    Args:
        y: mono signal
        azimuth: angle in degrees
        sr: sample rate
    Returns:
        Binaural stereo signal (2 row np array)
    """
    ITD = compute_itd(azimuth, elevation, distance)
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

    # set theta between -180:180 degrees
    theta = theta % math.pi
    d1 = math.sqrt(distance**2 + radius**2 - 2*distance*radius*math.sin(abs(theta)))
    # inc_angle should be equivalent to pi/2 - theta, but works for values >90
    inc_angle = math.acos((distance**2 + radius**2 - d1**2) / (2*distance*radius))
    tangent = math.sqrt(distance**2 - radius**2)
    d2 = tangent + radius * (math.pi - inc_angle - math.acos(radius / distance))
    # Use original d1 for computing d2,
    # but actual d1 may also wrap around head when distance and theta are small
    if tangent < d1:
        d1 = tangent + radius*(inc_angle - math.acos(radius / distance))
    delta_d = abs(d2 - d1)
    if -180 < azimuth < 0 or 180 < azimuth < 360:
        delta_d = -delta_d
    ITD = delta_d / c
    return ITD


def compute_itd_legacy(azimuth, elevation=0, distance=1, ear_distance=0.215):
    """
    Legacy code for future reference
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
    # Colin's Formula 2
    #IDD = radius * (math.pi - 2*(math.pi/2 - theta))
    #IDD = radius * (2*theta)
    #ITD = IDD / c
    return ITD


def apply_itd(y, sr, ITD):
    left = y
    right = y
    if ITD > 0:
        left = audio.delay(y, ITD, sr)
    if ITD < 0:
        right = audio.delay(y, abs(ITD), sr)
    y = audio.channel_merge([left, right])
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
    parser.add_argument('distance', type=float,
                        help='Distance in meters')
    parser.add_argument('output', type=str,
                        help='Output file')
    args = parser.parse_args()

    hrtf_file(**vars(args))
