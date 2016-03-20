import math

import numpy as np
import librosa


def hrtf_file(audio_path, azimuth, elevation=0, output=None):
    """
    Read mono audio file and write binaural wav file to output
    """
    y, sr = librosa.load(audio_path)
    y = hrtf(y, sr, azimuth, elevation)
    if output:
        librosa.output.write_wav(output, y, sr, norm=False)
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
    y = apply_itd(y, ITD, sr)
    return y


def compute_itd(azimuth, elevation=0, ear_distance=0.215):
    """
    Compute the Interaural Time Difference given the azimuth angle
    and distance between ears.

    Args:
        azimuth: Angle in degrees (-180 < θ < 180)
        elevation: Angle in degrees (-90 < θ < 90)
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
    ITD = (radius/c) * (math.asin(math.cos(phi)*math.sin(theta)) + math.cos(phi)*math.sin(theta))
    return ITD


def apply_itd(y, ITD, sr):
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
    y = np.pad(y, pad_width=[round(time*sr), 0], mode='constant', constant_values=0)
    return y
