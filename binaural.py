#!/usr/bin/env python
import math
import argparse
import logging

import numpy as np
import scipy.signal

import audio

logger = logging.getLogger(__name__)


def hrtf_file(audio_path, azimuth, elevation=0, distance=1, ear_distance=0.215, output=None):
    """
    Read mono audio file and write binaural wav file to output
    """
    logger.info('Loading signal into memory: {}'.format(audio_path))
    y, sr = audio.load(audio_path)
    y = hrtf(y, sr, azimuth, elevation, distance, ear_distance)
    if output:
        audio.write_wav(output, y, sr, norm=True)
    return y


def hrtf(y, sr, azimuth, elevation=0, distance=1, ear_distance=0.215):
    """
    Take a mono signal and azimuth angle and return a stereo binaural signal

    Args:
        y: mono signal
        azimuth: angle in degrees
        sr: sample rate
    Returns:
        Binaural stereo signal (2 row np array)
    """
    ITD, d_left, d_right = compute_itd(azimuth, elevation, distance, ear_distance)
    logger.debug('ITD: {}'.format(ITD))
    left, right = apply_itd(y, y, sr, ITD)
    left, right = apply_iid(left, right, sr, azimuth, elevation, ear_distance/2, d_left, d_right)
    left = prtf(left, elevation, sr)
    right = prtf(right, elevation, sr)
    y = audio.channel_merge([left, right])
    return y


def apply_iid(left, right, sr, azimuth, elevation, radius, d_left, d_right, ref_distance=1):
    logger.info('Applying ILD to signal')
    logger.debug('d_left: {}'.format(d_left))
    logger.debug('d_right: {}'.format(d_right))
    # apply headshadow
    b, a = headshadow_filter_coefficients(azimuth+90, elevation, radius, sr)
    logger.debug('left headshadow: {}'.format([b, a]))
    left = scipy.signal.filtfilt(b, a, left)
    b, a = headshadow_filter_coefficients(azimuth-90, elevation, radius, sr)
    logger.debug('right headshadow: {}'.format([b, a]))
    right = scipy.signal.filtfilt(b, a, right)

    # attenuate for distance traveled
    logger.info('Applying attenuation')
    logger.debug('left_attenuation: {}'.format(ref_distance / d_left))
    logger.debug('right_attenuation: {}'.format(ref_distance / d_right))
    left = left * (ref_distance / d_left)
    right = right * (ref_distance / d_right)
    return left, right


def headshadow_filter_coefficients(inc_angle, elevation, r, sr):
    """
    Compute the filter coefficients to a single zero, single pole filter
    that estimates headshadow effects of a head with radius r
    """
    logger.info('Computing headshadow filter coefficients')
    theta = math.acos(math.cos(abs(inc_angle)*2*math.pi/360)*math.cos(abs(elevation)*2*math.pi/360))
    theta = theta * 360 / (2*math.pi)
    logger.debug('theta: {}'.format(theta))
    theta = math.radians(theta)
    theta0 = 2.618
    alpha_min = 0.1
    c = 343.2
    w0 = c / r
    alpha = 1 + alpha_min/2 + (1-alpha_min/2)*math.cos(theta*math.pi/theta0)
    b = [(alpha+w0/sr)/(1+w0/sr), (-alpha+w0/sr)/(1+w0/sr)]
    a = [1, -(1-w0/sr)/(1+w0/sr)]
    return b, a


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
        Distance to left ear
        Distance to right ear
    """
    logger.info('Computing ITD')
    c = 343.2
    theta = math.radians(azimuth)
    phi = abs(math.radians(elevation))
    radius = ear_distance/2

    # set theta between 0:180 degrees
    theta = abs(theta % math.pi)
    d1 = math.sqrt(distance**2 + radius**2 - 2*distance*radius*math.sin(theta))
    # inc_angle should be equivalent to pi/2 - theta, but works for values >90
    inc_angle = math.acos((distance**2 + radius**2 - d1**2) / (2*distance*radius))
    tangent = math.sqrt(distance**2 - radius**2)
    arc = radius * (math.pi - max(inc_angle, phi) - math.acos(radius / distance))
    logger.debug('arc: {}'.format(arc))
    d2 = tangent + arc
    # Use original d1 for computing d2,
    # but actual d1 may also wrap around head when distance and theta are small
    if tangent < d1:
        d1 = tangent + radius*(inc_angle - math.acos(radius / distance))
    if phi > inc_angle:
        d1 = tangent + radius*(phi - math.acos(radius / distance))
    delta_d = abs(d2 - d1)
    if -180 < azimuth < 0 or 180 < azimuth < 360:
        delta_d = -delta_d
        logger.debug(d1)
        logger.debug(d2)
        d1, d2 = d2, d1
    ITD = delta_d / c
    return ITD, d2, d1


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


def apply_itd(left, right, sr, ITD):
    logger.info('Applying ITD delay')
    if ITD > 0:
        left = audio.fractional_delay(left, ITD, sr)
    if ITD < 0:
        right = audio.fractional_delay(right, abs(ITD), sr)
    return left, right


def prtf(y, elevation, sr):
    logger.info('Applying PRTF filters')

    freq = np.interp(elevation, (-90, 90), (1000, 4500))
    mag = 10
    res1 = resonance_filter(freq, 6000, mag, sr)

    freq = np.interp(elevation, (-90, 0, 30, 90), (11000, 13000, 8000, 10000))
    mag = 10
    res2 = resonance_filter(11000, 5000, mag, sr)

    freq = np.interp(elevation, (-90, -20, 0, 90), (6000, 6000, 6500, 10000))
    mag = np.interp(elevation, (-90, -20, 0, 90), (15, 15, 10, 1))
    notch1 = notch_filter(freq, 200, mag, sr)

    freq = np.interp(elevation, (-90, 0, 90), (10000, 9000, 9000))
    mag = np.interp(elevation, (-90, -20, 0, 90), (25, 25, 20, 1))
    notch2 = notch_filter(freq, mag, 25, sr)

    freq = np.interp(elevation, (-90, 90), (10000, 14000))
    mag = np.interp(elevation, (-90, -20, 0, 30, 40, 80, 90), (10, 20, 20, 5, 20, 15, 1))
    notch3 = notch_filter(freq, 200, mag, sr)

    y1 = scipy.signal.filtfilt(*res1, y)
    y2 = scipy.signal.filtfilt(*res2, y)
    y = audio.sum_signals([y1, y2])
    y = scipy.signal.filtfilt(*notch1, y)
    y = scipy.signal.filtfilt(*notch2, y)
    y = scipy.signal.filtfilt(*notch3, y)

    return y


def resonance_filter(freq, bandwidth, magnitude, sr):
    h = 1 / (1 + math.tan(math.pi*bandwidth/sr));
    d = -math.cos(2*math.pi*freq/sr)
    V0 = 10**(magnitude/20)

    b = [V0*(1 - h), 0, V0*(h-1)]
    a = [1, 2*d*h, 2*h - 1]
    return b, a


def notch_filter(freq, bandwidth, magnitude, sr):
    d = -math.cos(2*math.pi*freq/sr)
    V0 = 10**(-magnitude/20)
    H0 = V0 - 1
    k = (math.tan(math.pi*bandwidth/sr) - V0) / (math.tan(math.pi * bandwidth/sr) + V0)

    b = [1+(1+k)*H0/2, d*(1-k), (-k - (1+k)*H0/2)]
    a = [1, d*(1-k), -k]
    return b, a


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a binaural stereo wav file from a mono audio file."
    )
    parser.add_argument('audio_path', type=str,
                        help='Path to input audio file')
    parser.add_argument('output', type=str,
                        help='Output file')
    parser.add_argument('azimuth', type=float,
                        help='Azimuth angle in degrees')
    parser.add_argument('elevation', type=float,
                        help='Elevation angle in degrees')
    parser.add_argument('distance', type=float,
                        help='Distance in meters')
    parser.add_argument('-e', '--ear-distance', type=float, default=0.215,
                        help='Distance between ears in meters')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print debug messages to stdout')
    args = parser.parse_args()
    import logging.config
    logging.config.fileConfig('logging.ini', disable_existing_loggers=False)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose debugging activated")
    del args.verbose

    hrtf_file(**vars(args))
