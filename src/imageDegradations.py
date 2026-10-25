import random
from dataclasses import dataclass

import cv2
import numpy as np

from imageRotation import asUint8, rotateKeepingPaper
from pipelineConfig import (
    BLUR_SIGMA_PX, CONDITION_NAMES, GAUSSIAN_NOISE_STD, ILLUMINATION_MIN_FACTOR,
    LOW_CONTRAST_RANGE, SALT_PEPPER_RATIO, SKEW_ANGLE_RANGE_DEG, WHITE_LEVEL,
)

COMBINED_SEVERITY = 0.4


@dataclass
class DegradedSample:
    image: np.ndarray
    appliedSkewDeg: float


def degradeDocument(conditionName, cleanImage, rng: random.Random) -> DegradedSample:
    if conditionName not in CONDITION_NAMES:
        raise ValueError(f"Unknown condition: {conditionName}")
    image = cleanImage.copy()
    skewDeg = 0.0
    if conditionName in ("blur", "combined"):
        image = addBlur(image, BLUR_SIGMA_PX * severity(conditionName))
    if conditionName in ("lowContrast", "combined"):
        image = reduceContrast(image, severity(conditionName))
    if conditionName in ("unevenLight", "combined"):
        image = addUnevenIllumination(image, rng, severity(conditionName))
    if conditionName in ("skew", "combined"):
        skewDeg = rng.choice((-1, 1)) * rng.uniform(*SKEW_ANGLE_RANGE_DEG)
        image = rotateKeepingPaper(image, skewDeg, expand=True)
    if conditionName in ("noise", "combined"):
        image = addNoise(image, rng, severity(conditionName))
    return DegradedSample(image, skewDeg)


def severity(conditionName):
    return COMBINED_SEVERITY if conditionName == "combined" else 1.0


def addBlur(image, sigma):
    return cv2.GaussianBlur(image, (0, 0), sigma)


def reduceContrast(image, level):
    low, high = LOW_CONTRAST_RANGE
    lowShifted = WHITE_LEVEL - (WHITE_LEVEL - low) * level
    highShifted = WHITE_LEVEL - (WHITE_LEVEL - high) * level
    return asUint8(lowShifted + image.astype(np.float32) / WHITE_LEVEL * (highShifted - lowShifted))


def addUnevenIllumination(image, rng, level):
    height, width = image.shape
    angle = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    ramp = (xx / width - 0.5) * np.cos(angle) + (yy / height - 0.5) * np.sin(angle)
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min())
    minFactor = 1.0 - (1.0 - ILLUMINATION_MIN_FACTOR) * level
    return asUint8(image * (minFactor + (1.0 - minFactor) * ramp))


def addNoise(image, rng, level):
    noiseRng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    noisy = image.astype(np.float32) + noiseRng.normal(0, GAUSSIAN_NOISE_STD * level, image.shape)
    noisy = asUint8(noisy)
    speckle = noiseRng.random(image.shape)
    noisy[speckle < SALT_PEPPER_RATIO * level / 2] = 0
    noisy[speckle > 1 - SALT_PEPPER_RATIO * level / 2] = WHITE_LEVEL
    return noisy
