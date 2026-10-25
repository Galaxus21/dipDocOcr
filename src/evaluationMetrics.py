import re

import jiwer
import numpy as np
from pipelineConfig import PSNR_CAP_DB
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

WHITESPACE_PATTERN = re.compile(r"\s+")


def characterErrorRate(reference, hypothesis):
    # spacing differences are not character errors
    strippedReference = WHITESPACE_PATTERN.sub("", reference)
    strippedHypothesis = WHITESPACE_PATTERN.sub("", hypothesis)
    return jiwer.cer(strippedReference, strippedHypothesis) if strippedHypothesis else 1.0


def wordErrorRate(reference, hypothesis):
    collapsedReference = WHITESPACE_PATTERN.sub(" ", reference).strip()
    collapsedHypothesis = WHITESPACE_PATTERN.sub(" ", hypothesis).strip()
    return jiwer.wer(collapsedReference, collapsedHypothesis) if collapsedHypothesis else 1.0


def imageFidelity(processed, reference):
    """PSNR/SSIM vs the clean page; NaN if the size changed (skew)."""
    if processed.shape != reference.shape:
        return float("nan"), float("nan")
    psnr = min(PSNR_CAP_DB, peak_signal_noise_ratio(reference, processed, data_range=255))
    ssim = structural_similarity(reference, processed, data_range=255)
    return float(psnr), float(ssim)


def skewErrorDegrees(appliedSkewDeg, estimatedSkewDeg):
    return float(np.abs(appliedSkewDeg - estimatedSkewDeg))
