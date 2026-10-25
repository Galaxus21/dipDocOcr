import cv2
import numpy as np
from skimage.restoration import estimate_sigma

from imageRotation import asUint8, rotateKeepingPaper
from pipelineConfig import (
    ADAPTIVE_BLOCK_SIZE, ADAPTIVE_OFFSET, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID,
    DESKEW_COARSE_STEP_DEG, DESKEW_FINE_STEP_DEG, DESKEW_MIN_CORRECTION_DEG, DESKEW_PREVIEW_SCALE,
    DESKEW_SEARCH_RANGE_DEG, ILLUMINATION_CLOSE_KERNEL, ILLUMINATION_MEDIAN_KERNEL,
    MEDIAN_DENOISE_KERNEL, MORPHOLOGY_KERNEL, NLM_MAX_STRENGTH, NLM_MIN_STRENGTH,
    NLM_SEARCH_WINDOW, NLM_SIGMA_SCALE, NLM_TEMPLATE_WINDOW, NOISE_SKIP_SIGMA, STRETCH_HIGH_PERCENTILE, STRETCH_LOW_PERCENTILE,
)

ESTIMATED_SKEW_KEY = "estimatedSkewDeg"


def toGray(image, context):
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def otsuBinarize(image, context):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def denoise(image, context):
    # smoothing thins strokes, so skip clean pages
    if estimate_sigma(image) < NOISE_SKIP_SIGMA:
        return image
    # median first: NLM smears salt-and-pepper pixels instead of removing them
    despeckled = cv2.medianBlur(image, MEDIAN_DENOISE_KERNEL)
    strength = float(np.clip(estimate_sigma(despeckled) * NLM_SIGMA_SCALE, NLM_MIN_STRENGTH, NLM_MAX_STRENGTH))
    return cv2.fastNlMeansDenoising(
        despeckled, None, strength, NLM_TEMPLATE_WINDOW, NLM_SEARCH_WINDOW
    )


def illumination(image, context):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ILLUMINATION_CLOSE_KERNEL, ILLUMINATION_CLOSE_KERNEL)
    )
    background = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    background = cv2.medianBlur(background, ILLUMINATION_MEDIAN_KERNEL)
    return cv2.divide(image, background, scale=255)


def stretch(image, context):
    # median = paper level (-> white); text is <1% of pixels, so the low anchor is a tiny percentile
    low, high = np.percentile(image, (STRETCH_LOW_PERCENTILE, STRETCH_HIGH_PERCENTILE))
    if high <= low:
        return image
    return asUint8((image.astype(np.float32) - low) / (high - low) * 255)


def clahe(image, context):
    return cv2.createCLAHE(CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID).apply(image)


def deskew(image, context):
    correctionDeg = estimateCorrectionAngle(image)
    context[ESTIMATED_SKEW_KEY] = -correctionDeg
    # rotating blurs edges, so leave near-straight pages alone
    if abs(correctionDeg) < DESKEW_MIN_CORRECTION_DEG:
        return image
    return rotateKeepingPaper(image, correctionDeg, expand=False)


def estimateCorrectionAngle(image):
    preview = cv2.resize(image, None, fx=DESKEW_PREVIEW_SCALE, fy=DESKEW_PREVIEW_SCALE)
    _, textMask = cv2.threshold(preview, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coarseAngles = np.arange(
        -DESKEW_SEARCH_RANGE_DEG, DESKEW_SEARCH_RANGE_DEG + 1e-6, DESKEW_COARSE_STEP_DEG
    )
    coarseBest = bestProjectionAngle(textMask, coarseAngles)
    fineAngles = np.arange(
        coarseBest - DESKEW_COARSE_STEP_DEG,
        coarseBest + DESKEW_COARSE_STEP_DEG + 1e-6,
        DESKEW_FINE_STEP_DEG,
    )
    return bestProjectionAngle(textMask, fineAngles)


def bestProjectionAngle(textMask, candidateAngles):
    # straight text lines -> peaky row sums -> max variance
    scores = [
        np.var(rotateKeepingPaper(255 - textMask, angle, expand=False).sum(axis=1, dtype=np.float64))
        for angle in candidateAngles
    ]
    return float(candidateAngles[int(np.argmax(scores))])


def adaptiveBinarize(image, context):
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE, ADAPTIVE_OFFSET,
    )


def morphology(image, context):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPHOLOGY_KERNEL)
    textMask = 255 - image
    textMask = cv2.morphologyEx(textMask, cv2.MORPH_OPEN, kernel)
    textMask = cv2.morphologyEx(textMask, cv2.MORPH_CLOSE, kernel)
    return 255 - textMask


STAGE_REGISTRY = {
    "gray": toGray,
    "otsuBinarize": otsuBinarize,
    "denoise": denoise,
    "illumination": illumination,
    "stretch": stretch,
    "clahe": clahe,
    "deskew": deskew,
    "adaptiveBinarize": adaptiveBinarize,
    "morphology": morphology,
}
