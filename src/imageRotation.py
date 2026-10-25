import cv2
import numpy as np


def rotateKeepingPaper(image, angleDeg, expand):
    """Rotate counter-clockwise; border pixels are replicated."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angleDeg, 1.0)
    outWidth, outHeight = width, height
    if expand:
        cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
        outWidth = int(height * sine + width * cosine)
        outHeight = int(height * cosine + width * sine)
        matrix[0, 2] += outWidth / 2 - width / 2
        matrix[1, 2] += outHeight / 2 - height / 2
    return cv2.warpAffine(
        image, matrix, (outWidth, outHeight),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def asUint8(image):
    return np.clip(image, 0, 255).astype(np.uint8)
