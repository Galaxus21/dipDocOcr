import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from pipelineConfig import LINE_GROUP_HEIGHT_FRACTION


class RapidOcrEngine:
    """Wrapper so the OCR backend can be swapped."""

    def __init__(self):
        self.engine = RapidOCR()

    def readText(self, image) -> str:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
        detections, _ = self.engine(bgr)
        if not detections:
            return ""
        return "\n".join(groupIntoLines(detections))


def groupIntoLines(detections):
    boxes = [
        (np.mean([point[1] for point in box]), np.mean([point[0] for point in box]),
         max(point[1] for point in box) - min(point[1] for point in box), text)
        for box, text, _ in detections
    ]
    boxes.sort()
    medianHeight = float(np.median([height for _, _, height, _ in boxes]))
    lines, current, anchorY = [], [], None
    for centerY, centerX, _, text in boxes:
        if anchorY is not None and centerY - anchorY > medianHeight * LINE_GROUP_HEIGHT_FRACTION:
            lines.append(current)
            current = []
        if not current:
            anchorY = centerY
        current.append((centerX, text))
    lines.append(current)
    return [" ".join(text for _, text in sorted(line)) for line in lines]
