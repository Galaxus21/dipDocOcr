from dataclasses import dataclass, field

import numpy as np

from enhancementStages import ESTIMATED_SKEW_KEY, STAGE_REGISTRY


@dataclass
class PipelineResult:
    image: np.ndarray
    intermediates: dict = field(default_factory=dict)
    estimatedSkewDeg: float = 0.0


def runPipeline(image, stageNames) -> PipelineResult:
    unknown = [name for name in stageNames if name not in STAGE_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}")
    context = {}
    intermediates = {}
    current = image
    for name in stageNames:
        current = STAGE_REGISTRY[name](current, context)
        intermediates[name] = current
    return PipelineResult(current, intermediates, context.get(ESTIMATED_SKEW_KEY, 0.0))
