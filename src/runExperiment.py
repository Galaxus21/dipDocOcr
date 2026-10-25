import argparse
import random
import time
from pathlib import Path

from enhancementPipeline import runPipeline
from evaluationMetrics import characterErrorRate, imageFidelity, skewErrorDegrees, wordErrorRate
from experimentReport import writeReports
from imageDegradations import degradeDocument
from ocrEngine import RapidOcrEngine
from pipelineConfig import (
    ABLATION_CONDITION, ABLATION_STAGES, CONDITION_NAMES, DEFAULT_DOCUMENT_COUNT,
    DEFAULT_RANDOM_SEED, PIPELINE_METHODS,
)
from syntheticInvoiceGenerator import generateInvoice

SEED_STRIDE = 1000
SAMPLE_DOCUMENT_INDEX = 0
METHOD_GROUP = "method"
ABLATION_GROUP = "ablation"
ABLATION_FULL_NAME = "allStages"
OCR_TEXT_SEPARATOR = " | "


def main():
    parser = argparse.ArgumentParser(description="OCR accuracy with and without DIP enhancement")
    parser.add_argument("--docs", type=int, default=DEFAULT_DOCUMENT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--outputDir", default="outputs")
    args = parser.parse_args()
    runExperiment(args.docs, args.seed, Path(args.outputDir))


def runExperiment(documentCount, seed, outputDir):
    ocrEngine = RapidOcrEngine()
    rows, samples = [], {}
    for docIndex in range(documentCount):
        rng = random.Random(seed * SEED_STRIDE + docIndex)
        invoice = generateInvoice(rng)
        if docIndex == SAMPLE_DOCUMENT_INDEX:
            samples["clean"] = invoice.image
        for condition in CONDITION_NAMES:
            degraded = degradeDocument(condition, invoice.image, rng)
            for methodName, stages in configurationsFor(condition):
                group = ABLATION_GROUP if methodName not in PIPELINE_METHODS else METHOD_GROUP
                row, result = evaluateConfiguration(
                    group, docIndex, condition, methodName, stages, invoice, degraded, ocrEngine
                )
                rows.append(row)
                if docIndex == SAMPLE_DOCUMENT_INDEX and group == METHOD_GROUP:
                    samples[(condition, methodName)] = result.image
                    samples[(condition, "input")] = degraded.image
        print(f"document {docIndex + 1}/{documentCount} done", flush=True)
    writeReports(rows, samples, outputDir)


def configurationsFor(condition):
    configurations = list(PIPELINE_METHODS.items())
    if condition != ABLATION_CONDITION:
        return configurations
    configurations.append((ABLATION_FULL_NAME, ABLATION_STAGES))
    for removed in ABLATION_STAGES:
        if removed != "gray":
            remaining = [stage for stage in ABLATION_STAGES if stage != removed]
            configurations.append((f"allMinus-{removed}", remaining))
    return configurations


def evaluateConfiguration(group, docIndex, condition, methodName, stages, invoice, degraded, ocrEngine):
    startTime = time.perf_counter()
    result = runPipeline(degraded.image, stages)
    processingSeconds = time.perf_counter() - startTime
    text = ocrEngine.readText(result.image)
    psnr, ssim = imageFidelity(result.image, invoice.image)
    row = {
        "group": group, "docIndex": docIndex, "condition": condition, "method": methodName,
        "cer": characterErrorRate(invoice.groundTruthText, text),
        "wer": wordErrorRate(invoice.groundTruthText, text),
        "psnr": psnr, "ssim": ssim,
        "skewErrorDeg": skewErrorDegrees(degraded.appliedSkewDeg, result.estimatedSkewDeg)
        if "deskew" in stages else float("nan"),
        "processingSeconds": processingSeconds,
        "ocrText": text.replace("\n", OCR_TEXT_SEPARATOR),
    }
    return row, result


if __name__ == "__main__":
    main()
