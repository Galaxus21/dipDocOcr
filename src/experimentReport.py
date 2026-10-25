from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from enhancementPipeline import runPipeline
from pipelineConfig import ABLATION_CONDITION, ABLATION_STAGES, CONDITION_NAMES, PIPELINE_METHODS

FIGURE_DPI = 130
CROP_ROWS = (0.04, 0.30)
CROP_COLUMNS = (0.12, 0.85)
BEFORE_AFTER_METHODS = ("input", "gray", "enhanced", "enhancedBinary")
METRIC_COLUMNS = ["cer", "wer", "psnr", "ssim", "skewErrorDeg", "processingSeconds"]
FAILURE_CASE_COUNT = 5


def writeReports(rows, samples, outputDir: Path):
    outputDir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(outputDir / "results.csv", index=False)
    summary = table.groupby(["group", "condition", "method"], sort=False)[METRIC_COLUMNS].mean(numeric_only=True)
    summary.to_csv(outputDir / "summary.csv")
    writeMarkdownSummary(table, summary, outputDir)
    plotCerByCondition(summary, outputDir)
    plotBeforeAfter(samples, summary, outputDir)
    plotStageMontage(samples, outputDir)
    plotAblation(summary, outputDir)


def writeMarkdownSummary(table, summary, outputDir):
    methodSummary = summary.loc["method"]
    cerGrid = methodSummary["cer"].unstack("method")[list(PIPELINE_METHODS)].reindex(list(CONDITION_NAMES))
    wers = methodSummary["wer"].unstack("method")[list(PIPELINE_METHODS)].reindex(list(CONDITION_NAMES))
    documents = table["docIndex"].nunique()
    lines = [f"Documents per condition: {documents}", "", "## Mean CER (lower is better)", cerGrid.round(3).to_markdown(),
             "", "## Mean WER (lower is better)", wers.round(3).to_markdown(), "",
             "## Five worst OCR results across all methods",
             worstFailures(table)]
    (outputDir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def worstFailures(table):
    worst = table[table["group"] == "method"].nlargest(FAILURE_CASE_COUNT, "cer")
    return worst[["condition", "method", "docIndex", "cer", "ocrText"]].round(3).to_markdown(index=False)


def plotCerByCondition(summary, outputDir):
    cerGrid = summary.loc["method"]["cer"].unstack("method")[list(PIPELINE_METHODS)].reindex(list(CONDITION_NAMES))
    axes = cerGrid.plot.bar(figsize=(11, 4.8), width=0.82)
    axes.set_ylabel("Mean character error rate (lower is better)")
    axes.set_xlabel("Degradation condition")
    axes.set_title("OCR error by condition and preprocessing method")
    axes.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(outputDir / "cerByCondition.png", dpi=FIGURE_DPI)
    plt.close()


def cropRegion(image):
    height, width = image.shape[:2]
    return image[int(height * CROP_ROWS[0]):int(height * CROP_ROWS[1]),
                 int(width * CROP_COLUMNS[0]):int(width * CROP_COLUMNS[1])]


def plotBeforeAfter(samples, summary, outputDir):
    figure, axes = plt.subplots(len(CONDITION_NAMES), len(BEFORE_AFTER_METHODS), figsize=(15, 13))
    for rowIndex, condition in enumerate(CONDITION_NAMES):
        for columnIndex, method in enumerate(BEFORE_AFTER_METHODS):
            axis = axes[rowIndex][columnIndex]
            axis.imshow(cropRegion(samples[(condition, method)]), cmap="gray", vmin=0, vmax=255)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(titleFor(summary, condition, method), fontsize=8)
        axes[rowIndex][0].set_ylabel(condition)
    plt.tight_layout()
    plt.savefig(outputDir / "beforeAfter.png", dpi=FIGURE_DPI)
    plt.close()


def titleFor(summary, condition, method):
    if method == "input":
        return "degraded input"
    return f"{method}  (mean CER {summary.loc[('method', condition, method)]['cer']:.3f})"


def plotStageMontage(samples, outputDir):
    degraded = samples[(ABLATION_CONDITION, "input")]
    stageImages = {"input": degraded, **runPipeline(degraded, ABLATION_STAGES).intermediates}
    figure, axes = plt.subplots(2, 5, figsize=(17, 7.5))
    for axis, (name, image) in zip(axes.ravel(), stageImages.items()):
        axis.imshow(cropRegion(image), cmap="gray", vmin=0, vmax=255)
        axis.set_title(name)
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
        if not axis.has_data():
            axis.axis("off")
    plt.tight_layout()
    plt.savefig(outputDir / "pipelineStages.png", dpi=FIGURE_DPI)
    plt.close()


def plotAblation(summary, outputDir):
    ablation = summary.loc["ablation"]["cer"].xs(ABLATION_CONDITION, level="condition")
    fullCer = summary.loc[("ablation", ABLATION_CONDITION, "allStages")]["cer"]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    ablation.sort_values().plot.barh(ax=axis)
    axis.axvline(fullCer, color="black", linestyle="--", label="all stages")
    axis.set_xlabel(f"Mean CER on '{ABLATION_CONDITION}' (leave-one-stage-out)")
    axis.set_ylabel("")
    axis.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outputDir / "ablation.png", dpi=FIGURE_DPI)
    plt.close()
