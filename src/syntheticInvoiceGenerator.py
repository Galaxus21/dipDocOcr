import os
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipelineConfig import (
    BODY_FONT_SIZE_PX, FONT_DIRECTORIES, FONT_FILE_CANDIDATES, PAGE_HEIGHT_PX,
    PAGE_MARGIN_PX, PAGE_WIDTH_PX, ROW_HEIGHT_PX, TITLE_FONT_SIZE_PX, WHITE_LEVEL,
)

INSURERS = ("Shield Assure", "Nova Mutual", "Bharat Secure", "Lotus General", "Apex Indemnity")
CLAIM_TYPES = ("Motor Own Damage", "Health Cashless", "Travel Delay", "Home Fire Cover")
CLIENT_NAMES = ("Rohan Mehta", "Ananya Iyer", "Kabir Singh", "Meera Nair", "Vikram Desai")
LINE_ITEMS = (
    "Front bumper repair", "Paint and polish", "Hospital room charges", "Diagnostic scan",
    "Windshield replacement", "Pharmacy bill", "Labour charges", "Towing service",
)
TABLE_COLUMN_X = (PAGE_MARGIN_PX, 700, 880, 1010)
TABLE_HEADER = ("Description", "Qty", "Rate", "Amount")
TABLE_ROW_COUNT = 6
FOOTER_LINES = (
    "Please quote the claim number in all correspondence",
    "This is a computer generated document",
)


@dataclass
class InvoiceDocument:
    image: np.ndarray
    groundTruthText: str


def generateInvoice(rng: random.Random) -> InvoiceDocument:
    rows = buildRows(rng)
    fontPath = pickFontPath(rng)
    bodyFont = ImageFont.truetype(fontPath, BODY_FONT_SIZE_PX)
    titleFont = ImageFont.truetype(fontPath, TITLE_FONT_SIZE_PX)
    page = Image.new("L", (PAGE_WIDTH_PX, PAGE_HEIGHT_PX), WHITE_LEVEL)
    draw = ImageDraw.Draw(page)
    positionY = PAGE_MARGIN_PX
    for cells in rows:
        for positionX, text, isTitle in cells:
            draw.text((positionX, positionY), text, fill=0, font=titleFont if isTitle else bodyFont)
        positionY += ROW_HEIGHT_PX * (2 if cells[0][2] else 1)
    groundTruth = "\n".join(" ".join(text for _, text, _ in cells) for cells in rows)
    return InvoiceDocument(np.array(page), groundTruth)


def buildRows(rng):
    insurer = rng.choice(INSURERS)
    rows = [[(PAGE_MARGIN_PX, f"{insurer} Insurance", True)]]
    rows += [[(PAGE_MARGIN_PX, line, False)] for line in buildMetaLines(rng)]
    rows.append([(x, label, False) for x, label in zip(TABLE_COLUMN_X, TABLE_HEADER)])
    total = 0.0
    for _ in range(TABLE_ROW_COUNT):
        row, amount = buildTableRow(rng)
        rows.append(row)
        total += amount
    rows.append([(TABLE_COLUMN_X[2], "Total", False), (TABLE_COLUMN_X[3], f"{total:,.2f}", False)])
    rows += [[(PAGE_MARGIN_PX, line, False)] for line in FOOTER_LINES]
    return rows


def buildMetaLines(rng):
    return [
        f"Claim No: CLM{rng.randint(100000, 999999)}",
        f"Policy No: POL{rng.randint(10000000, 99999999)}",
        f"Insured: {rng.choice(CLIENT_NAMES)}",
        f"Claim Type: {rng.choice(CLAIM_TYPES)}",
        f"Date: {rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2023, 2026)}",
    ]


def buildTableRow(rng):
    quantity = rng.randint(1, 4)
    rate = rng.randint(250, 9500)
    amount = float(quantity * rate)
    cells = [
        (TABLE_COLUMN_X[0], rng.choice(LINE_ITEMS), False),
        (TABLE_COLUMN_X[1], str(quantity), False),
        (TABLE_COLUMN_X[2], f"{rate:,}", False),
        (TABLE_COLUMN_X[3], f"{amount:,.2f}", False),
    ]
    return cells, amount


def pickFontPath(rng):
    available = [
        os.path.join(directory, name)
        for directory in FONT_DIRECTORIES for name in FONT_FILE_CANDIDATES
        if os.path.exists(os.path.join(directory, name))
    ]
    if not available:
        raise FileNotFoundError("No usable TrueType font found; extend FONT_DIRECTORIES")
    return rng.choice(available)
