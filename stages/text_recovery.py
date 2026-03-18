"""Stage 5.5: Text Recovery — backfill Jmail text, extract PDF text layers, OCR."""

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def classify_page_pixels(pixels: np.ndarray) -> str:
    """Classify a grayscale page image by pixel distribution."""
    white_pct = (pixels > 230).mean()
    black_pct = (pixels < 25).mean()
    mid_pct = 1.0 - white_pct - black_pct

    if black_pct > 0.90:
        return "redacted"
    if mid_pct > 0.40:
        return "photo"
    if white_pct > 0.98:
        return "blank"
    return "text"
