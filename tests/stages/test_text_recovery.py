import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from stages.text_recovery import classify_page_pixels


def test_classify_all_black_as_redacted():
    pixels = np.zeros((100, 100), dtype=np.uint8)
    assert classify_page_pixels(pixels) == "redacted"

def test_classify_all_white_as_blank():
    pixels = np.full((100, 100), 240, dtype=np.uint8)
    assert classify_page_pixels(pixels) == "blank"

def test_classify_photo_as_photo():
    pixels = np.linspace(50, 200, 10000, dtype=np.uint8).reshape(100, 100)
    assert classify_page_pixels(pixels) == "photo"

def test_classify_text_page_as_text():
    pixels = np.full((100, 100), 240, dtype=np.uint8)
    pixels[10:12, 10:60] = 5
    pixels[20:22, 10:70] = 5
    pixels[30:32, 10:50] = 5
    assert classify_page_pixels(pixels) == "text"

def test_classify_mostly_redacted_with_some_white():
    pixels = np.zeros((100, 100), dtype=np.uint8)
    pixels[0:5, :] = 240
    assert classify_page_pixels(pixels) == "redacted"
