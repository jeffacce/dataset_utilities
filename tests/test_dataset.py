import pytest
from ..dataset import magnitude_and_scale
import pandas as pd
import numpy as np


def test_magnitude_and_scale():
    mag, scale = magnitude_and_scale(pd.Series([1,2,3, np.nan]))
    assert mag == 1
    assert scale == 0

    mag, scale = magnitude_and_scale(pd.Series([123.0, 1.0, 1234.0, np.nan]))
    assert mag == 4
    assert scale == 0
    
    mag, scale = magnitude_and_scale(pd.Series([0]))
    assert mag == 1
    assert scale == 0
    
    mag, scale = magnitude_and_scale(pd.Series([123.1234, 12345.1234567, 12.12345678000]))
    assert mag == 5
    assert scale == 8
