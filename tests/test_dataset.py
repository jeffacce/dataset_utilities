import pytest
from ..dataset import magnitude_and_scale
import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.14f}'.format


def test_magnitude_and_scale_int():
    mag, scale = magnitude_and_scale(pd.Series([1, 2, 3]).astype(int))
    assert mag == 1
    assert scale == 0

def test_magnitude_and_scale_float_type_int():
    mag, scale = magnitude_and_scale(pd.Series([123.0, 1.0, 1234.0, np.nan]))
    assert mag == 4
    assert scale == 0

def test_magnitude_and_scale_zero():
    mag, scale = magnitude_and_scale(pd.Series([0]))
    assert mag == 1
    assert scale == 0

def test_magnitude_and_scale_float():
    mag, scale = magnitude_and_scale(pd.Series([123.1234, 12345.1234567, 12.1234567800]))
    assert mag == 5
    assert scale == 8

def test_magnitude_and_scale_only_frac_part():
    mag, scale = magnitude_and_scale(pd.Series([0.12345, 0.123456, 0.123]))
    assert mag == 1
    assert scale == 6

def test_magnitude_and_scale_empty_raises_error():
    with pytest.raises(ValueError) as e_info:
        mag, scale = magnitude_and_scale(pd.Series([], dtype='float64'))

def test_magnitude_and_scale_nan_raises_error():
    with pytest.raises(ValueError) as e_info:
        mag, scale = magnitude_and_scale(pd.Series([np.nan]))
