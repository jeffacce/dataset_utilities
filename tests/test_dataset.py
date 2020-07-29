import pytest
from ..dataset import magnitude_and_scale, get_type
import pandas as pd
import numpy as np

# dataset.magnitude_and_scale
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


def test_magnitude_and_scale_inf_raises_error():
    with pytest.raises(ValueError) as e_info:
        mag, scale = magnitude_and_scale(pd.Series([np.inf]))


# dataset.get_type
def test_get_type_decimal():
    result, comment = get_type(pd.Series([1.1, 2.1, 3.0])) 
    assert result == 'decimal(2, 1)'
    assert comment == ''

    result, comment = get_type(pd.Series([123.1234, 12345.1234567, 12.1234567800]))
    assert result == 'decimal(19, 12)'
    assert comment == ''

    result, comment = get_type(pd.Series([0.12345, 0.123456, 0.123]))
    assert result == 'decimal(10, 9)'

    result, comment = get_type(pd.Series([1.1, 2.1, 3.0, np.nan]))
    assert result == 'decimal(2, 1) NULL'


def test_get_type_str():
    result, comment = get_type(pd.Series(['123']))
    assert result == 'nvarchar(6)'
    assert comment == ''

    result, comment = get_type(pd.Series(['a' * 2001]))
    assert result == 'nvarchar(4000)'
    assert comment == ''

    result, comment = get_type(pd.Series(['a', 'b', 'c', 'def', np.nan]))
    assert result == 'nvarchar(6) NULL'
    assert comment == ''


def test_get_type_int():
    result, comment = get_type(pd.Series([0, 1, 0.0, 1.00]))
    assert result == 'bit'
    assert comment == ''

    result, comment = get_type(pd.Series([0, 1, 2, 3, 3.0, 4.0]))
    assert result == 'tinyint'
    assert comment == ''

    result, comment = get_type(pd.Series([-2.0, -1, 0.000, 1, 2.0]))
    assert result == 'smallint'
    assert comment == ''

    result, comment = get_type(pd.Series([-60000, 0.000, 60000]))
    assert result == 'int'
    assert comment == ''

    result, comment = get_type(pd.Series([-2147490000, 0.000, 2147490000]))
    assert result == 'bigint'
    assert comment == ''


def test_get_type_mixed():
    result, comment = get_type(pd.Series([1, 2.0, 3.1, 'abc', pd.Timestamp('2020-01-01 00:00:00'), np.nan]))
    assert result == 'nvarchar(38) NULL'


def test_get_type_datetime():
    result, comment = get_type(pd.to_datetime(['2020-01-01', '2020-01-02']))
    assert result == 'datetime'
    assert comment == ''


def test_get_type_date():
    result, comment = get_type(pd.to_datetime(['2020-01-01', '2020-01-02']).date)
    assert result == 'date'
    assert comment == ''


def test_get_type_empty():
    result, comment = get_type(pd.Series([], dtype=object))
    assert result == 'nvarchar(255) NULL'
    assert comment == 'empty column, defaulting to nvarchar(255)'

    result, comment = get_type(pd.Series([np.nan]))
    assert result == 'nvarchar(255) NULL'
    assert comment == 'empty column, defaulting to nvarchar(255)'
