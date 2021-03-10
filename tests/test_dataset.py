import pytest
from ..dataset import (
    magnitude_and_scale, get_type,
    _try_import, indent, get_df_type,
    cast_and_clean_df
)
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


def test_magnitude_and_scale_float_with_inf():
    mag, scale = magnitude_and_scale(pd.Series([1.0, 2.0, np.inf, -np.inf]))
    assert mag == 1
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
    dtype, params, has_null, comment = get_type(pd.Series([1.1, 2.1, 3.0])) 
    assert dtype == 'decimal'
    assert params == [2, 1]
    assert has_null == False
    assert comment == ''

    dtype, params, has_null, comment = get_type(pd.Series([123.1234, 12345.1234567, 12.1234567800]))
    assert dtype == 'decimal'
    assert params == [19, 12]
    assert has_null == False
    assert comment == ''

    dtype, params, has_null, comment = get_type(pd.Series([0.12345, 0.123456, 0.123]))
    assert dtype == 'decimal'
    assert params == [10, 9]
    assert has_null == False
    assert comment == ''

    dtype, params, has_null, comment = get_type(pd.Series([1.1, 2.1, 3.0, np.nan]))
    assert dtype == 'decimal'
    assert params == [2, 1]
    assert has_null == True
    assert comment == ''

    dtype, params, has_null, comment = get_type(pd.Series([1.1, 2.1, 3.0, np.nan, np.inf]))
    assert dtype == 'decimal'
    assert params == [2, 1]
    assert has_null == True
    assert comment == ''


def test_get_type_str():
    dtype, params, has_null, comment = get_type(pd.Series(['123']))
    assert dtype == 'nvarchar'
    assert params == [6]
    assert has_null == False
    assert comment == ''

    dtype, params, has_null, comment = get_type(pd.Series(['a' * 2001]))
    assert dtype == 'nvarchar'
    assert params == [4000]
    assert has_null == False
    assert comment == ''

    dtype, params, has_null, comment = get_type(pd.Series(['a', 'b', 'c', 'def', np.nan]))
    assert dtype == 'nvarchar'
    assert params == [6]
    assert has_null == True
    assert comment == ''


def test_get_type_bool():
    dtype, params, has_null, comment = get_type(pd.Series([True, False]))
    assert dtype == 'bit'
    assert params == []
    assert has_null == False
    assert comment == ''

    dtype, params, has_null, comment = get_type(pd.Series([True, False, np.nan]))
    assert dtype == 'bit'
    assert params == []
    assert has_null == True
    assert comment == ''


def test_get_type_int():
    dtype, params, has_null, comment = get_type(pd.Series([0, 1, 0.0, 1.00]))
    assert dtype == 'bit'
    assert params == []
    assert has_null == False
    assert comment == ''

    dtype, params, has_null, comment = get_type(pd.Series([0, 1, 2, 3, 3.0, 4.0]))
    assert dtype == 'tinyint'
    assert params == []
    assert has_null == False
    assert comment == ''

    dtype, params, has_null, comment = get_type(pd.Series([-2.0, -1, 0.000, 1, 2.0]))
    assert dtype == 'smallint'
    assert params == []
    assert has_null == False
    assert comment == ''

    dtype, params, has_null, comment = get_type(pd.Series([-60000, 0.000, 60000]))
    assert dtype == 'int'
    assert params == []
    assert has_null == False
    assert comment == ''

    dtype, params, has_null, comment = get_type(pd.Series([-2147490000, 0.000, 2147490000]))
    assert dtype == 'bigint'
    assert params == []
    assert has_null == False
    assert comment == ''


def test_get_type_mixed():
    dtype, params, has_null, comment = get_type(pd.Series([1, 2.0, 3.1, 'abc', pd.Timestamp('2020-01-01 00:00:00'), np.nan]))
    assert dtype == 'nvarchar'
    assert params == [38]
    assert has_null == True
    assert comment == ''


def test_get_type_datetime():
    dtype, params, has_null, comment = get_type(pd.to_datetime(['2020-01-01', '2020-01-02']))
    assert dtype == 'datetime'
    assert params == []
    assert has_null == False
    assert comment == ''


def test_get_type_date():
    dtype, params, has_null, comment = get_type(pd.to_datetime(['2020-01-01', '2020-01-02']).date)
    assert dtype == 'date'
    assert params == []
    assert has_null == False
    assert comment == ''


def test_get_type_empty():
    dtype, params, has_null, comment = get_type(pd.Series([], dtype=object))
    assert dtype == 'nvarchar'
    assert params == [255]
    assert has_null == True
    assert comment == 'empty column, defaulting to nvarchar(255)'

    dtype, params, has_null, comment = get_type(pd.Series([np.nan]))
    assert dtype == 'nvarchar'
    assert params == [255]
    assert has_null == True
    assert comment == 'empty column, defaulting to nvarchar(255)'


def test_try_import():
    package = _try_import('numpy')
    assert package is np

    module = _try_import('numpy.abs')
    assert module is np.abs

    from pandas.tseries import offsets
    module = _try_import('pandas.tseries.offsets')
    assert module is offsets

    from pandas.tseries.offsets import DateOffset
    method = _try_import('pandas.tseries.offsets.DateOffset')
    assert method is DateOffset


def test_indent():
    assert indent(['blah']) == ['    blah']


df = pd.DataFrame({
    'intcol': pd.Series([1,2,3]),
    'strcol': pd.Series(['a', 'b', 'c']),
    'floatcol': pd.Series([np.inf, 1.100, 2.100]),
    'boolcol': pd.Series([np.nan, False, True]),
    'boolcol2': pd.Series([False, True, True]),
})

df_type = [
    ['intcol', 'tinyint', [], False, ''],
    ['strcol', 'nvarchar', [2], False, ''],
    ['floatcol', 'decimal', [2, 1], True, ''],
    ['boolcol', 'bit', [], True, ''],
    ['boolcol2', 'bit', [], False, ''],
]

def test_get_df_type():
    assert get_df_type(df) == df_type


def test_cast_and_clean_df():
    with pytest.warns(UserWarning, match='infinity'):
        df_clean = cast_and_clean_df(df, df_type)
    assert np.inf not in df_clean
    assert -np.inf not in df_clean
    assert False not in df_clean
    assert True not in df_clean
    boolcol_clean = pd.Series([pd.NA, 0, 1]).astype('Int64').rename('boolcol')
    boolcol2_clean = pd.Series([0,1,1]).astype('Int64').rename('boolcol2')
    pd.testing.assert_series_equal(df_clean['boolcol'], boolcol_clean)
    pd.testing.assert_series_equal(df_clean['boolcol2'], boolcol2_clean)

