import pytest
from ..dataset import (
    magnitude_and_scale,
    get_type,
    _try_import,
    indent,
    get_df_type,
    cast_and_clean_df,
    sql_dataset,
)
import pandas as pd
import numpy as np
import datetime

CMD_DROP_TEST_TABLE_IF_EXISTS = "IF OBJECT_ID('test_table', 'U') IS NOT NULL DROP TABLE test_table;"
CMD_CREATE_TEST_TABLE = """
    CREATE TABLE test_table (
        [dt] datetime NULL,
        [dt2] date NOT NULL,
        [uid] nvarchar(10) NOT NULL,
        [strcol] nvarchar(max) NOT NULL,
        [name] nvarchar(10) NULL,
        [empty_col] nvarchar(100) NULL,
        [float] decimal(22,3) NULL,
        [float_k] decimal(22,3) NULL,
        [float_m] decimal(22,13) NULL,
        [float_b] decimal(22,9) NULL,
        [float_na] decimal(22,3) NULL,
        [bit] bit NULL,
        [bit_na] bit NULL,
        [tinyint] tinyint NULL,
        [tinyint_na] tinyint NULL,
        [smallint] smallint NOT NULL,
        [smallint_na] smallint NULL,
        [int] int NOT NULL,
        [int_na] int NULL,
        [bigint] bigint NULL,
        [bigint_na] bigint NULL,
        [bool] bit NULL,
        [bool_na] bit NULL,
        [empty_str_col] nvarchar(100) NULL
    );
"""
expected_schema = [
    ['dt', 'datetime', [], True, ''],
    ['dt2', 'date', [], False, ''],
    ['uid', 'nvarchar', [10], False, ''],
    ['strcol', 'nvarchar', ['max'], False, ''],
    ['name', 'nvarchar', [10], True, ''],
    ['empty_col', 'nvarchar', [100], True, ''],
    ['float', 'decimal', [22,3], True, ''],
    ['float_k', 'decimal', [22,3], True, ''],
    ['float_m', 'decimal', [22,13], True, ''],
    ['float_b', 'decimal', [22,9], True, ''],
    ['float_na', 'decimal', [22,3], True, ''],
    ['bit', 'bit', [], True, ''],
    ['bit_na', 'bit', [], True, ''],
    ['tinyint', 'tinyint', [], True, ''],
    ['tinyint_na', 'tinyint', [], True, ''],
    ['smallint', 'smallint', [], False, ''],
    ['smallint_na', 'smallint', [], True, ''],
    ['int', 'int', [], False, ''],
    ['int_na', 'int', [], True, ''],
    ['bigint', 'bigint', [], True, ''],
    ['bigint_na', 'bigint', [], True, ''],
    ['bool', 'bit', [], True, ''],
    ['bool_na', 'bit', [], True, ''],
    ['empty_str_col', 'nvarchar', [100], True, ''],
]


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

    dtype, params, has_null, comment = get_type(pd.Series(['a' * 1000]))
    assert dtype == 'nvarchar'
    assert params == [2000]
    assert has_null == False
    assert comment == ''

    dtype, params, has_null, comment = get_type(pd.Series(['a' * 2001]))
    assert dtype == 'nvarchar'
    assert params == [4000]
    assert has_null == False
    assert comment == ''

    with pytest.warns(None):
        dtype, params, has_null, comment = get_type(pd.Series(['a' * 4001]))
    assert dtype == 'nvarchar'
    assert params == ['max']
    assert has_null == False
    assert comment == 'Maximum string length is 4001. Using nvarchar(max).'

    dtype, params, has_null, comment = get_type(pd.Series(['a', 'b', 'c', 'def', np.nan]))
    assert dtype == 'nvarchar'
    assert params == [6]
    assert has_null == True
    assert comment == ''

    dtype, params, has_null, comment = get_type(pd.Series(['', '', '', '', '']))
    assert dtype == 'nvarchar'
    assert params == [255]
    assert has_null == False
    assert comment == 'zero-length string column, defaulting to nvarchar(255)'


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
    # if the entire column is 0, default to int
    dtype, params, has_null, comment = get_type(pd.Series([0, 0, 0]))
    assert dtype == 'int'
    assert params == []
    assert has_null == False
    assert comment == 'column contains only zeros; defaulting to int'

    # if the entire column is 0 and null, default to int
    dtype, params, has_null, comment = get_type(pd.Series([0, 0, 0, np.nan]))
    assert dtype == 'int'
    assert params == []
    assert has_null == True
    assert comment == 'column contains only zeros; defaulting to int'

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
    # test different orders of the same mixed values; these should all return nvarchar.
    # this is to guard against naively detecting types by the first non-empty value in object dtype columns
    dtype, params, has_null, comment = get_type(pd.Series([1, 2.0, 3.1, 'abc', pd.Timestamp('2020-01-01 00:00:00'), np.nan]))
    assert dtype == 'nvarchar'
    assert params == [38]
    assert has_null == True
    assert comment == ''

    dtype, params, has_null, comment = get_type(pd.Series([pd.Timestamp('2020-01-01 00:00:00'), 1, 2.0, 3.1, 'abc', np.nan]))
    assert dtype == 'nvarchar'
    assert params == [38]
    assert has_null == True
    assert comment == ''

    dtype, params, has_null, comment = get_type(pd.Series([datetime.date(2020, 1, 1), 1, 2.0, 3.1, 'abc', np.nan]))
    assert dtype == 'nvarchar'
    assert params == [20]
    assert has_null == True
    assert comment == ''

    dtype, params, has_null, comment = get_type(pd.Series([datetime.datetime(2020, 1, 1, 0, 0, 0), 1, 2.0, 3.1, 'abc', np.nan]))
    assert dtype == 'nvarchar'
    assert params == [38]
    assert has_null == True
    assert comment == ''

    dtype, params, has_null, comment = get_type(pd.Series([1, 2.0, 3.1, pd.Timestamp('2020-01-01 00:00:00'), 'abc', np.nan]))
    assert dtype == 'nvarchar'
    assert params == [38]
    assert has_null == True
    assert comment == ''

    dtype, params, has_null, comment = get_type(pd.Series([2.0, 1, 3.1, pd.Timestamp('2020-01-01 00:00:00'), 'abc', np.nan]))
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
    'intcol2': pd.Series([1,2,np.nan]),
    'strcol': pd.Series(['a', 'b', 'c']),
    'strcol2': pd.Series(['a'*10, 'b'*10, 'c'*10]),
    'strcol3': pd.Series(['a'*4001, 'b'*4001, 'c'*4001]),
    'floatcol': pd.Series([np.inf, 1.100, 2.100]),
    'floatcol2': pd.Series([1.12345, 2.12345, 3.12345]),
    'boolcol': pd.Series([np.nan, False, True]),
    'boolcol2': pd.Series([False, True, True]),
})

df_type = [
    ['intcol', 'tinyint', [], False, ''],
    ['intcol2', 'tinyint', [], True, ''],
    ['strcol', 'nvarchar', [2], False, ''],
    ['strcol2', 'nvarchar', [20], False, ''],
    ['strcol3', 'nvarchar', ['max'], False, 'Maximum string length is 4001. Using nvarchar(max).'],
    ['floatcol', 'decimal', [2, 1], True, ''],
    ['floatcol2', 'decimal', [8, 7], False, ''],
    ['boolcol', 'bit', [], True, ''],
    ['boolcol2', 'bit', [], False, ''],
]

df_type_truncate = [
    ['intcol', 'tinyint', [], False, ''],
    ['intcol2', 'tinyint', [], True, ''],
    ['strcol', 'nvarchar', [2], False, ''],
    ['strcol2', 'nvarchar', [5], False, ''],
    ['strcol3', 'nvarchar', [5], False, ''],
    ['floatcol', 'decimal', [2, 0], True, ''],
    ['floatcol2', 'decimal', [8, 3], False, ''],
    ['boolcol', 'bit', [], True, ''],
    ['boolcol2', 'bit', [], False, ''],
]

boolcol_clean = pd.Series([pd.NA, 0, 1]).astype('Int64')
boolcol2_clean = pd.Series([0, 1, 1]).astype('Int64')
intcol_clean = pd.Series([1, 2, 3]).astype('Int64')
intcol2_clean = pd.Series([1, 2, pd.NA]).astype('Int64')

strcol2_trunc = pd.Series(['a'*5, 'b'*5, 'c'*5])
strcol3_trunc = pd.Series(['a'*5, 'b'*5, 'c'*5])
floatcol_trunc = pd.Series([np.nan, 1.0, 2.0])
floatcol2_trunc = pd.Series([1.123, 2.123, 3.123])

def test_get_df_type():
    with pytest.warns(UserWarning, match=r'nvarchar\(max\)'):
        assert get_df_type(df) == df_type

def test_cast_and_clean_df():
    with pytest.warns(UserWarning, match='infinity'):
        df_clean = cast_and_clean_df(df, df_type, truncate=False, verbose=True)
    assert np.inf not in df_clean
    assert -np.inf not in df_clean
    assert False not in df_clean
    assert True not in df_clean

    pd.testing.assert_series_equal(df_clean['boolcol'], boolcol_clean, check_names=False)
    pd.testing.assert_series_equal(df_clean['boolcol2'], boolcol2_clean, check_names=False)
    pd.testing.assert_series_equal(df_clean['intcol'], intcol_clean, check_names=False)
    pd.testing.assert_series_equal(df_clean['intcol2'], intcol2_clean, check_names=False)


def test_cast_and_clean_df_truncate():
    with pytest.raises(ValueError, match='truncation'):
        df_clean = cast_and_clean_df(df, df_type_truncate, truncate=False, verbose=True)
    with pytest.warns(UserWarning, match='infinity'):
        df_clean = cast_and_clean_df(df, df_type_truncate, truncate=True, verbose=False)

    assert np.inf not in df_clean
    assert -np.inf not in df_clean
    assert False not in df_clean
    assert True not in df_clean

    pd.testing.assert_series_equal(df_clean['boolcol'], boolcol_clean, check_names=False)
    pd.testing.assert_series_equal(df_clean['boolcol2'], boolcol2_clean, check_names=False)
    pd.testing.assert_series_equal(df_clean['intcol'], intcol_clean, check_names=False)
    pd.testing.assert_series_equal(df_clean['intcol2'], intcol2_clean, check_names=False)
    pd.testing.assert_series_equal(df_clean['strcol2'], strcol2_trunc, check_names=False)
    pd.testing.assert_series_equal(df_clean['strcol3'], strcol3_trunc, check_names=False)
    pd.testing.assert_series_equal(df_clean['floatcol'], floatcol_trunc, check_names=False)
    pd.testing.assert_series_equal(df_clean['floatcol2'], floatcol2_trunc, check_names=False)
    

def test__table_exists():
    db = sql_dataset('./tests/database.yml')
    db.send_cmd("IF OBJECT_ID('test_table', 'U') IS NOT NULL DROP TABLE test_table;")
    db.send_cmd("CREATE TABLE test_table ([uid] nvarchar(100) NULL);")
    assert db._table_exists(db.config['conn'], 'test_table')

    db.send_cmd("DROP TABLE test_table")
    assert (not db._table_exists(db.config['conn'], 'test_table'))


def test__get_table_schema():
    db = sql_dataset('./tests/database.yml')
    db.send_cmd(CMD_DROP_TEST_TABLE_IF_EXISTS)
    db.send_cmd(CMD_CREATE_TEST_TABLE)

    result = db._get_table_schema(db.config['conn'], 'test_table')
    for i in range(len(result)):
        for j in range(len(result[i])):
            assert result[i][j] == expected_schema[i][j]
    
    db.send_cmd(CMD_DROP_TEST_TABLE_IF_EXISTS)
