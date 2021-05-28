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
import pyodbc
import requests


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

def test_get_type_decimal_na_inf():
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

def test_get_type_str_max():
    with pytest.warns(None):
        dtype, params, has_null, comment = get_type(pd.Series(['a' * 4001]))
    assert dtype == 'nvarchar'
    assert params == ['max']
    assert has_null == False
    assert comment == 'Maximum string length is 4001. Using nvarchar(max).'

def test_get_type_str_na():
    dtype, params, has_null, comment = get_type(pd.Series(['a', 'b', 'c', 'def', np.nan]))
    assert dtype == 'nvarchar'
    assert params == [6]
    assert has_null == True
    assert comment == ''

def test_get_type_str_empty():
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

def test_get_type_bool_na():
    dtype, params, has_null, comment = get_type(pd.Series([True, False, np.nan]))
    assert dtype == 'bit'
    assert params == []
    assert has_null == True
    assert comment == ''

def test_get_type_bit():
    dtype, params, has_null, comment = get_type(pd.Series([0, 1, 0.0, 1.00]))
    assert dtype == 'bit'
    assert params == []
    assert has_null == False
    assert comment == ''

def test_get_type_tinyint():
    dtype, params, has_null, comment = get_type(pd.Series([0, 1, 2, 3, 3.0, 4.0]))
    assert dtype == 'tinyint'
    assert params == []
    assert has_null == False
    assert comment == ''

def test_get_type_smallint():
    dtype, params, has_null, comment = get_type(pd.Series([-2.0, -1, 0.000, 1, 2.0]))
    assert dtype == 'smallint'
    assert params == []
    assert has_null == False
    assert comment == ''

def test_get_type_int():
    dtype, params, has_null, comment = get_type(pd.Series([-60000, 0.000, 60000]))
    assert dtype == 'int'
    assert params == []
    assert has_null == False
    assert comment == ''

def test_get_type_int_zeros():
    # if the entire column is 0, default to int
    dtype, params, has_null, comment = get_type(pd.Series([0, 0, 0]))
    assert dtype == 'int'
    assert params == []
    assert has_null == False
    assert comment == 'column contains only zeros; defaulting to int'

def test_get_type_int_zeros_na():
    # if the entire column is 0 and null, default to int
    dtype, params, has_null, comment = get_type(pd.Series([0, 0, 0, np.nan]))
    assert dtype == 'int'
    assert params == []
    assert has_null == True
    assert comment == 'column contains only zeros; defaulting to int'

def test_get_type_bigint():
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

def test_get_type_empty_only_na():
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
    db = sql_dataset('./tests/config/database.yml')
    db.send_cmd("IF OBJECT_ID('test_table', 'U') IS NOT NULL DROP TABLE test_table;")
    db.send_cmd("CREATE TABLE test_table ([uid] nvarchar(100) NULL);")
    conn = pyodbc.connect(**db.config['conn'])
    assert db._table_exists(conn, 'test_table')

    db.send_cmd("DROP TABLE test_table")
    assert (not db._table_exists(conn, 'test_table'))


def test__get_table_schema():
    db = sql_dataset('./tests/config/database.yml')
    db.send_cmd(CMD_DROP_TEST_TABLE_IF_EXISTS)
    db.send_cmd(CMD_CREATE_TEST_TABLE)

    conn = pyodbc.connect(**db.config['conn'])
    result = db._get_table_schema(conn, 'test_table')
    for i in range(len(result)):
        for j in range(len(result[i])):
            assert result[i][j] == expected_schema[i][j]
    
    db.send_cmd(CMD_DROP_TEST_TABLE_IF_EXISTS)


def test__connect():
    db = sql_dataset('./tests/config/database.yml')
    conn = db._connect(db.config['conn'])


def test__connect_fake_database_raises_connection_error(verbose=True):
    sd = sql_dataset('./tests/config/fake_database.yml')
    with pytest.raises(requests.ConnectionError):
        sd._connect(sd.config['conn'], max_retries=1, delay=5, verbose=False)


READ_OVERRIDE_PARAMS = {
    'filepath': 'p.xls',
    'kwargs': {
        'header': 'p_header',
        'sheet_name': 'p_sheet_name',
        'encoding': 'p_encoding',
        'kwarg0': 'p_kwarg0',
        'kwarg1': 'p_kwarg1',
    },
}

def test__get_config_read_default():
    db = sql_dataset('./tests/config/empty.yml')
    config = db._get_config(
        'read',
        filepath=None,
        kwargs={},
    )
    assert config == (
        None,
        {},
    )

def test__get_config_read_global():
    db = sql_dataset('./tests/config/global.yml')
    config = db._get_config(
        'read',
        filepath=None,
        kwargs={},
    )
    assert config == (
        'g.xls',
        {
            'header': 'g_header',
            'sheet_name': 'g_sheet_name',
            'encoding': 'g_encoding',
        },
    )


def test__get_config_read_local():
    db = sql_dataset('./tests/config/read/local.yml')
    config = db._get_config(
        'read',
        filepath=None,
        kwargs={},
    )
    assert config == (
        'r.xls',
        {
            'header': 'r_header',
            'sheet_name': 'r_sheet_name',
            'encoding': 'r_encoding',
            'kwarg0': 'r_kwarg0',
            'kwarg1': 'r_kwarg1',
        }, 
    )


def test__get_config_read_local_overrides_global():
    db = sql_dataset('./tests/config/read/local_global.yml')
    config = db._get_config(
        'read',
        filepath=None,
        kwargs={},
    )
    assert config == (
        'r.xls',
        {
            'header': 'r_header',
            'sheet_name': 'r_sheet_name',
            'encoding': 'r_encoding',
            'kwarg0': 'r_kwarg0',
            'kwarg1': 'r_kwarg1',
        },
    )


def test__get_config_read_param_overrides_global():
    db = sql_dataset('./tests/config/global.yml')
    config = db._get_config(
        'read',
        **READ_OVERRIDE_PARAMS,
    )
    assert config == (
        'p.xls',
        {
            'header': 'p_header',
            'sheet_name': 'p_sheet_name',
            'encoding': 'p_encoding',
            'kwarg0': 'p_kwarg0',
            'kwarg1': 'p_kwarg1',
        },
    )


def test__get_config_read_param_overrides_local():
    db = sql_dataset('./tests/config/read/local.yml')
    config = db._get_config(
        'read',
        **READ_OVERRIDE_PARAMS,
    )
    assert config == (
        'p.xls',
        {
            'header': 'p_header',
            'sheet_name': 'p_sheet_name',
            'encoding': 'p_encoding',
            'kwarg0': 'p_kwarg0',
            'kwarg1': 'p_kwarg1',
        },
    )


def test__get_config_read_param_overrides_local_overrides_global():
    db = sql_dataset('./tests/config/read/local_global.yml')
    config = db._get_config(
        'read',
        **READ_OVERRIDE_PARAMS,
    )
    assert config == (
        'p.xls',
        {
            'header': 'p_header',
            'sheet_name': 'p_sheet_name',
            'encoding': 'p_encoding',
            'kwarg0': 'p_kwarg0',
            'kwarg1': 'p_kwarg1',
        },
    )


WRITE_OVERRIDE_PARAMS = {
    'filepath': 'p.xls',
    'kwargs': {
        'header': 'p_header',
        'index': 'p_index',
        'sheet_name': 'p_sheet_name',
        'encoding': 'p_encoding',
        'kwarg0': 'p_kwarg0',
        'kwarg1': 'p_kwarg1',
    },
}


def test__get_config_write_default():
    db = sql_dataset('./tests/config/empty.yml')
    config = db._get_config(
        'write',
        filepath=None,
        kwargs={},
    )
    assert config == (
        None,
        {
            'index': False,
            'encoding': 'utf-8-sig',
        }
    )


def test__get_config_write_global():
    db = sql_dataset('./tests/config/global.yml')
    config = db._get_config(
        'write',
        filepath=None,
        kwargs={},
    )
    assert config == (
        'g.xls',
        {
            'header': 'g_header',
            'index': 'g_index',
            'sheet_name': 'g_sheet_name',
            'encoding': 'g_encoding',
        },
    )


def test__get_config_write_local():
    db = sql_dataset('./tests/config/write/local.yml')
    config = db._get_config(
        'write',
        filepath=None,
        kwargs={},
    )
    assert config == (
        'w.xls',
        {
            'header': 'w_header',
            'index': 'w_index',
            'sheet_name': 'w_sheet_name',
            'encoding': 'w_encoding',
            'kwarg0': 'w_kwarg0',
            'kwarg1': 'w_kwarg1',
        },
    )


def test__get_config_read_local_overrides_global():
    db = sql_dataset('./tests/config/write/local_global.yml')
    config = db._get_config(
        'write',
        filepath=None,
        kwargs={},
    )
    assert config == (
        'w.xls',
        {
            'header': 'w_header',
            'index': 'w_index',
            'sheet_name': 'w_sheet_name',
            'encoding': 'w_encoding',
            'kwarg0': 'w_kwarg0',
            'kwarg1': 'w_kwarg1',
        },
    )


def test__get_config_write_param_overrides_global():
    db = sql_dataset('./tests/config/global.yml')
    config = db._get_config(
        'write',
        **WRITE_OVERRIDE_PARAMS,
    )
    assert config == (
        'p.xls',
        {
            'header': 'p_header',
            'index': 'p_index',
            'sheet_name': 'p_sheet_name',
            'encoding': 'p_encoding',
            'kwarg0': 'p_kwarg0',
            'kwarg1': 'p_kwarg1',
        },
    )


def test__get_config_write_param_overrides_local():
    db = sql_dataset('./tests/config/write/local.yml')
    config = db._get_config(
        'write',
        **WRITE_OVERRIDE_PARAMS,
    )
    assert config == (
        'p.xls',
        {
            'header': 'p_header',
            'index': 'p_index',
            'sheet_name': 'p_sheet_name',
            'encoding': 'p_encoding',
            'kwarg0': 'p_kwarg0',
            'kwarg1': 'p_kwarg1',
        },
    )


def test__get_config_write_param_overrides_local_overrides_global():
    db = sql_dataset('./tests/config/write/local_global.yml')
    config = db._get_config(
        'write',
        **WRITE_OVERRIDE_PARAMS,
    )
    assert config == (
        'p.xls',
        {
            'header': 'p_header',
            'index': 'p_index',
            'sheet_name': 'p_sheet_name',
            'encoding': 'p_encoding',
            'kwarg0': 'p_kwarg0',
            'kwarg1': 'p_kwarg1',
        },
    )


def test__get_config_transform_global():
    db = sql_dataset('./tests/config/global.yml')
    config = db._get_config(
        'transform',
        transform_function=None,
    )
    assert config is np.abs


def test__get_config_transform_param_overrides_global():
    def f(x):
        return x
    
    db = sql_dataset('./tests/config/global.yml')
    config = db._get_config(
        'transform',
        transform_function=f,
    )
    assert config is f


QUERY_EMPTY_PARAMS = {
    'conn': None,
    'get_data': None,
    'get_row_count': None,
    'chunksize': None,
    'template_vars': {},
}

QUERY_OVERRIDE_PARAMS = {
    'conn': {
        'server': 'p_conn_server',
        'user': 'p_conn_user',
        'password': 'p_conn_password',
        'database': 'p_conn_database',
        'driver': 'p_conn_driver',
    },
    'get_data': 'p_get_data',
    'get_row_count': 'p_get_row_count',
    'chunksize': 'p_chunksize',
    'template_vars': {
        'p_template_var_0': 'foo',
        'p_template_var_1': 'bar',
    },
}


def test__get_config_query_default():
    db = sql_dataset('./tests/config/empty.yml')
    config = db._get_config(
        'query',
        **QUERY_EMPTY_PARAMS,
    )
    assert config == (
        None,
        None,
        None,
        1000,
        {},
    )


def test__get_config_query_local():
    db = sql_dataset('./tests/config/query/local.yml')
    config = db._get_config(
        'query',
        **QUERY_EMPTY_PARAMS,
    )
    assert config == (
        {
            'server': 'q_conn_server',
            'user': 'q_conn_user',
            'password': 'q_conn_password',
            'database': 'q_conn_database',
            'driver': 'q_conn_driver',
        },
        'q_get_data',
        'q_get_row_count',
        'q_chunksize',
        {},
    )


def test__get_config_query_local_no_conn_global():
    db = sql_dataset('./tests/config/query/local_no_conn_global.yml')
    config = db._get_config(
        'query',
        **QUERY_EMPTY_PARAMS,
    )
    assert config == (
        {
            'server': 'g_conn_server',
            'user': 'g_conn_user',
            'password': 'g_conn_password',
            'database': 'g_conn_database',
            'driver': 'g_conn_driver',
        },
        'q_get_data',
        'q_get_row_count',
        'q_chunksize',
        {},
    )


def test__get_config_query_local_overrides_global():
    db = sql_dataset('./tests/config/query/local_global.yml')
    config = db._get_config(
        'query',
        **QUERY_EMPTY_PARAMS,
    )
    assert config == (
        {
            'server': 'q_conn_server',
            'user': 'q_conn_user',
            'password': 'q_conn_password',
            'database': 'q_conn_database',
            'driver': 'q_conn_driver',
        },
        'q_get_data',
        'q_get_row_count',
        'q_chunksize',
        {},
    )


def test__get_config_query_params_overrides_global():
    db = sql_dataset('./tests/config/global.yml')
    config = db._get_config(
        'query',
        **QUERY_OVERRIDE_PARAMS,
    )
    assert config == (
        {
            'server': 'p_conn_server',
            'user': 'p_conn_user',
            'password': 'p_conn_password',
            'database': 'p_conn_database',
            'driver': 'p_conn_driver',
        },
        'p_get_data',
        'p_get_row_count',
        'p_chunksize',
        {
            'p_template_var_0': 'foo',
            'p_template_var_1': 'bar',
        },
    )


def test__get_config_query_params_overrides_local():
    db = sql_dataset('./tests/config/query/local.yml')
    config = db._get_config(
        'query',
        **QUERY_OVERRIDE_PARAMS,
    )
    assert config == (
        {
            'server': 'p_conn_server',
            'user': 'p_conn_user',
            'password': 'p_conn_password',
            'database': 'p_conn_database',
            'driver': 'p_conn_driver',
        },
        'p_get_data',
        'p_get_row_count',
        'p_chunksize',
        {
            'p_template_var_0': 'foo',
            'p_template_var_1': 'bar',
        },
    )


def test__get_config_query_params_overrides_local_overrides_global():
    db = sql_dataset('./tests/config/query/local_global.yml')
    config = db._get_config(
        'query',
        **QUERY_OVERRIDE_PARAMS,
    )
    assert config == (
        {
            'server': 'p_conn_server',
            'user': 'p_conn_user',
            'password': 'p_conn_password',
            'database': 'p_conn_database',
            'driver': 'p_conn_driver',
        },
        'p_get_data',
        'p_get_row_count',
        'p_chunksize',
        {
            'p_template_var_0': 'foo',
            'p_template_var_1': 'bar',
        },
    )


UPLOAD_EMPTY_PARAMS = {
    'conn': None,
    'table': None,
    'mode': None,
    'bcp': None,
    'truncate': None,
    'schema_sample': None,
    'chunksize': None,
    'verbose': None,
}

UPLOAD_OVERRIDE_PARAMS = {
    'conn': {
        'server': 'p_conn_server',
        'user': 'p_conn_user',
        'password': 'p_conn_password',
        'database': 'p_conn_database',
        'driver': 'p_conn_driver',
    },
    'table': 'p_table',
    'mode': 'p_mode',
    'bcp': 'p_bcp',
    'truncate': 'p_truncate',
    'schema_sample': 'p_schema_sample',
    'chunksize': 'p_chunksize',
    'verbose': 'p_verbose',
}


def test__get_config_upload_default():
    db = sql_dataset('./tests/config/empty.yml')
    config = db._get_config(
        'upload',
        **UPLOAD_EMPTY_PARAMS,
    )
    assert config == (
        None,
        None,
        'append',
        True,
        False,
        None,
        1000,
        False,
    )


def test__get_config_upload_local():
    db = sql_dataset('./tests/config/upload/local.yml')
    config = db._get_config(
        'upload',
        **UPLOAD_EMPTY_PARAMS,
    )
    assert config == (
        {
            'server': 'u_conn_server',
            'user': 'u_conn_user',
            'password': 'u_conn_password',
            'database': 'u_conn_database',
            'driver': 'u_conn_driver',
        },
        'u_table',
        'u_mode',
        'u_bcp',
        'u_truncate',
        'u_schema_sample',
        'u_chunksize',
        'u_verbose',
    )


def test__get_config_upload_local_no_conn_global():
    db = sql_dataset('./tests/config/upload/local_no_conn_global.yml')
    config = db._get_config(
        'upload',
        **UPLOAD_EMPTY_PARAMS,
    )
    assert config == (
        {
            'server': 'g_conn_server',
            'user': 'g_conn_user',
            'password': 'g_conn_password',
            'database': 'g_conn_database',
            'driver': 'g_conn_driver',
        },
        'u_table',
        'u_mode',
        'u_bcp',
        'u_truncate',
        'u_schema_sample',
        'u_chunksize',
        'u_verbose',
    )


def test__get_config_upload_local_overrides_global():
    db = sql_dataset('./tests/config/upload/local_global.yml')
    config = db._get_config(
        'upload',
        **UPLOAD_EMPTY_PARAMS,
    )
    assert config == (
        {
            'server': 'u_conn_server',
            'user': 'u_conn_user',
            'password': 'u_conn_password',
            'database': 'u_conn_database',
            'driver': 'u_conn_driver',
        },
        'u_table',
        'u_mode',
        'u_bcp',
        'u_truncate',
        'u_schema_sample',
        'u_chunksize',
        'u_verbose',
    )


def test__get_config_upload_params_overrides_global():
    db = sql_dataset('./tests/config/global.yml')
    config = db._get_config(
        'upload',
        **UPLOAD_OVERRIDE_PARAMS,
    )
    assert config == (
        {
            'server': 'p_conn_server',
            'user': 'p_conn_user',
            'password': 'p_conn_password',
            'database': 'p_conn_database',
            'driver': 'p_conn_driver',
        },
        'p_table',
        'p_mode',
        'p_bcp',
        'p_truncate',
        'p_schema_sample',
        'p_chunksize',
        'p_verbose',
    )


def test__get_config_upload_params_overrides_local():
    db = sql_dataset('./tests/config/upload/local.yml')
    config = db._get_config(
        'upload',
        **UPLOAD_OVERRIDE_PARAMS,
    )
    assert config == (
        {
            'server': 'p_conn_server',
            'user': 'p_conn_user',
            'password': 'p_conn_password',
            'database': 'p_conn_database',
            'driver': 'p_conn_driver',
        },
        'p_table',
        'p_mode',
        'p_bcp',
        'p_truncate',
        'p_schema_sample',
        'p_chunksize',
        'p_verbose',
    )


def test__get_config_upload_params_overrides_local_overrides_global():
    db = sql_dataset('./tests/config/upload/local_global.yml')
    config = db._get_config(
        'upload',
        **UPLOAD_OVERRIDE_PARAMS,
    )
    assert config == (
        {
            'server': 'p_conn_server',
            'user': 'p_conn_user',
            'password': 'p_conn_password',
            'database': 'p_conn_database',
            'driver': 'p_conn_driver',
        },
        'p_table',
        'p_mode',
        'p_bcp',
        'p_truncate',
        'p_schema_sample',
        'p_chunksize',
        'p_verbose',
    )


SEND_CMD_EMPTY_PARAMS = {
    'cmd': None,
    'conn': None,
    'verbose': None,
}

SEND_CMD_OVERRIDE_PARAMS = {
    'cmd': 'p_cmd',
    'conn': {
        'server': 'p_conn_server',
        'user': 'p_conn_user',
        'password': 'p_conn_password',
        'database': 'p_conn_database',
        'driver': 'p_conn_driver',
    },
    'verbose': 'p_verbose',
}


def test__get_config_send_cmd_default():
    db = sql_dataset('./tests/config/empty.yml')
    config = db._get_config(
        'send_cmd',
        **SEND_CMD_EMPTY_PARAMS,
    )
    assert config == (
        None,
        None,
        False,
    )


def test__get_config_send_cmd_local():
    db = sql_dataset('./tests/config/send_cmd/local.yml')
    config = db._get_config(
        'send_cmd',
        **SEND_CMD_EMPTY_PARAMS,
    )
    assert config == (
        'sc_cmd',
        {
            'server': 'sc_conn_server',
            'user': 'sc_conn_user',
            'password': 'sc_conn_password',
            'database': 'sc_conn_database',
            'driver': 'sc_conn_driver',
        },
        'sc_verbose',
    )


def test__get_config_send_cmd_local_no_conn_global():
    db = sql_dataset('./tests/config/send_cmd/local_no_conn_global.yml')
    config = db._get_config(
        'send_cmd',
        **SEND_CMD_EMPTY_PARAMS,
    )
    assert config == (
        'sc_cmd',
        {
            'server': 'g_conn_server',
            'user': 'g_conn_user',
            'password': 'g_conn_password',
            'database': 'g_conn_database',
            'driver': 'g_conn_driver',
        },
        'sc_verbose',
    )


def test__get_config_send_cmd_local_overrides_global():
    db = sql_dataset('./tests/config/send_cmd/local_global.yml')
    config = db._get_config(
        'send_cmd',
        **SEND_CMD_EMPTY_PARAMS,
    )
    assert config == (
        'sc_cmd',
        {
            'server': 'sc_conn_server',
            'user': 'sc_conn_user',
            'password': 'sc_conn_password',
            'database': 'sc_conn_database',
            'driver': 'sc_conn_driver',
        },
        'sc_verbose',
    )


def test__get_config_send_cmd_params_overrides_global():
    db = sql_dataset('./tests/config/global.yml')
    config = db._get_config(
        'send_cmd',
        **SEND_CMD_OVERRIDE_PARAMS,
    )
    assert config == (
        'p_cmd',
        {
            'server': 'p_conn_server',
            'user': 'p_conn_user',
            'password': 'p_conn_password',
            'database': 'p_conn_database',
            'driver': 'p_conn_driver',
        },
        'p_verbose',
    )


def test__get_config_send_cmd_params_overrides_local():
    db = sql_dataset('./tests/config/send_cmd/local.yml')
    config = db._get_config(
        'send_cmd',
        **SEND_CMD_OVERRIDE_PARAMS,
    )
    assert config == (
        'p_cmd',
        {
            'server': 'p_conn_server',
            'user': 'p_conn_user',
            'password': 'p_conn_password',
            'database': 'p_conn_database',
            'driver': 'p_conn_driver',
        },
        'p_verbose',
    )


def test__get_config_send_cmd_params_overrides_local_overrides_global():
    db = sql_dataset('./tests/config/send_cmd/local_global.yml')
    config = db._get_config(
        'send_cmd',
        **SEND_CMD_OVERRIDE_PARAMS,
    )
    assert config == (
        'p_cmd',
        {
            'server': 'p_conn_server',
            'user': 'p_conn_user',
            'password': 'p_conn_password',
            'database': 'p_conn_database',
            'driver': 'p_conn_driver',
        },
        'p_verbose',
    )
