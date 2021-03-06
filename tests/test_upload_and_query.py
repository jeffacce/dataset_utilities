from .test_dataset import CMD_CREATE_TEST_TABLE
import pytest
import pandas as pd
import numpy as np
import os
from ..dataset import sql_dataset
from .gen_rand_data import rand_df


CMD_DROP_TEST_TABLE_IF_EXISTS = "IF OBJECT_ID('test_table', 'U') IS NOT NULL DROP TABLE test_table;"
CMD_CREATE_TRUNCATED_TEST_TABLE = """
    CREATE TABLE test_table (
        [dt] datetime NULL,
        [uid] nvarchar(10) NULL,
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
        [smallint] smallint NULL,
        [smallint_na] smallint NULL,
        [int] int NULL,
        [int_na] int NULL,
        [bigint] bigint NULL,
        [bigint_na] bigint NULL,
        [bool] bit NULL,
        [bool_na] bit NULL,
        [empty_str_col] nvarchar(100) NULL
    );
"""


def cleanup_test_data_csv():
    try:
        os.remove('./tests/test_data.csv')
    except:
        pass


def cleanup_test_data_copy_csv():
    try:
        os.remove('./tests/test_data_copy.csv')
    except:
        pass


@pytest.fixture(scope='session')
def gen_test_csv(request):
    df = rand_df(100000)
    df.to_csv('./tests/test_data.csv', encoding='utf-8-sig', index=False)
    request.addfinalizer(cleanup_test_data_csv)


def test_read_upload_query_bcp(gen_test_csv, verbose=True):
    sd = sql_dataset('./tests/config/integration/database.yml').read()
    sd.data['dt'] = pd.to_datetime(sd.data['dt'])
    df_orig = sd.data.copy()

    sd.send_cmd(CMD_DROP_TEST_TABLE_IF_EXISTS)

    sd.upload(mode='overwrite_table', bcp=True, verbose=verbose)
    df_queried = sd.query().data
    pd.testing.assert_frame_equal(df_queried, df_orig, check_dtype=False, check_names=False)

    sd.upload(mode='overwrite_data', bcp=True, verbose=verbose)
    df_queried = sd.query().data
    pd.testing.assert_frame_equal(df_queried, df_orig, check_dtype=False, check_names=False)

    sd.send_cmd(CMD_DROP_TEST_TABLE_IF_EXISTS)


def test_read_upload_query_bcp_truncate(gen_test_csv, verbose=True):
    sd = sql_dataset('./tests/config/integration/database.yml').read()
    sd.data['dt'] = pd.to_datetime(sd.data['dt'])
    df_orig = sd.data.copy()

    # create a table too short to test upload(truncate=True/False)
    sd.send_cmd(CMD_DROP_TEST_TABLE_IF_EXISTS)
    sd.send_cmd(CMD_CREATE_TRUNCATED_TEST_TABLE)

    with pytest.raises(ValueError):
        # should raise errors because it won't fit
        sd.upload(bcp=True, truncate=False, verbose=verbose, mode='overwrite_data')
    sd.upload(bcp=True, truncate=True, verbose=verbose, mode='overwrite_data')
    df_queried = sd.query().data

    # truncate df_orig accordingly for equality assertion
    df_orig['uid'] = df_orig['uid'].str[:10]
    df_orig['name'] = df_orig['name'].str[:10]
    df_orig['float'] = df_orig['float'].round(3)
    df_orig['float_k'] = df_orig['float_k'].round(3)
    df_orig['float_na'] = df_orig['float_na'].round(3)
    pd.testing.assert_frame_equal(df_queried, df_orig, check_dtype=False, check_names=False)
    
    sd.send_cmd(CMD_DROP_TEST_TABLE_IF_EXISTS)


def test_read_upload_query_pyodbc(gen_test_csv, verbose=True):
    sd = sql_dataset('./tests/config/integration/database.yml').read()
    sd.data['dt'] = pd.to_datetime(sd.data['dt'])
    df_orig = sd.data.copy()

    sd.send_cmd(CMD_DROP_TEST_TABLE_IF_EXISTS)

    sd.upload(mode='overwrite_table', bcp=False, verbose=verbose)
    df_queried = sd.query().data
    pd.testing.assert_frame_equal(df_queried, df_orig, check_dtype=False, check_names=False)

    sd.upload(mode='overwrite_data', bcp=False, verbose=verbose)
    df_queried = sd.query().data
    pd.testing.assert_frame_equal(df_queried, df_orig, check_dtype=False, check_names=False)

    sd.send_cmd(CMD_DROP_TEST_TABLE_IF_EXISTS)


def test_read_upload_query_pyodbc_truncate(gen_test_csv, verbose=True):
    sd = sql_dataset('./tests/config/integration/database.yml').read()
    sd.data['dt'] = pd.to_datetime(sd.data['dt'])
    df_orig = sd.data.copy()

    # create a table too short to test upload(truncate=True/False)
    sd.send_cmd(CMD_DROP_TEST_TABLE_IF_EXISTS)
    sd.send_cmd(CMD_CREATE_TRUNCATED_TEST_TABLE)

    with pytest.raises(ValueError):
        # should raise errors because it won't fit
        sd.upload(bcp=False, truncate=False, verbose=verbose, mode='overwrite_data')
    sd.upload(bcp=False, truncate=True, verbose=verbose, mode='overwrite_data')
    df_queried = sd.query().data

    # truncate df_orig accordingly for equality assertion
    df_orig['uid'] = df_orig['uid'].str[:10]
    df_orig['name'] = df_orig['name'].str[:10]
    df_orig['float'] = df_orig['float'].round(3)
    df_orig['float_k'] = df_orig['float_k'].round(3)
    df_orig['float_na'] = df_orig['float_na'].round(3)

    pd.testing.assert_frame_equal(df_queried, df_orig, check_dtype=False, check_names=False)

    sd.send_cmd(CMD_DROP_TEST_TABLE_IF_EXISTS)


def test_read_upload_query_write_bcp(gen_test_csv, verbose=True):
    sd = sql_dataset('./tests/config/integration/read_upload_query_write.yml').read()
    sd.data['dt'] = pd.to_datetime(sd.data['dt'])
    df_orig = sd.data.copy()

    sd.send_cmd(CMD_DROP_TEST_TABLE_IF_EXISTS)
    cleanup_test_data_copy_csv()

    sd.upload(mode='overwrite_table', bcp=True, verbose=verbose)
    sd.query().write()
    df_queried = pd.read_csv('./tests/test_data_copy.csv')
    df_queried['dt'] = pd.to_datetime(df_queried['dt'])
    pd.testing.assert_frame_equal(df_queried, df_orig, check_dtype=False, check_names=False)

    cleanup_test_data_copy_csv()

    sd.upload(mode='overwrite_data', bcp=True, verbose=verbose)
    sd.query().write()
    df_queried = pd.read_csv('./tests/test_data_copy.csv')
    df_queried['dt'] = pd.to_datetime(df_queried['dt'])
    pd.testing.assert_frame_equal(df_queried, df_orig, check_dtype=False, check_names=False)

    sd.send_cmd(CMD_DROP_TEST_TABLE_IF_EXISTS)
    cleanup_test_data_copy_csv()


def test_read_upload_query_write_pyodbc(gen_test_csv, verbose=True):
    sd = sql_dataset('./tests/config/integration/read_upload_query_write.yml').read()
    sd.data['dt'] = pd.to_datetime(sd.data['dt'])
    df_orig = sd.data.copy()

    sd.send_cmd(CMD_DROP_TEST_TABLE_IF_EXISTS)
    cleanup_test_data_copy_csv()

    sd.upload(mode='overwrite_table', bcp=False, verbose=verbose)
    sd.query().write()
    df_queried = pd.read_csv('./tests/test_data_copy.csv')
    df_queried['dt'] = pd.to_datetime(df_queried['dt'])
    pd.testing.assert_frame_equal(df_queried, df_orig, check_dtype=False, check_names=False)

    cleanup_test_data_copy_csv()

    sd.upload(mode='overwrite_data', bcp=False, verbose=verbose)
    sd.query().write()
    df_queried = pd.read_csv('./tests/test_data_copy.csv')
    df_queried['dt'] = pd.to_datetime(df_queried['dt'])
    pd.testing.assert_frame_equal(df_queried, df_orig, check_dtype=False, check_names=False)

    sd.send_cmd(CMD_DROP_TEST_TABLE_IF_EXISTS)
    cleanup_test_data_copy_csv()
