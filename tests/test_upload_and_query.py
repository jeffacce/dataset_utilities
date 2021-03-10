import pytest
import pandas as pd
import numpy as np
import requests
import os
from ..dataset import sql_dataset
from .gen_rand_data import rand_df


@pytest.fixture(scope='session')
def gen_test_csv(request):
    print('here')
    df = rand_df(100000)
    df.to_csv('./tests/test_data.csv', encoding='utf-8-sig', index=False)
    def finalize():
        print('hi')
        os.remove('./tests/test_data.csv')
    request.addfinalizer(finalize)

def test_read_upload_query_bcp(gen_test_csv, verbose=True):
    sd = sql_dataset('./tests/database.yml').read()
    sd.data['dt'] = pd.to_datetime(sd.data['dt'])
    df_orig = sd.data.copy()

    sd.upload(mode='overwrite_table', bcp=True, verbose=verbose)
    df_queried = sd.query().data
    pd.testing.assert_frame_equal(df_queried, df_orig, check_names=False)

    sd.upload(mode='overwrite_data', bcp=True, verbose=verbose)
    df_queried = sd.query().data
    pd.testing.assert_frame_equal(df_queried, df_orig, check_names=False)

def test_read_upload_query_pyodbc(gen_test_csv, verbose=True):
    sd = sql_dataset('./tests/database.yml').read()
    sd.data['dt'] = pd.to_datetime(sd.data['dt'])
    df_orig = sd.data.copy()

    sd.upload(mode='overwrite_table', bcp=False, verbose=verbose)
    df_queried = sd.query().data
    pd.testing.assert_frame_equal(df_queried, df_orig, check_names=False)

    sd.upload(mode='overwrite_data', bcp=False, verbose=verbose)
    df_queried = sd.query().data
    pd.testing.assert_frame_equal(df_queried, df_orig, check_names=False)

def test_ping_fake_database_raises_connection_error(gen_test_csv):
    sd = sql_dataset('./tests/fake_database.yml').read()
    assert not sd.ping(max_retries=1)

