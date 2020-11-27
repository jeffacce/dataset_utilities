import pytest
from ..dataset import sql_dataset
import pandas as pd
import numpy as np
import requests


def test_read_upload_bcp_query_overwrite_table(verbose=True):
    sd = sql_dataset('./tests/database.yml').read()
    sd.data['dt'] = pd.to_datetime(sd.data['dt'])
    df_orig = sd.data.copy()
    sd.upload(mode='overwrite_table', bcp=True, verbose=verbose)
    df_queried = sd.query().data
    return (df_queried == df_orig).all()

def test_read_upload_bcp_query_overwrite_data(verbose=True):
    sd = sql_dataset('./tests/database.yml').read()
    sd.data['dt'] = pd.to_datetime(sd.data['dt'])
    df_orig = sd.data.copy()
    sd.upload(mode='overwrite_data', bcp=True, verbose=verbose)
    df_queried = sd.query().data
    return (df_queried == df_orig).all()

def test_read_upload_pyodbc_query_overwrite_table(verbose=True):
    sd = sql_dataset('./tests/database.yml').read()
    sd.data['dt'] = pd.to_datetime(sd.data['dt'])
    df_orig = sd.data.copy()
    sd.upload(mode='overwrite_table', bcp=False, verbose=verbose)
    df_queried = sd.query().data
    return (df_queried == df_orig).all()

def test_read_upload_pyodbc_query_overwrite_data(verbose=True):
    sd = sql_dataset('./tests/database.yml').read()
    sd.data['dt'] = pd.to_datetime(sd.data['dt'])
    df_orig = sd.data.copy()
    sd.upload(mode='overwrite_data', bcp=False, verbose=verbose)
    df_queried = sd.query().data
    return (df_queried == df_orig).all()

def test_ping_fake_database_raises_connection_error():
    sd = sql_dataset('./tests/fake_database.yml').read()
    assert not sd.ping(max_retries=1)
