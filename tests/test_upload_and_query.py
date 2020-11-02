import pytest
from ..dataset import sql_dataset
import pandas as pd
import numpy as np


def test_read_upload_bcp_query(verbose=False):
    sd = sql_dataset('./tests/database.yml').read()
    sd.data['dt'] = pd.to_datetime(sd.data['dt'])
    df_orig = sd.data.copy()
    sd.upload(mode='overwrite_table', bcp=True, verbose=verbose)
    df_queried = sd.query().data

    return (df_queried == df_orig).all()


def test_read_upload_pyodbc_query(verbose=False):
    sd = sql_dataset('./tests/database.yml').read()
    sd.data['dt'] = pd.to_datetime(sd.data['dt'])
    df_orig = sd.data.copy()
    sd.upload(mode='overwrite_table', bcp=False, verbose=verbose)
    df_queried = sd.query().data

    return (df_queried == df_orig).all()


if __name__ == '__main__':
    test_read_upload_pyodbc_query(True)
    test_read_upload_bcp_query(True)
