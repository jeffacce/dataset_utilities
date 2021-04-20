import pandas as pd
import numpy as np

# seeded RNG
rng = np.random.RandomState(42)

def rand_str_array(n, str_len):
    result = pd.DataFrame(rng.randint(97, 123, (n, str_len)))
    return result.applymap(chr).sum(axis=1)

def rand_float_array(n, na_ratio=0):
    result = pd.Series(rng.random(n))
    if na_ratio > 0:
        na_idx = rng.choice(range(n), size=int(min(na_ratio, 1) * n), replace=False)
        result.loc[na_idx] = np.nan
    return result

def rand_dt_array(n):
    start = int(pd.Timestamp('2020-01-01 12:00:00 AM').value / 1e9)
    end = int(pd.Timestamp('2021-01-01 12:00:00 AM').value / 1e9)
    result = rng.randint(start, end, (n))
    result = pd.to_datetime(result, unit='s')
    return result

def rand_bool_array(n, na_ratio=0):
    result = rng.randint(0, 2, (n))
    result = pd.Series(result)
    if na_ratio > 0:
        na_idx = rng.choice(range(n), size=int(min(na_ratio, 1) * n), replace=False)
        result.loc[na_idx] = np.nan
    result = result.replace({0:False, 1:True})
    return result

def rand_int_array(n, lower, upper, na_ratio=0):
    result = rng.randint(lower, upper, (n)) 
    result = pd.Series(result).astype('Int64')
    if na_ratio > 0:
        na_idx = rng.choice(range(n), size=int(min(na_ratio, 1) * n), replace=False)
        result.loc[na_idx] = pd.NA 
    return result

def empty_series(n):
    return pd.Series([np.nan] * n)

def rand_df(n):
    # gen random data of datetime, string, and float
    result = pd.DataFrame({
        'dt': rand_dt_array(n),
        'uid': rand_str_array(n, 50),
        'name': rand_str_array(n, 30),
        'empty_col': empty_series(n),
        'float': rand_float_array(n, na_ratio=0),
        'float_k': rand_float_array(n, na_ratio=0) * 1e3,
        'float_m': rand_float_array(n, na_ratio=0) * 1e6,
        'float_b': rand_float_array(n, na_ratio=0) * 1e9,
        'float_na': rand_float_array(n, na_ratio=0.2),
        'bit': rand_int_array(n, 0, 2, na_ratio=0),
        'bit_na': rand_int_array(n, 0, 2, na_ratio=0.2),
        'tinyint': rand_int_array(n, 0, 2**8, na_ratio=0),
        'tinyint_na': rand_int_array(n, 0, 2**8, na_ratio=0.2),
        'smallint': rand_int_array(n, -2**15, 2**15, na_ratio=0),
        'smallint_na': rand_int_array(n, -2**15, 2**15, na_ratio=0.2),
        'int': rand_int_array(n, -2**31, 2**31, na_ratio=0),
        'int_na': rand_int_array(n, -2**31, 2**31, na_ratio=0.2),
        'bigint': rand_int_array(n, -2**63, 2**63, na_ratio=0),
        'bigint_na': rand_int_array(n, -2**63, 2**63, na_ratio=0.2),
        'bool': rand_bool_array(n, na_ratio=0),
        'bool_na': rand_bool_array(n, na_ratio=0.2),
    })
    result['empty_str_col'] = ''


if __name__ == '__main__':
    df = rand_df(100000)
    df.to_csv('test_data.csv', encoding='utf-8-sig', index=False)

