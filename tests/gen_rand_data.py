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

def empty_series(n):
    return pd.Series([np.nan] * n)

def rand_df(n):
    # gen random data of datetime, string, and float
    return pd.DataFrame({
        'dt': rand_dt_array(n),
        'uid': rand_str_array(n, 50),
        'empty_col': empty_series(n),
        'float': rand_float_array(n, na_ratio=0.2),
        'float2': rand_float_array(n, na_ratio=0),
        'name': rand_str_array(n, 30),
        'flag': rand_bool_array(n, na_ratio=0.2),
        'flag2': rand_bool_array(n, na_ratio=0),
    })


if __name__ == '__main__':
    df = rand_df(100000)
    df.to_csv('test_data.csv', encoding='utf-8-sig', index=False)

