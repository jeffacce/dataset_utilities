import pandas as pd
import numpy as np

def rand_str_array(n, str_len):
    result = pd.DataFrame(np.random.randint(97, 123, (n, str_len)))
    return result.applymap(chr).sum(axis=1)

def rand_float_array(n):
    return pd.Series(np.random.random(n))

def rand_dt_array(n):
    start = int(pd.Timestamp('2020-01-01 12:00:00 AM').value / 1e9)
    end = int(pd.Timestamp('2021-01-01 12:00:00 AM').value / 1e9)
    result = np.random.randint(start, end, (n))
    result = pd.to_datetime(result, unit='s')
    return result

def rand_df(n):
    # gen random data of datetime, string, and float
    return pd.DataFrame({
        'dt': rand_dt_array(n),
        'id': rand_str_array(n, 50),
        'float': rand_float_array(n),
    })

df = rand_df(100000)
df.to_csv('test_data.csv', encoding='utf-8-sig', index=False)
