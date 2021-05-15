import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
import datetime
import warnings
import pyodbc
import os
import importlib
import uuid
import subprocess
import requests
import sys
import time


# vectorized and adapted from:
# https://stackoverflow.com/questions/3018758/determine-precision-and-scale-of-particular-number-in-python
def magnitude_and_scale(x):
    MAX_DIGITS = 15 # originally 14; change back to 14 if buggy

    x_abs = x.replace([-np.inf, np.inf], np.nan).dropna().abs()
    if len(x_abs) == 0:
        raise ValueError('Empty series after dropping NaN, inf.')

    int_part = x_abs.astype(np.int64)
    magnitude = np.log10(int_part.replace(0, 1)).astype(int) + 1
    magnitude = magnitude.clip(1, MAX_DIGITS).astype(int).max()
    frac_part = x_abs - int_part
    multiplier = 10 ** np.int64(MAX_DIGITS - magnitude)
    frac_digits = (multiplier + (multiplier * frac_part + 0.5).astype(np.int64))
    while np.all(frac_digits % 10 == 0):
        frac_digits /= 10
    scale = np.log10(frac_digits.min()).astype(int)
    
    return magnitude, scale


def get_type(x, force_allow_null=False):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan)
    has_null = (x.isna().sum() > 0) or force_allow_null
    params = []
    comment = ''
    x = x.dropna().reset_index(drop=True)

    MAX_PRECISION = 38
    
    if len(x) == 0:
        has_null = True
        dtype = 'nvarchar'
        params = [255]
        comment = 'empty column, defaulting to nvarchar(255)'
    else:
        if pd.api.types.is_object_dtype(x):
            unique_types = x.apply(type).unique()
            if (len(unique_types) == 1) and (unique_types[0] is datetime.date):
                dtype = 'date'
            elif (len(unique_types) == 1) and ((unique_types[0] is datetime.datetime) or (unique_types[0] is pd.Timestamp)):
                dtype = 'datetime'
            elif set(x) == set([False, True]):
                # NAs are dropped at the start of this function.
                # But if the original bool series had NAs,
                # its dtype will be object, not bool (numeric).
                dtype = 'bit'
            else:
                dtype = 'nvarchar'
                size = int(pd.Series(x.unique()).astype(str).str.len().max())
                if size > 4000:
                    comment = 'Maximum string length is %s. Using nvarchar(max).' % size
                    size = 'max'
                    warnings.warn(comment)
                elif size == 0:
                    # if the entire column is empty but non-null strings,
                    # set default to nvarchar(255)
                    size = 255
                    comment = 'zero-length string column, defaulting to nvarchar(255)'
                else:
                    size = min(int(size * 2), 4000)
                params = [size]
        elif pd.api.types.is_numeric_dtype(x):
            magnitude, scale = magnitude_and_scale(x)
            if (scale == 0) or pd.api.types.is_integer_dtype(x) or pd.api.types.is_bool_dtype(x):
                if ((x == 0).all()):
                    dtype = 'int'
                    comment = 'column contains only zeros; defaulting to int'
                elif ((x >= 0) & (x <= 1)).all():
                    dtype = 'bit'
                elif ((x >= 0) & (x <= 2**8-1)).all():
                    dtype = 'tinyint'
                elif ((x >= -2**15) & (x <= 2**15-1)).all():
                    dtype = 'smallint'
                elif ((x >= -2**31) & (x <= 2**31-1)).all():
                    dtype = 'int'
                else:
                    dtype = 'bigint'
            else:
                precision = magnitude + scale
                if precision > MAX_PRECISION:
                    dtype = 'nvarchar'
                    params = [255]
                    comment = 'number too big for decimal, defaulting to nvarchar(255)'
                else:
                    ratio = min(MAX_PRECISION / precision, 1.5)
                    magnitude = int(magnitude * ratio)
                    scale = int(scale * ratio)
                    precision = magnitude + scale
                    dtype = 'decimal'
                    params = [precision, scale]
        elif pd.api.types.is_datetime64_any_dtype(x):
            dtype = 'datetime'
        else:
            dtype = 'nvarchar'
            params = [255]
            comment = 'unable to infer type, defaulting to nvarchar(255)'
    
    return dtype, params, has_null, comment


def indent(str_arr, times=1, spaces=4):
    indentation = ' ' * spaces
    return [indentation + x for x in str_arr]


def get_df_type(df, force_allow_null=False, sample=None, verbose=False):
    if not sample is None:
        df = df.sample(sample)
    cols = df.columns
    if verbose:
        cols = tqdm(cols)
    result = []

    for col in cols:
        dtype, params, has_null, comment = get_type(df[col], force_allow_null=force_allow_null)
        result.append([col, dtype, params, has_null, comment])
    return result


def cast_and_clean_df(
    df,
    df_types=None,
    truncate=False,
    force_allow_null=False,
    sample=None,
    verbose=False,
):
    result = df.copy()

    if df_types is None:
        df_types = get_df_type(df, force_allow_null=force_allow_null, sample=sample, verbose=verbose)
        truncate = True  # safe to truncate since df_types are derived from the actual data

    for col in df_types:
        colname, dtype, params, has_null, comment = col
        if dtype == 'decimal':
            magnitude, scale = params
            result[colname] = result[colname].round(scale)
            if result[colname].isin([np.inf, -np.inf]).any():
                warnings.warn('MS SQL Server does not support infinity. Replacing with NaN.')
                result[colname].replace([np.inf, -np.inf], np.nan, inplace=True)
        elif dtype in ['bit', 'tinyint', 'smallint', 'int', 'bigint']:
            if dtype == 'bit':
                # cast boolean to 0/1/NaN
                result[colname].replace({False: 0, True: 1}, inplace=True)
            # Int64 dtype (pandas>=0.24) to deal with NaN casting int to float
            # this fixes int types having decimal points when uploaded into nvarchar columns
            result[colname] = result[colname].astype('Int64')
        elif dtype in ['varchar', 'nvarchar']:
            if not result[colname].isna().all():
                size = result[colname].dropna().astype(str).str.len().max()
                if size > params[0]:
                    msg = '[{colname}] max length ({size}) exceeds {dtype}({params[0]}).'.format(
                        colname=colname,
                        size=size,
                        dtype=dtype,
                        params=params,
                    )
                    if truncate:
                        if verbose:
                            print(msg)
                        result[colname] = result[colname].str[:size]
                    else:
                        warnings.warn(msg + ' Specify `truncate=True` to force truncation.')
    

    return result


def get_create_statement(df_types, table_name):
    template = '''CREATE TABLE %s (\n%s\n);'''

    col_defs = []
    for col in df_types:
        colname, dtype, params, has_null, comment = col
        col_def = '[%s] %s' % (colname, dtype)
        if len(comment) > 0:
            col_defs.append('-- %s' % comment)
        if len(params) > 0:
            params = ','.join(str(x) for x in params)
            col_def += '(%s)' % params
        if has_null:
            col_def += ' NULL'
        col_defs.append(col_def)
    col_defs = ',\n'.join(indent(col_defs))
    
    return template % (table_name, col_defs)


def _try_import(s):
    arr = s.lstrip('.').split('.')

    if len(arr) == 1:
        package_name = None
        module_name = arr[0]
        method_name = None
    elif len(arr) == 2:
        package_name = None
        module_name = arr[0]
        method_name = arr[1]
    elif len(arr) >= 3:
        package_name = arr[0]
        module_name = '.' + '.'.join(arr[1:-1])
        method_name = arr[-1]
    
    if package_name is None:
        module = importlib.import_module(module_name)
    else:
        module = importlib.import_module(module_name, package=package_name)
    if method_name is None:
        return module
    else:
        return getattr(module, method_name)


class dataset:
    def __init__(self, config_path=None):
        if config_path is None:
            self.config = {}
        else:
            with open(config_path, 'r', encoding='utf-8-sig') as f:
                self.config = yaml.safe_load(f)
        
        if 'transform' in self.config:
            self.transform_function = _try_import(self.config['transform'])
        else:
            self.transform_function = None
    
    def read(self, filepath=None):
        if filepath is None:
            if 'filepath' in self.config:
                filepath = self.config['filepath']
            else:
                raise ValueError('filepath must be specified here or in the config file.')
        
        ext = os.path.splitext(filepath)[1]
        if ext == '.csv':
            self.data = pd.read_csv(
                self.config['filepath'],
                encoding=self.config.get('encoding'),
                float_precision='round_trip',
            )
        elif ext in ['.h5', '.hdf5', '.hdf']:
            self.data = pd.read_hdf(self.config['filepath'])
        elif ext in ['.xls', '.xlsx']:
            self.data = pd.read_excel(
                self.config['filepath'],
                header=self.config.get('header', 0),
                sheet_name=self.config.get('sheet_name', 0),
            )
        else:
            raise ValueError('Only .h5/hdf/hdf5/csv/xls/xlsx supported.')
        return self
    
    def write(self, filepath=None):
        if filepath is None:
            if 'filepath' in self.config:
                filepath = self.config['filepath']
            else:
                raise ValueError('filepath must be specified here or in the config file.')
        ext = os.path.splitext(filepath)[1]
        if ext == '.csv':
            self.data.to_csv(filepath, index=False, encoding='utf-8-sig')
        elif ext in ['.h5', '.hdf5', '.hdf']:
            self.data.to_hdf(filepath, index=False, encoding='utf-8-sig')
        elif ext in ['.xls', 'xlsx']:
            self.data.to_excel(filepath, index=False, encoding='utf-8-sig')
        else:
            raise ValueError('Only .h5/hdf/hdf5/csv/xls/xlsx supported.')
        return self
    
    def transform(self, transform_function=None):
        if not transform_function is None:
            self.data = transform_function(self.data)
        else:
            if not self.transform_function is None:
                self.data = self.transform_function(self.data)
        return self


class sql_dataset(dataset):
    def __init__(self, config_path=None):
        super().__init__(config_path=config_path)
    
    def _format_host_config_args(self, conn):
        result = [
            '-S', conn['server'],
            '-d', conn['database'],
        ]
        if ('user' in conn) or ('uid' in conn):
            result += ['-U', conn.get('user') or conn.get('uid')]
        if ('password' in conn) or ('pwd' in conn):
            result += ['-P', conn.get('password') or conn.get('pwd')]
        if ('trusted_connection') in [x.lower() for x in conn.keys()]:
            result += ['-T']
        return result
    
    def _connect(self, conn, max_retries=3, delay=5, verbose=False, **kwargs):
        '''
        Connect to a database with retries.
        Returns a pyodbc connection (success), or raises `ConnectionError` (failure).
        `conn`: connection config dictionary for `pyodbc`.
        `max_retries`: maximum number of tries to ping the database.
        `delay`: initial delay after failure (seconds). Each successive delay will be doubled in time.
        `verbose`: verbose output. Default False.
        `kwargs`: extra keyword arguments are passed to `pyodbc.connect`.
        '''
        success = False
        retries = 0
        while (not success) and retries < max_retries:
            try:
                if verbose:
                    print('Connecting to database... Try %s/%s' % (retries + 1, max_retries))
                conn = pyodbc.connect(**self.config['conn'], **kwargs)
                result = pd.read_sql('SELECT 1;', conn).values.item()
                success = (result == 1)
            except:
                retries += 1
                if verbose:
                    print('- Error:', sys.exc_info()[1])
                    print('- Retry in %s seconds.' % delay)
                time.sleep(delay)
                delay *= 2 # exponential decay for retry delay
        if success:
            if verbose:
                print('Connected.')
        else:
            if verbose:
                print('Failed to connect.')
            raise requests.ConnectionError('Failed to connect to database.')
        return conn
    
    def _get_table_schema(self, conn, table):
        # TODO: conn is a dict. Check if this is desired.
        template = '''
            SELECT
                COLUMN_NAME, IS_NULLABLE, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION, NUMERIC_SCALE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '%s'
        '''
        query = template % table
        if self._table_exists(conn, table):
            conn = self._connect(conn)
            df = pd.read_sql(query, conn)
            result = []
            for i in range(len(df)):
                row = df.iloc[i,:]
                # convert row to [col, dtype, params, has_null, comment]
                col = row['COLUMN_NAME']
                dtype = row['DATA_TYPE']
                has_null = row['IS_NULLABLE']
                comment = ''
                if dtype.lower() in ['numeric', 'decimal']:
                    params = [int(row['NUMERIC_PRECISION']), int(row['NUMERIC_SCALE'])]
                elif dtype.lower() in ['nvarchar', 'varchar']:
                    params = [int(row['CHARACTER_MAXIMUM_LENGTH'])]
                else:
                    params = []
                result.append([col, dtype, params, has_null, comment])
        else:
            raise ValueError('Table %s does not exist.' % table)
        return result

    def _table_exists(self, conn, table):
        # TODO: conn is a dict. Check if this is desired.
        query = "IF OBJECT_ID('%s', 'U') IS NULL SELECT 0 ELSE SELECT 1" % table
        conn = self._connect(conn)
        result = pd.read_sql(query, conn).values.item() == 1
        return result

    def query(self, get_data=None, get_row_count=None, chunksize=1000, **template_vars):
        '''
        Query a database.
        `get_data`:
            SQL query for the actual data.
            Must be supplied either in config or as an argument here.
            If both are supplied, argument always overrides config.
        `get_row_count`:
            Display a progress bar if supplied in config or as an argument here.
            If both are supplied, argument always overrides config.
        `chunksize`:
            Chunk size. Default 1000.
        `template_vars`:
            Any additional keyword arguments are interpreted as template variables
            and used to replace variable appearances like `{xyz}` in the query string.
        '''
        if get_data is None:
            # left part is evaluated before the right part
            if ('query' in self.config) and ('get_data' in self.config['query']):
                get_data = self.config['query']['get_data']
            else:
                raise ValueError('`get_data` SQL query must be specified as an argument or in the config file.')

            if get_row_count is None:
                # left part is evaluated before the right part
                if ('query' in self.config) and ('get_row_count' in self.config['query']):
                    get_row_count = self.config['query']['get_row_count']
        
        conn = self._connect(self.config['conn'])
        
        if not (get_row_count is None):
            get_row_count = get_row_count.format(**template_vars)
            row_count = pd.read_sql(get_row_count, conn).values.item()
            chunk_count = np.ceil(row_count / chunksize).astype(int)

        chunks = pd.read_sql(get_data.format(**template_vars), conn, chunksize=chunksize)
        
        if not (get_row_count is None):
            chunks = tqdm(chunks, total=chunk_count)

        data = []
        for chunk in chunks:
            # fix pandas return None for NaN in object-typed columns
            obj_cols = list(chunk.select_dtypes(include=['object']).columns.values)
            chunk[obj_cols] = chunk[obj_cols].replace([None], np.nan)
            data.append(chunk)
        if len(data) > 0:
            self.data = pd.concat(data).reset_index(drop=True)
        else:
            self.data = pd.DataFrame()
        if self.data.size == 1:
            self.data = self.data.iloc[0].item()
        
        conn.close()
        
        return self

    def send_cmd(self, cmd, verbose=False):
        '''
        Send a command to the database w/o returning results.
        `cmd`: SQL query to send.
        `verbose`: verbose output. Default False.
        '''
        conn = self._connect(self.config['conn'], autocommit=False)
        crsr = conn.cursor()
        try:
            crsr.execute(cmd)
        except pyodbc.Error as err:
            conn.rollback()
            conn.close()
            raise err
        else:
            conn.commit()
        conn.close()

    def upload(self, mode='append', bcp=True, truncate=False, schema_sample=None, chunksize=1000, verbose=False):
        '''
        Upload data to database.
        `mode`:
            - `append`: append to an existing table. (Default)
            - `overwrite_data`: truncate the existing table and upload data.
            - `overwrite_table`: drop the existing table, create a new table, and upload data.
        `bcp`: use MS SQL `bcp` utilities. Default True. If False, uses pyodbc with `fast_executemany`.
        `truncate`: whether to silently truncate values that are too long to fit. (True = truncate silently; False = raise warnings)
        `schema_sample`: # of rows scanned for automatically generating the CREATE schema. Default None (scan the entire dataframe).
        `chunksize`: chunk size. Default 1000.
        `verbose`: verbose output. Default False.
        '''
        # TODO: self.config['conn'], self.config['table'] needs to refactored during API change.

        # TODO: detect df_types from df; compare to df_types from table. If df cannot fit in table: if truncate, truncate silently; else, raise errors.
        # this is to enable auto-truncate for already created tables.
        # for mode=append, overwrite_data: cast and clean df, truncate with warning unless silent truncation
        # for mode=overwrite_table: cast and clean df, truncate without warning...? or still with warning? (can we assume that df_types can always hold the df, if generated?)

        # get column type definitions and cast data (deals with float errors)
        if verbose:
            print('Determining data types.')
        
        if mode == 'append':
            df_types = self._get_table_schema(self.config['conn'], self.config['table'])
        elif mode == 'overwrite_data':
            df_types = self._get_table_schema(self.config['conn'], self.config['table'])
            if verbose:
                print('Deleting data from database.')
            self.send_cmd('TRUNCATE TABLE %s;' % self.config['table'])
        elif mode == 'overwrite_table':
            df_types = get_df_type(self.data, force_allow_null=True, sample=schema_sample, verbose=verbose)
            
            # drop old table
            if verbose:
                print('Dropping old table.')

            self.send_cmd("IF OBJECT_ID('%s', 'U') IS NOT NULL DROP TABLE %s;" % (self.config['table'], self.config['table']))

            # create new table
            schema_def_query = get_create_statement(df_types, self.config['table'])
            if verbose:
                print('Creating new table.')
                print(schema_def_query)
            self.send_cmd(schema_def_query)
        else:
            raise ValueError("mode must be one of ['append', 'overwrite_data', 'overwrite_table']")
        
        if verbose:
            print('Preprocessing data.')
        self.data = cast_and_clean_df(self.data, df_types)

        # try to connect even if we're using BCP,
        # since BCP doesn't catch connection errors
        conn = self._connect(self.config['conn'], autocommit=False)

        if bcp:
            temp_filename = 'bcp_temp_%s' % uuid.uuid4()
            if verbose:
                print('Writing to: %s' % (temp_filename + '.csv'))
            self.data.to_csv(temp_filename + '.csv', sep='⁂', index=False)

            if verbose:
                print('Uploading.')
                stdout = None
            else:
                stdout = subprocess.DEVNULL
            p = subprocess.Popen([
                'bcp',
                self.config['table'],
                'in',
                temp_filename + '.csv',
                '-c', r'-t⁂', '-k', '-E',
                '-e', temp_filename + '.err',
                '-F2',
                '-b', str(int(chunksize)),
                *self._format_host_config_args(self.config['conn']),
            ], stdout=stdout).wait()

            # clean up temp files
            if verbose:
                print('Cleaning up.')
            if os.path.exists(temp_filename + '.csv'):
                os.remove(temp_filename + '.csv')

            # delete error file if empty
            err_path = temp_filename + '.err'
            if os.path.exists(err_path):
                if os.stat(err_path).st_size == 0:
                    os.remove(err_path)
        else:
            crsr = conn.cursor()
            crsr.fast_executemany = True
            
            colnames = pd.Series(self.data.columns).apply(lambda x: '[%s]' % x)
            colnames = ','.join(list(colnames))
            blanks = ','.join(['?'] * len(self.data.columns))
            
            batch_range = range(int(np.ceil(len(self.data) / chunksize)))
            if verbose:
                batch_range = tqdm(batch_range)
            for i in batch_range:
                chunk = self.data.iloc[i*chunksize : (i+1)*chunksize].values
                sql = 'INSERT INTO %s (%s) VALUES (%s)' % (self.config['table'], colnames, blanks)
                chunk[pd.isna(chunk)] = None  # cast [pd.NA, np.nan] to None for pyodbc

                try:
                    crsr.executemany(sql, chunk.tolist())
                except pyodbc.Error as err:
                    conn.rollback()
                    conn.close()
                    raise err
                else:
                    conn.commit()
        
        conn.close()
        
    def upload_bcp(self, mode='append', verbose=False, schema_sample=None):
        warnings.warn("`sql_dataset.upload_bcp` is deprecated. Use `sql_dataset.upload(bcp=True)` instead.", DeprecationWarning)
        self.upload(mode=mode, bcp=True, verbose=verbose, schema_sample=schema_sample)
