# Dataset Utilities (Supports MS SQL)
![](https://github.com/jeffacce/dataset_utilities/workflows/Tests/badge.svg)
*This utility snippet is under active development. To report a bug, please open an issue.*

A minimalistic encapsulation of `pandas`, `pyodbc`, and `bcp` for in-memory dataset ETL workflows, and automatic upload to MS SQL Server, encouraging organizing everything into separate, modular files. Your script can look like this:
```python
from dataset import sql_dataset
my_dataset = sql_dataset('config.yml')
df = my_dataset.query().transform().data

uploader = sql_dataset('upload_target.yml')
uploader.data = df
uploader.upload(mode='overwrite_table', bcp=True, verbose=True)
```

Right now it supports MS SQL only. If `BCP` is not installed, specify `bcp=False` for uploading.

## Features
- Organize datasets as separate *config files* and *transform scripts*
  - Specify read/write file path
  - Specify SQL host config and SQL queries
  - Specify transform functions naturally like imports, e.g.
    - From an existing package: `numpy.abs`
    - From a custom script at `./etl_scripts/transforms.py/` containing the transform function `clean`: `etl_scripts.transforms.clean`
  - Specify upload table name for bulk inserting data directly from Pandas DataFrame
- Query SQL databases with automatic connection test/retry logic
- (Experimental) Automatically generate storage-optimized DDL (`CREATE TABLE` statements) from Pandas DataFrame
  - Supports `decimal`, `nvarchar`, `date`, `datetime`, and integer types
  - Detects type and necessary storage space (e.g. precision, scale, max chars, integer types) from Pandas DataFrame
- Convenient `upload` wrapper

## Installation
Just copy the snippet `dataset.py` and import it into your script. Package requirements are listed in `requirements.txt`. Requires `pandas`, `numpy`, `tqdm`, `requests`, `pyodbc`, `PyYAML`.

## Getting Started: Example ETL Workflow
Imagine we need to summarize some sales numbers.

We have a remote SQL server at `sql-server.acme.com`, with a database called `sales`. The database has a table called `half_year_sales` with the following entries:

| Company | H1 | H2 |
|---------|----|----|
| A       | 1  | 2  |
| B       | 3  | 4  |
| C       | 5  | 6  |

Suppose we would like to:
1. (Extract) Read the data from the remote SQL server
2. (Transform) Add the half-year columns together to get the full year sales
3. (Save) Write it into a local file, and
4. (Upload) Upload it back to the remote server as a new table.

### Main program
We could organize our working directory like this, with data, transforms, and config grouped in different folders:
```
.
├── main.py
├── data
│   └── my_dataset.csv
├── transforms
│   └── my_dataset.py
├── config
│   └── my_dataset.yml
└── dataset.py
```

In `main.py`, we just specify the dataset config file and tell the dataset what we would like to do:
```python
# ./main.py
from dataset import sql_dataset

if __name__ == '__main__':
    my_dataset = sql_dataset(
        config_file='./config/my_dataset.yml',
    )
    my_dataset.query().transform().write().upload(mode='overwrite_table', bcp=True, verbose=True)

    df = my_dataset.data
    # do additional things with the raw dataframe
```

### Transform function
Write a transform function that takes in the original dataframe and returns a transformed dataframe:

In this example, we'd like to calculate the full year sales for each company by adding the two halves.
```python
# ./transforms/my_dataset.py
def calc_full_year_sales(df):
    df['full_year'] = df['H1'] + df['H2']
    df = df.drop(columns=['H1', 'H2'])
    return df
```

### Config
Now, we need to write necessary information into a config file `my_dataset.yml`.

#### Connection
The `conn` (connection) parameters are passed to `pyodbc` to create a connection.
```yaml
# ./config/my_dataset.yml
conn:
    server: sql-server.acme.com
    database: sales
    user: acmeUser
    password: acmeUserPassword
    port: 1433
    driver: 'ODBC Driver 17 for SQL Server'
```

#### Query strings
Write SQL queries into the config file:

`get_data` is required. `get_row_count` is optional; if supplied, the query will display a progress bar.
```yaml
# ./config/my_dataset.yml
query:
    get_data: "
        SELECT * FROM [half_year_sales];
    "
    get_row_count: "
        SELECT COUNT(*) FROM [half_year_sales];
    "
```

#### Filepath
Set the default path for local file reading/writing.
```yaml
# ./config/my_dataset.yml
filepath: ./data/my_dataset.csv
```

#### Table name
Set the upload table name. If you specified `'overwrite_table'` in the main program, the script will look at your dataframe column by column, determine the data type and the variable length required, and send a `CREATE TABLE` command to the SQL server to create your table.
```yaml
# ./config/my_dataset.yml
table: full_year_sales
```

#### Transform function
If we put this script at `./transforms/my_dataset.py`, we can specify the transform function as:
```yaml
transform: transforms.my_dataset.calc_full_year_sales
```
Note that this import location is written from the perspective of the calling script in the root folder `./main.py`.

Finally, we have a full configuration file:
```yaml
# ./config/my_dataset.yml
conn:
    server: sql-server.acme.com
    database: sales
    user: acmeUser
    password: acmeUserPassword
    port: 1433
    driver: 'ODBC Driver 17 for SQL Server'

query:
    get_data: "
        SELECT * FROM [half_year_sales];
    "
    get_row_count: "
        SELECT COUNT(*) FROM [half_year_sales];
    "

filepath: ./data/my_dataset.csv

table: full_year_sales

transform: transforms.my_dataset.calc_full_year_sales
```

That's it!

If you run `main.py` now, the script will query the SQL database, transform the data using the function specified, save the transformed data to `./data/my_dataset.csv`, automatically create a table in the database called `full_year_sales` and upload the transformed dataframe there.
