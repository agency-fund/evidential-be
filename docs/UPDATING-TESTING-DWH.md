# Testing Dataset for Testing DWH

## What is the testing DWH?

The testing DWH is a sample datasource we use for unit tests and for manually verification of functionality. We store
the dataset in a compressed CSV form in GitHub, and various tools load this dataset into various databases.

There are two important pieces:

1. The data itself (in the .csv.zst file).
1. The DDL statements used to create the tables.

## Modifying the Testing DWH Data

You can use [duckdb](https://duckdb.org/) to modify the testing data warehouse dataset.

Here's an example script:

```sql
CREATE TABLE testing_dwh AS
SELECT *
FROM 'src/xngin/apiserver/testdata/testing_dwh.csv.zst';

DESCRIBE testing_dwh;

ALTER TABLE testing_dwh
    ADD sample_date DATE
        DEFAULT NOW() + CAST((90 * random()) AS integer) * INTERVAL '1 days';
ALTER TABLE testing_dwh
    ADD sample_timestamp DATETIME
        DEFAULT DATETRUNC('second',
                          NOW() - CAST((525600 * random()) as integer) * interval '1 minute'
                              + CAST ((60*random()) as integer) * interval '1 second');
ALTER TABLE testing_dwh
    ADD timestamp_with_tz TIMESTAMPTZ
        DEFAULT DATETRUNC('second',
                          NOW() - CAST((525600 * random()) as integer) * interval '1 minute'
                              + CAST ((60*random()) as integer) * interval '1 second');

COPY testing_dwh (
    gender,
    ethnicity,
    first_name,
    last_name,
    id,
    income,
    potential_0,
    potential_1,
    is_recruited,
    is_registered,
    is_onboarded,
    is_engaged,
    is_retained,
    baseline_income,
    current_income,
    uuid_filter,
    sample_date,
    sample_timestamp,
    timestamp_with_tz)
    TO 'src/xngin/apiserver/testdata/testing_dwh.csv.zst';
```

You can recompress the file at a greater compression level:

```shell
ls -lh src/xngin/apiserver/testdata/testing_dwh.csv.zst
zstd -d -c src/xngin/apiserver/testdata/testing_dwh.csv.zst | zstd -9 > /tmp/smaller.tgz
mv /tmp/smaller.tgz src/xngin/apiserver/testdata/testing_dwh.csv.zst
ls -lh src/xngin/apiserver/testdata/testing_dwh.csv.zst
```

If you add or modify columns, be sure to also update
[testing_dwh_def.py](../src/xngin/apiserver/testing/testing_dwh_def.py) to reflect any changes. A
small unittest (`test_admin.py::test_user_from_token`) checks that names and type mappings agree.

## Modifying the DDL

Different databases have different syntax for table creation. Variants may use different type names or support different
types. To handle these gracefully, we also keep distinct DDL per testing backend.

You can find them in [the \*.ddl files](../src/xngin/apiserver/testdata). These files inform
`xngin-cli create-testing-dwh` when creating the tables.
