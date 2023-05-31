## Fixtures:

* `ami1_empty.csv.gz` is an empty csv file with column names only
* `ami100_watts.csv.gz` and `ami200_watts.csv.gz` represent several days of 15-minute AMI readings. They were extracted from actual AMI time series, stripped of their IDs, and then multiplied elementwise by random values drawn from the uniform distribution [0, 2]
