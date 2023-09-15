
On AWS/ULHPC use this:

```
aws s3 sync --no-sign-request s3://openneuro.org/<DS_NUMBER> <DS_NUMBER>
```

and then to track the dataset with DVC:

```
dvc add <DS_NUMBER>
```

