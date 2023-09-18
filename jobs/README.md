# Jobs

This directory contains the job definitions for the various jobs that are run on HPC. The jobs are defined using the ([SLURM](https://hpc-wiki.info/hpc/SLURM) job scheduler.


## Naming convention

Job description files are named according to the following convention:

```
<command>_<operation>_<parameters>.sh
```

where:

* `<command>` is the name of the command that is run (e.g., `s3`, `aws`, `datalad`, etc),
* `<operation>` is the operation that is performed by the command (e.g., `download`), and
* `<parameters>` is a list of parameters that are passed to the command.
