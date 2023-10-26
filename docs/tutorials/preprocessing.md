# Preprocessing

## fMRI

We use a custom but fast pipeline to minimally pre-process resting-state fMRI images. Preprocessing steps include:

- register_to_t1w
- clean_signal
- register_to_template

It requires ANTsPyX, which can be installed using spack on AWS/ULHPC cluster:

```bash
spack install antspyx
spack load antspyx
```
