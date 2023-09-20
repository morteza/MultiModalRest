# TODO How-to: install OpenNeuro dataset using Datalad
1. instsll and configure spack
3. create a new local repo for the spack
4. add git-annex package and properly modify the file (as of 2023-0

url: https://downloads.kitenet.net/git-annex/linux/current/git-annex-standalone-arm64.tar.gz

6. spack install py-datalad
7. spack load py-datalad
8. install datalad openneuro super dataset (recursively) -> do not use `get`

Codes:

add to spack repos/custom/packages/git-annex/package.py
```
    if platform.system() == "Linux" and platform.machine() == "aarch64":
        # git-annex-standalone-arm64.tar.gz
        version(
            "10.20230828",
            sha256="ba7fd3ad7aad28ce93965234034094c4c950eb30237cb36adb7956c1d83de429",
            url="https://downloads.kitenet.net/git-annex/linux/current/git-annex-standalone-arm64.tar.gz",
        )
```
```
cd %PROJ_DIR/data/ && datalad install -r https://datasets.datalad.org/openneuro
```
