# Package management

We use [spack](https://spack.io/) to manage software packages. Spack is a package manager for supercomputers, Linux, and macOS. It makes installing scientific software easy. With spack, you can build a package with multiple versions, configurations, platforms, and compilers, and all of these builds can coexist on the same machine.


## Installation

```bash
git clone https://github.com/spack/spack.git
. spack/share/spack/setup-env.sh
```

Then:


```
spack env create -d .
spack env activate .
```

