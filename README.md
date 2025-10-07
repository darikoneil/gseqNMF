# gseqNMF

This package is a re-implementation of the seqNMF algorithm 
described in [Mackevicius et al., 2019](https://elifesciences.org/articles/38471), 
and provided as a MATLAB toolbox [here](https://github.com/FeeLab/seqNMF). It utilizes
standard sklearn syntax for easy plug-and-play usage.

### Features
- Compatible with scikit-learn pipelines
- Significant performance optimizations (benchmarks pending)
- Drop-in GPU acceleration via CuPy
- Comprehensive test suite
- Linted with fully-typed codebase
- Optional visualization module

### Installation
The package is available on PyPI and can be installed via pip.

```
pip install gseqnmf
```

GPU acceleration can be enabled by installing the package with the `cuda12` extra.
Development dependencies can be installed with the `dev` extra.

```
pip install gseqnmf[cuda12,dev]
```

### Usage Example

```python
import numpy as np
from gseqnmf import GseqNMF

# Load synthetic dataset (samples x neurons)
data = np.load("your_data.npy")
n_components = 3
seqeuence_length = 50
lam = 5e-2
model = GseqNMF(
    n_components=n_components,
    sequence_length=seqeuence_length,
    lam=lam,
)
model.fit(data)
```

### License

This project is licensed under the terms of the MIT license.
