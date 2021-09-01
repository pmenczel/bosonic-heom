# bosonic-heom

This package facilitates the simulation of the Hierachical Equations of Motion (HEOM) for open quantum systems coupled to bosonic environments.
It is based on [QuTiP](https://github.com/qutip/qutip), the quantum toolbox for python.

The Hierarchical Equations of Motion are explained, for example, in Ref. [1].
They can be used to simulate Caldeira-Leggett open quantum systems if the auto-correlation functions of the environments can be written, or approximated, as sums of exponentials.
The package contains helper methods to deal with environments that have the spectral density of underdamped Brownian motion, and to approximate correlation functions using multi-exponential fits.

Once the correlation functions are defined, it is straightforward to construct and simulate the HEOM.
Consult [sample.ipynb](https://github.com/pmenczel/bosonic-heom/blob/main/sample.ipynb) for complete examples with more detailed explanations.

[1] [Kato and Tanimura, J. Chem. Phys. (2016)](https://aip.scitation.org/doi/10.1063/1.4971370)