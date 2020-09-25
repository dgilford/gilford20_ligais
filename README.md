# Code Supporting "Could the Last Interglacial Constrain Projections of Future Antarctic Ice Mass Loss and Sea-level Rise?" by Gilford et al. (2020)

This repository is the Python code base for [Gilford et al. 2020](https://www.essoar.org/doi/10.1002/essoar.10501078.3):  "Could the Last Interglacial Constrain Projections of Future Antarctic Ice Mass Loss and Sea-level Rise?"
Gilford et al. (2020), published in *JGR-Earth Surface* (updated September 2020).

**Project abstract:**

> Previous studies have interpreted Last Interglacial (LIG; ~129--116 ka) sea-level estimates in multiple different ways to calibrate projections of future Antarctic ice-sheet (AIS) mass loss and associated sea-level rise. This study systematically explores the extent to which LIG constraints could inform future Antarctic contributions to sea-level rise. We develop a Gaussian process emulator of an ice-sheet model to produce continuous probabilistic projections of Antarctic sea-level contributions over the LIG and a future high-emissions scenario. We use a Bayesian approach conditioning emulator projections on a set of LIG constraints to find associated likelihoods of model parameterizations. LIG estimates inform both the probability of past and future ice-sheet instabilities and projections of future sea-level rise through 2150. Although best-available LIG estimates do not meaningfully constrain Antarctic mass loss projections or physical processes until 2060, they become increasingly informative over the next 130 years. Uncertainties of up to 50 cm remain in future projections even if LIG Antarctic mass loss is precisely known (+/-5 cm), indicating there is a limit to how informative the LIG could be for ice-sheet model future projections. The efficacy of LIG constraints on Antarctic mass loss also depends on assumptions about the Greenland ice sheet and LIG sea-level chronology. However, improved field measurements and understanding of LIG sea levels still have potential to improve future sea-level projections, highlighting the importance of continued observational efforts.

If you have any questions, comments, or feedback on this work or code, please [contact Daniel](mailto:daniel.gilford@rutgers.edu) or open an [Issue](https://github.com/Rutgers-ESSP/lig_constraints/issues) in the repository.

## Citation

A [preprint](https://www.essoar.org/doi/10.1002/essoar.10501078.3) of Gilford et al. (2020) is available at ESSOAr.
If you use any or all of this code in your work, please include the citation:
```
Daniel M. Gilford, Erica L. Ashe, Robert M. DeConto, Robert E. Kopp, David Pollard, Alessio Rovere, 2020: 
Could the Last Interglacial Constrain Projections of Future Antarctic Ice Mass Loss and Sea-level Rise? 
JGR-Earth Surface (accepted September 2020).
```

## Data

The ice-sheet data used to construct the emulator used in this work has been archived at Zenodo with the doi:

<a href="https://doi.org/10.5281/zenodo.3478486"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.3478486.svg" alt="DOI"></a>

## Getting Started

This codebase requires **Python version 3.5+** to run. It was written and tested with Python 3.5.2 and Jupyter Notebook 5.4.0.
To get it up and running on your system, clone the repository and ensure that you have the [required dependencies](requirements.txt) and Jupyter Notebook.

### Dependencies

* [Pandas](https://pandas.pydata.org/) 0.19.2
* [GPflow](https://gpflow.readthedocs.io/en/stable/) 1.3.0
* [Matplotlib](https://matplotlib.org/) 1.5.3
* [NumPy](https://numpy.org/) 1.16.0
* [TensorFlow](https://www.tensorflow.org/) 1.12.1
* [SciPy](https://www.scipy.org/) 1.4.1

## File Descriptions

### Analysis and Emulator:

* **[lig_utilities.py](./lig_utilities.py)** - Utility file containing functions for emulation and analysis.
* **[construct_lig_emulator.ipynb](./construct_lig_emulator.ipynb)** - Notebook creating the Last Interglacial emulator optimized+conditioned on the training data.
* **[construct_rcp_emulator.ipynb](./construct_rcp_emulator.ipynb)** - Notebook creating the RCP8.5 emulator optimized+conditioned on the training data.

### Data:

* **[lig_data_dec18.pk1](./data/lig_data_dec18.pk1)** - Last Interglacial Ensemble of 196 PSU3Dice simulations developed by Rob DeConto in December 2018.
* **[rcp85_data_sept18.pk1](./data/lig_data_dec18.pk1)** - Representative Concentration Pathway 8.5 Ensemble of 196 PSU3Dice simulations developed by Rob DeConto in September 2018.

### Figures (see manuscript for full captions):

#### Main Text (see manuscript for full captions):

* **[Figure 1a](./figures/Fig1a.pdf)** - Timeseries of LIG ice-sheet model simulations (0-5ka), color-coded by CLIFVMAX
* **[Figure 1b](./figures/Fig1b.pdf)** - Timeseries of RCP8.5 ice-sheet model simulations (2000-2150), color-coded by CLIFVMAX
* **[Figure 2a](./figures/Fig2a.pdf)** - LIG ice-sheet model simulations and emulator output across the ice-sheet model parameter space

#### Supplemental (see manuscript supplement for full captions):

* **[Figure S1a](./figures/FigS1a.pdf)** - Timeseries of LIG ice-sheet model simulations (0-5ka), color-coded by CREVLIQ
* **[Figure S1b](./figures/FigS1b.pdf)** - Timeseries of RCP8.5 ice-sheet model simulations (2000-2150), color-coded by CREVLIQ
* **[Figure S2](./figures/FigS2.pdf)** - Contours of LIG and RCP8.5 (2100) simulations across the ice-sheet model parameter space
* **[Figure S7](./figures/FigS7.pdf)** - LIG emulator variance across the ice-sheet model parameter space

## Authors

### Contributors
* **Daniel M. Gilford, PhD** - *Primary Author, Development & Maintenance* - [GitHub](https://github.com/dgilford)
* **Erica Ashe, PhD** - *Co-author, Bayesian Statistics* - [GitHub](https://github.com/ericaashe)

### Co-authors
* **Prof. Bob Kopp**
* **Prof. Rob DeConto**
* **Prof. David Pollard**
* **Prof. Alessio Rovere**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

We thank Andrea Dutton, Anna Ruth Halberstadt, Jacky Austermann, three anonymous reviewers, and the editors for helpful comments which improved this manuscript. DG, RK, RD, and DP were supported by NSF Grant ICER-1663807 and NASA Grant 80NSSC17K0698. EA and RK were supported by NSF Grant OCE-1702587. AR is supported by the European Research Council, under the European Union's Horizon 2020 research and innovation programme (grant agreement n. 802414). This project grew, in part, out of discussions 
at the [2018 PALeo constraints on SEA level rise (PALSEA) annual meeting](https://palseagroup.weebly.com/2018-meeting.html). PALSEA is a working group of the International Union for Quaternary Sciences (INQUA) and Past Global Changes (PAGES), which in turn received support from the Swiss Academy of Sciences and the Chinese Academy of Sciences.
GP regression was performed with GPflow ([Matthews et al., 2017](http://www.jmlr.org/papers/volume18/16-537/16-537.pdf)); some color maps provided by [Crameri (2018)](http://www.fabiocrameri.ch/colourmaps.php).