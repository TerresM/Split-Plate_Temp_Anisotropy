# Split-Plate Temperature Anisotropy

This repository contains the official codebase for extracting the instantaneous proton core temperature anisotropy from split-plate Faraday Cup spectra. The method uses a gyrotropic linear-fitting approach and a hierarchical Bayesian Markov Chain Monte Carlo (MCMC) framework to derive the 3D temperature anisotropy from 1D reduced velocity distribution functions (RDFs).

The methodology and pipeline are designed and validated for the **Solar Probe Cup (SPC)** onboard NASA's **Parker Solar Probe (PSP)**, but the mathematical framework is applicable to any split-plate Faraday cup capable of measuring incident plasma flow angles (e.g., the upcoming HelioSwarm mission).

## Features
* **Bayesian MCMC Inference:** Utilizes a 5D Non-Centered Parameterization (NCP) hierarchical model to jointly evaluate the X-Z and Y-Z measurement planes.
* **Robust Outlier Handling:** Implements a heavy-tailed Student-$t$ likelihood ($\nu=4$) to accurately isolate the proton core and down-weight secondary populations (e.g., proton beams) and instrumental artifacts.
* **Automated Data Ingestion:** Seamlessly downloads and preprocesses Level 2 and Level 3 SWEAP/SPC and FIELDS data via `pySPEDAS`.
* **Physics Validation:** Outputs include asymmetric uncertainty bounds and goodness-of-fit metrics, allowing direct comparison with linear Vlasov instability thresholds.

## Installation and Dependencies

The pipeline requires Python 3.9+ and relies on the standard scientific Python stack, along with specific packages for MCMC sampling and space physics data ingestion.

```bash
pip install numpy scipy pandas xarray numba
pip install emcee corner
pip install pyspedas cdflib
```

## Usage

The primary execution script is `run_temperature_anisotropy.py`. This script sets the configuration parameters and loops over specified observation dates. 

Before running, ensure that any local calibration files (e.g., the SPC flow angle calibration IDL table) are correctly linked in your configuration or environment.

```bash
python run_temperature_anisotropy.py
```

### Configuration Options
Inside `run_temperature_anisotropy.py`, you can modify the `config` dictionary to adjust the pipeline behavior:
* `'likelihood_method'`: We strongly recommend leaving this set to `"T"` (Student-$t$) for processing raw solar wind data, as it robustly handles secondary beam populations (see Appendix C of the associated paper). `"Gauss"` (Gaussian) and `"GMM"` (Gaussian Mixture Model) are also available.
* `'fit_seconds'`: The temporal window over which RDFs are aggregated to increase statistical leverage (default is 4 seconds, roughly matching the SPAN-i cadence).
* `'flow_correction'`: Set to `True` to apply instrument-specific flow angle corrections based on the calibration lookup tables.

## Citation

If you use this codebase, the Bayesian MCMC framework, or the derived temperature anisotropy datasets in your research, please cite our submitted manuscript:

> Terres, M., Paulson, K. W., Stevens, M. L., Bharati Das, S., Niembro, T., Verniero, J. L., Badman, S. T., Rivera, Y. J., Huang, J., Klein, K. G., Fraschetti, F., Larson, D., Livi, R., Whittlesey, P., Bale, S. D., Szabo, A., Case, A. W., & Raouafi, N. E. (2026). *Extracting Proton Core Anisotropy from Split-Plate Faraday Cups: A Linear Regression Method Validated by Parker Solar Probe*. Submitted to *The Astrophysical Journal Supplement Series*.

```bibtex
@unpublished{terres2026anisotropy,
  title  = {Extracting Proton Core Anisotropy from Split-Plate Faraday Cups: A Linear Regression Method Validated by Parker Solar Probe},
  author = {Terres, Michael and Paulson, Kristoff W. and Stevens, Michael L. and {Bharati Das}, Srijan and Niembro, Tatiana and Verniero, Jaye L. and Badman, Samuel T. and Rivera, Yeimy J. and Huang, Jia and Klein, Kristopher G. and Fraschetti, Federico and Larson, Davin and Livi, Roberto and Whittlesey, Phyllis and Bale, Stuart D. and Szabo, Adam and Case, Anthony W. and Raouafi, Nour E.},
  note   = {Submitted to The Astrophysical Journal Supplement Series; preprint available upon request},
  year   = {2026}
}
```

*Note: The Zenodo DOI badge at the top of this file provides a permanent archive and citation for this specific software release.*

## Acknowledgments and References

**Funding & Support:**
This work was supported by the Parker Solar Probe mission SWEAP investigation under NASA contract NNN06AA01C and partially supported by NASA HSR Grant 80NSSC25K7759.

**Instrument Teams:**
* **SWEAP (SPC):** Data provided by the SWEAP Science Operations Center at the Smithsonian Astrophysical Observatory (SAO). (Case et al. 2020, ApJS; Kasper et al. 2016, Space Sci. Rev.)
* **FIELDS:** Data provided by the FIELDS team at the University of California, Berkeley. (Bale et al. 2016, Space Sci. Rev.)
* Special thanks to Kristoff Paulson for providing calibrated/downsampled reduced velocity distribution function routines.

**Software Libraries:**
* `emcee`: The MCMC Hammer (Foreman-Mackey et al. 2013, PASP).
* `pySPEDAS`: Python Space Physics Environment Data Analysis Software.
