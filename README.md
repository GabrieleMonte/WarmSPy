# WarmSPy (Warm Inflation Scalar Perturbations)

WarmSPy release 1.0 (2023). This code has been written by Gabriele Montefalcone, Vikas Aragam and Dr. Luca Visinelli.

WarmSPy (Warm Scalar Perturbations) is a Python code that provides the solution for the perturbation equations in warm inflationary models and calculates the corresponding scalar power spectrum at the Cosmic Microwave Background (CMB) horizon crossing.
The code also allows to numerically fit for the so called Scalar Dissipation Function G(Q) which defines the enhancement (suppression) induced by a positive (negative) temperature dependence in the dissipation rate. Overall, the code is designed to 
facilitate the analysis of warm inflation scenarios and help in precisely testing specific inflationary models against current and future constraints from CMB data and beyond.
For more details, please consult our research paper (link below).

## Features

- Solves the perturbation equations in warm inflationary models
- Calculates the scalar power spectrum at CMB horizon crossing
- Performs numerical fits to the scalar dissipation function G(Q)
- Provides example notebooks to illustrate code usage
- Includes figures and data files from the associated research paper

## Installation

To use WarmSPy, follow these steps:

1. Clone the GitHub repository using the following command:
```sh
git clone https://github.com/your-username/WarmSPy.git
```

2. Make sure you have Python installed on your system. WarmSPy is compatible with Python 3.

3. Install the required dependencies. WarmSPy relies on the following Python packages: numpy,scipy, matplotlib, os, multiprocessing.The simplest way to install these dependencies is by using pip, e.g.:
```sh
pip install numpy
```

## Usage
WarmSPy provides example notebooks to help you understand how to use the code effectively. These notebooks demonstrate the various functionalities of WarmSPy and guide you through different use cases. You can find these example notebooks in the repository.

Additionally, the repository includes figures and data files that were used in the associated research paper. These resources can be used to replicate the results or further analyze the outputs of the WarmSPy code.

## Contributions
Contributions to WarmSPy are welcome! If you find any issues or have suggestions for improvements, please create an issue on the GitHub repository. You can also submit pull requests to contribute code enhancements.

## Acknowledgments

We would like to acknowledge the Texas Advanced Computing Center (TACC) for providing computational resources and support. The program was tested and all calculations were computed on the TACC infrastructure.

If you use WarmSPy in your research work, please consider citing our paper:

- [Universality of cosmological perturbations in warm inflation](https://doi.org/xxx/xxx-xxx)

## Contact
For any inquiries or questions regarding WarmSPy, please contact montefalcone@utexas.edu .
