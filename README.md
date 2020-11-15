# Learning as the Unsupervised Alignment of Conceptual Systems

## Authors
* Brett D. Roads (corresponding author, b.roads@ucl.ac.uk)
* Bradley C. Love

## System Requirements
The experiment code is written in Python. Therefore, any OS capable of running a python interpreter will suffice.
* Python >= 3.6
* The `requirements.txt` lists the python packages that are necessary to run the script.

The software has been tested on both Linux (4.13.0-36-generic) and macOS Mojave (10.14.5) with the package versions listed in `requirements.txt`.

No non-standard hardware is necessary to run the Python script.

## Installation Guide
Move the project to the desired location. In the rest of this document it is assumed that the project resides at `latest/`. After moving the project the to the desired location on your computer, you must do two things:
* Create an appropriate Python environment.
* Modify the variable `fp_base` in the python script `suite.py`.

To create the appropriate python environment using either conda or pip and install the appropriate packages specified in the `requirements.txt` file. In conda, activate your desired environment and execute `conda install --file latest/requirements.txt`. In pip, activate your desired virtual environment and execute `pip install -r latest/requirements.txt`.

To modify the variable `fp_base`, open an editor, scroll to the bottom of `suite.py` and change `fp_base` so that it corresponds to the root of the project.

The typical install time is around 10 minutes.

## Demo
By default the script `suite.py` is set to run a demo rather than the entire experiment. To run the demo, execute `python latest/python/suite.py`. The demo will generate Figures 2 and 3 of the corresponding paper. The demo run an abbreviated version of the full experiment, so Figure 3 will not look identical to the paper, but will show the same qualitative patterns. The demo takes approximately 10 minutes.

## Reproduction
To reproduce the results in the paper, change `is_demo = False` at the bottom of `suite.py`. Then execute `python latest/python/suite.py`. Note that this script is computationally intensive and will take a couple of weeks to finish. If you would like to reproduce certain components of the experiment and not others, you can change the `do_X` (where X is some string) variables in the script (e.g., `do_strength_noise = False`).
# FirstYearProject
