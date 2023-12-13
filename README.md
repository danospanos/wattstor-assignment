# Simple Model Training and In-Sample Evaluation 
This project contains a Python script (train_model.py) for training a crossvalidated Lasso model. The script outputs results as PNG plots and displays evaluation metrics in the terminal.

## Prerequisites

 - Python 3.10.0
 - Pipenv (recommended for creating a virtual environment)
 
## Installation

 1) Clone this repository to your local machine.
 2) Setting up the virtual environment.
You can initialize using pipenv, in the project directory with:

> pipenv --python 3.10.0

> pipenv install -r requirements.txt

 3) Set the Python path
Make sure the module is on your Python path for proper loading.

## Example Usage

In you virtual environment run:

> python3 train_model.py --input SG.csv --quantity Consumption

## Output

 - PNG Plot: The script will save PNG plot in the project directory.
 - Terminal Output: Evaluation metrics and other outputs will be displayed in the terminal.

## Running Tests

Navigate to the test directory and run the tests using:

> python3 unittests.py
