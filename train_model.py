import argparse
from data_modeling.model_pipeline import DataProcessor, ModelEstimator

def parse_input_arguments():
    """Parses input arguments for model training.

    Returns:
        A tuple containing the filename with data for model training, 
        the name of the target variable to model and length for test data.

    Raises:
        ValueError: If either the input filename or target variable is not provided.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input',
        dest='filename',
        action='store',
        type=str,
        default=None,
        help='Filename with data for model training.')

    parser.add_argument(
        '--quantity',
        dest='target_variable',
        action='store',
        type=str,
        default=None,
        help='Name of target variable to model.')

    args = parser.parse_args()

    if args.filename is None:
        raise ValueError('Missing input argument --input must be provided')

    if args.target_variable is None:
        raise ValueError('Missing input argument --quantity must be provided')

    return args.filename, args.target_variable


if __name__=='__main__':
    # Parsing input arguments
    filename, target_variable = parse_input_arguments()

    # Loading and preprocessing data
    data_processor = DataProcessor.from_csv(target_variable, filename)
    X, y = data_processor.process_data()

    # Fitting model
    model = ModelEstimator().fit(X, y)
    predicted = model.predict(X)

    model.fitted_model_plot(y) 
    model.evaluate(y,X[target_variable+'_l1'])
