import torch
from torch.utils.data import TensorDataset


def load_dataset(dataset_path, mean_subtraction, normalization):
    """
    Reads the train and validation data

    Arguments
    ---------
    dataset_path: (string) representing the file path of the dataset
    mean_subtraction: (boolean) specifies whether to do mean centering or not. Default: False
    normalization: (boolean) specifies whether to normalizes the data or not. Default: False

    Returns
    -------
    train_ds (TensorDataset): The features and their corresponding labels bundled as a dataset
    """
    # Load the dataset and extract the features and the labels    
    file = torch.load(dataset_path)
    features = file['features']
    labels = file['labels']

    # Do mean_subtraction if it is enabled
    if mean_subtraction:
        mean_per_feature = torch.mean(features, axis=0)         # mean per column
        features = torch.subtract(features, mean_per_feature)   # subtracts training set mean from each example in the set

    # do normalization if it is enabled
    if normalization:
        std_per_feature = torch.std(features, axis=0)           # std per column
        # if std = 0 for a feature, normalization step for that feature has to be skipped, i.e. divide feature by 1
        std_per_feature[std_per_feature == 0] = 1              
        features = torch.divide(features, std_per_feature)      # divide training set by the per-feature standard devations

    # create tensor dataset train_ds
    train_ds = TensorDataset(features, labels)
    return train_ds

# load_dataset("iris_dataset.pt", True, True)