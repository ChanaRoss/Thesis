from datasets.MaxFlowDataset import get_max_flow_dataset

"""
the get function should return a train and validation dataset and take
any number of arguements, providing they are listed in the "dataset_params"
section of the config file.
"""

dataset_registry = {
    'max_flow': get_max_flow_dataset
}