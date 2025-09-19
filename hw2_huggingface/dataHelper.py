# hw2_huggingface/dataHelper.py

import os
import json
import random
from datasets import Dataset, DatasetDict, load_dataset

## * You can add more helper functions or modify function arguments if needed


def restaurant(sep_token: str):
    '''Load the ABSA restaurant dataset.'''
    
    ## TODO: the ABSA restaurant dataset.
    raise NotImplementedError(
        "The 'restaurant' function is not yet implemented.")


def laptop(sep_token: str):
    '''Load the ABSA laptop dataset.'''
    
    ## TODO: the ABSA laptop dataset.
    raise NotImplementedError("The 'laptop' function is not yet implemented.")


def acl(sep_token: str):
    '''Load the ACL-ARC dataset.'''
    
    ## TODO: the ACL-ARC dataset.
    raise NotImplementedError("The 'acl' function is not yet implemented.")


def agnews(sep_token: str):
    '''Load the AGNews dataset (test set onlys).'''
    
    ## TODO: the AGNews dataset.
    raise NotImplementedError("The 'agnews' function is not yet implemented.")


def get_fs(dataset_name: str, sep_token: str, sample_size: int):
    '''
    Get few-shot dataset. Call this function inside `get_dataset` if needed.
    dataset_name: str, the name of the dataset
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
    '''

    ## TODO: your code for preparing the few-shot dataset
    raise NotImplementedError("The 'get_fs' function is not yet implemented.")

## ! DO NOT change the function name or arguments
def get_dataset(dataset_name: str, sep_token: str) -> DatasetDict:
    '''
	dataset_name: str, the name of the dataset
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
	'''
    dataset = None

    ## TODO: your code for preparing the dataset

    if isinstance(dataset_name, str):
        ## TODO: implement for single dataset and few-shot dataset
        raise NotImplementedError(
            "The 'get_dataset' function is not yet implemented.")

    elif isinstance(dataset_name, list):
        ## TODO: implement for aggregation
        raise NotImplementedError(
            "The 'get_dataset' function for aggregation is not yet implemented."
        )

    else:
        raise ValueError("Unsupported dataset!")

    return dataset