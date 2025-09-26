# hw2_huggingface/test_trainscript.py

import pytest
import numpy as np
from .train import load_data, get_model, compute_metrics


def test_load_data_restaurant():
    '''
    Test loading the restaurant dataset.
    '''
    raw_dataset, num_labels = load_data("restaurant_sup")
    assert raw_dataset is not None
    assert num_labels == 3


def test_load_data_acl():
    '''
    Test loading the ACL-ARC dataset.
    '''
    raw_dataset, num_labels = load_data("acl_sup")
    assert raw_dataset is not None
    assert num_labels == 6


def test_load_data_agnews():
    '''
    Test loading the AGNews dataset.
    '''
    raw_dataset, num_labels = load_data("agnews_sup")
    assert raw_dataset is not None
    assert num_labels == 4


def test_get_model_bert():
    '''
    Test getting a BERT model and tokenizer.
    '''
    tokenizer, model = get_model("bert-base-uncased", 3)
    assert tokenizer is not None
    assert model is not None

    # Check model configuration
    assert hasattr(model.config, "model_type")
    assert model.config.model_type == "bert"
    assert model.config.num_labels == 3

    # Check tokenizer and model vocab size consistency
    assert model.config.vocab_size == tokenizer.vocab_size

    # Check pad_token settings
    assert tokenizer.pad_token is not None
    assert tokenizer.pad_token_id is not None
    assert model.config.pad_token_id == tokenizer.pad_token_id

def test_compute_metrics():
    '''
    Test the compute_metrics function with dummy predictions and labels.
    '''

    class DummyPredictions:

        def __init__(self, preds, labels):
            self.predictions = preds
            self.label_ids = labels

    preds = [[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0],
             [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]]
    labels = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2]

    dummy_preds = DummyPredictions(preds=np.array(preds),
                                   labels=np.array(labels))
    metrics = compute_metrics(dummy_preds)
    
    # Check expected values (these are pre-calculated for the dummy data)
    expected_accuracy = 0.5
    expected_macro_f1 = 0.4666
    expected_micro_f1 = 0.5
    expected_weighted_f1 = 0.5467
    print(metrics)
    assert abs(metrics["accuracy"] - expected_accuracy) < 1e-3
    assert abs(metrics["macro_f1"] - expected_macro_f1) < 1e-3
    assert abs(metrics["micro_f1"] - expected_micro_f1) < 1e-3
    assert abs(metrics["weighted_f1"] - expected_weighted_f1) < 1e-3
