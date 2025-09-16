# hw2_huggingface/test_datahelper.py

import pytest
from .dataHelper import get_dataset, get_fs


def test_restaurant():
    '''
    Test the restaurant dataset loading function.
    '''
    raw_dataset = get_dataset("restaurant_sup", sep_token="<SEP>")
    train, test = raw_dataset["train"], raw_dataset["test"]
    assert len(train) > 0 and len(test) > 0
    assert "text" in train.column_names and "label" in train.column_names
    assert "text" in test.column_names and "label" in test.column_names
    assert set(train["label"]).issubset(set(range(3)))
    assert set(test["label"]).issubset(set(range(3)))


def test_laptop():
    '''
    Test the laptop dataset loading function.
    '''
    raw_dataset = get_dataset("laptop_sup", sep_token="<SEP>")
    train, test = raw_dataset["train"], raw_dataset["test"]
    assert len(train) > 0 and len(test) > 0
    assert "text" in train.column_names and "label" in train.column_names
    assert "text" in test.column_names and "label" in test.column_names
    assert set(train["label"]).issubset(set(range(3)))
    assert set(test["label"]).issubset(set(range(3)))


def test_acl():
    '''
    Test the ACL-ARC dataset loading function.
    '''
    raw_dataset = get_dataset("acl_sup", sep_token="<SEP>")
    train, test = raw_dataset["train"], raw_dataset["test"]
    assert len(train) > 0 and len(test) > 0
    assert "text" in train.column_names and "label" in train.column_names
    assert "text" in test.column_names and "label" in test.column_names
    assert set(train["label"]).issubset(set(range(6)))
    assert set(test["label"]).issubset(set(range(6)))


def test_agnews():
    '''
    Test the AGNews dataset loading function.
    '''
    raw_dataset = get_dataset("agnews_sup", sep_token="<SEP>")
    train, test = raw_dataset["train"], raw_dataset["test"]
    assert len(train) > 0 and len(test) > 0
    assert "text" in train.column_names and "label" in train.column_names
    assert "text" in test.column_names and "label" in test.column_names
    assert set(train["label"]).issubset(set(range(4)))
    assert set(test["label"]).issubset(set(range(4)))


def test_restaurant_fs():
    '''
    Test the few-shot restaurant dataset loading function.
    '''
    raw_dataset = get_fs("restaurant", sep_token="<SEP>", sample_size=32)
    train, test = raw_dataset["train"], raw_dataset["test"]
    assert len(train) == 32 and len(test) == 32
    assert "text" in train.column_names and "label" in train.column_names
    assert "text" in test.column_names and "label" in test.column_names
    assert set(train["label"]).issubset(set(range(3)))
    assert set(test["label"]).issubset(set(range(3)))


def test_laptop_fs():
    '''
    Test the few-shot laptop dataset loading function.
    '''
    raw_dataset = get_fs("laptop", sep_token="<SEP>", sample_size=32)
    train, test = raw_dataset["train"], raw_dataset["test"]
    assert len(train) == 32 and len(test) == 32
    assert "text" in train.column_names and "label" in train.column_names
    assert "text" in test.column_names and "label" in test.column_names
    assert set(train["label"]).issubset(set(range(3)))
    assert set(test["label"]).issubset(set(range(3)))


def test_acl_fs():
    '''
    Test the few-shot ACL-ARC dataset loading function.
    '''
    raw_dataset = get_fs("acl_fs", sep_token="<SEP>", sample_size=32)
    train, test = raw_dataset["train"], raw_dataset["test"]
    assert len(train) == 32 and len(test) == 32
    assert "text" in train.column_names and "label" in train.column_names
    assert "text" in test.column_names and "label" in test.column_names
    assert set(train["label"]).issubset(set(range(6)))
    assert set(test["label"]).issubset(set(range(6)))


def test_agnews_fs():
    '''
    Test the few-shot AGNews dataset loading function.
    '''
    raw_dataset = get_fs("agnews", sep_token="<SEP>", sample_size=32)
    train, test = raw_dataset["train"], raw_dataset["test"]
    assert len(train) == 32 and len(test) == 32
    assert "text" in train.column_names and "label" in train.column_names
    assert "text" in test.column_names and "label" in test.column_names
    assert set(train["label"]).issubset(set(range(4)))
    assert set(test["label"]).issubset(set(range(4)))


def test_aggregation_1():
    raw_dataset = get_dataset(["restaurant_sup", "laptop_sup"],
                              sep_token="<SEP>")
    train, test = raw_dataset["train"], raw_dataset["test"]
    assert len(train) > 0 and len(test) > 0
    assert "text" in train.column_names and "label" in train.column_names
    assert "text" in test.column_names and "label" in test.column_names
    assert set(train["label"]).issubset(set(range(6)))
    assert set(test["label"]).issubset(set(range(6)))


def test_aggregation_2():
    raw_dataset = get_dataset(["acl_sup", "laptop_sup"], sep_token="<SEP>")
    train, test = raw_dataset["train"], raw_dataset["test"]
    assert len(train) > 0 and len(test) > 0
    assert "text" in train.column_names and "label" in train.column_names
    assert "text" in test.column_names and "label" in test.column_names
    assert set(train["label"]).issubset(set(range(10)))
    assert set(test["label"]).issubset(set(range(10)))


def test_aggregation_3():
    raw_dataset = get_dataset(
        ["restaurant_fs", "laptop_fs", "acl_fs", "agnews_fs"],
        sep_token="<SEP>")
    train, test = raw_dataset["train"], raw_dataset["test"]
    assert len(train) > 0 and len(test) > 0
    assert "text" in train.column_names and "label" in train.column_names
    assert "text" in test.column_names and "label" in test.column_names
    assert set(train["label"]).issubset(set(range(16)))
    assert set(test["label"]).issubset(set(range(16)))
