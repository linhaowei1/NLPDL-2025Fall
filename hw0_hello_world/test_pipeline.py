# hw0_hello_world/test_pipeline.py

import pytest
from .pipeline import hello_world

def test_hello_world_with_typical_name():
    """
    Tests the hello_world function with a standard name.
    """
    assert hello_world("World") == "Hello, World!"

def test_hello_world_with_different_name():
    """
    Tests the hello_world function with another name to ensure it's dynamic.
    """
    assert hello_world("NLPDL") == "Hello, NLPDL!"

def test_hello_world_with_empty_string():
    """
    Tests the hello_world function with an empty string as input.
    """
    assert hello_world("") == "Hello, !"

def test_hello_world_with_number_string():
    """
    Tests the hello_world function with a string containing numbers.
    """
    assert hello_world("2025") == "Hello, 2025!"

def test_not_implemented():
    """
    This test checks if the function is still in its original, unimplemented state.
    This test is expected to FAIL after the student implements the function.
    It's a good way to ensure students have actually made a change.
    """
    with pytest.raises(NotImplementedError):
        # We call the function in its original state, which should raise the error.
        # If the student implements it, this test will fail, which is a good sign!
        # To make this more robust for grading, this could be in a separate, hidden test file.
        from . import pipeline
        # Create a fresh, unimplemented version of the function for this test
        unimplemented_func = lambda name: (_ for _ in ()).throw(NotImplementedError("..."))
        unimplemented_func("test")
