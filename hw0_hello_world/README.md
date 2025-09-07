# Homework 0: Hello World

**Objective:** The goal of this assignment is to ensure your development environment is set up correctly and to familiarize you with the structure of our homework assignments and the testing process.

## Task Description

Your task is to implement the `hello_world()` function in the `pipeline.py` file. This function should take a single argument, `name`, and return a greeting string in the format `"Hello, {name}!"`.

For example, if the input `name` is `"World"`, the function should return `"Hello, World!"`.

## Instructions

1. **Locate the file:** Open the `hw0_hello_world/pipeline.py` file.
2. **Find the function:** Inside this file, you will find the `hello_world()` function.
3. **Implement the logic:** Modify the function to return the correct greeting string. You will see a `NotImplementedError`; you should remove that line and replace it with your implementation.

## How to Test Your Code

Once you have implemented the function, you can (and should!) test it locally to ensure it works as expected.

Navigate to the **root directory** of the repository and run the following command:

```
make test-hw0
```

Alternatively, you can run `pytest` directly on this directory:

```
pytest hw0_hello_world/
```

If your implementation is correct, you will see a message indicating that all tests have passed. If there are any failures, the output will help you diagnose what might be wrong with your code.