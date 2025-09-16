# Homework 2: Huggingface & PEFT

**Objective:** This assignment is designed to help you study Huggingface🤗, which is the most popular tool for NLP researchers. Through this assignment, you will be able to write codes easily for nearly all the NLP tasks and become a real NLPer!

## Task 1: Build Your Dataset

[🤗Datasets](https://huggingface.co/docs/datasets/index) is a library for easily accessing and sharing datasets, and evaluation metrics for Natural Language Processing (NLP), computer vision, and audio tasks. You have to read the docs to learn how to use this library and finish the tasks below. In this task, you should hand in such a `dataHelper.py` file. The requirements are as follows.

```python
from dataHelper import get_dataset
```

1. It should contain a get_dataset() function like this.

```python
def get_dataset(dataset_name, sep_token):
	'''
	dataset_name: str, the name of the dataset
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
	'''
	dataset = None

	# your code for preparing the dataset...

	return dataset
```

2. Support a bunch of datasets for classification tasks.

   a. Implement `restaurant_sup` and `laptop_sup` dataset for Aspect Based Sentiment Analysis(ABSA)

   > Aspect-based sentiment analysis (ABSA) is a text analysis technique that categorizes data by aspect and identifies the sentiment attributed to each one. Aspect-based sentiment analysis can be used to analyze customer feedback by associating specific sentiments with different aspects of a product or service.

   Here we use data from SemEval-2014 Task 4, which is already downloaded (see [google drive](https://drive.google.com/drive/folders/1H5rmibrg4VfEvM6uqobkrZlGla3xk78-?usp=share_link)). You should prepare the dataset as a `DatasetDict` object, which contains 'train' and 'test' items (you don't need to care about the dev set) and each item is a `Dataset` object that contains `text` and `label`.

   - Hint 1. `text` and `label` are actually two lists of data (the text list should contain text strings, and label list contains integers $0\sim\#label-1$). See Tutorials to know this data structure better.
   - Hint 2. Refer to more ABSA task introduction to know how to prepare `text`. There are many ways to format inputs, and you can use `sep_token` to process the raw text.
   - Hint 3. In this dataset, text is the term and the description, and label is the sentiment polarity.

   b. Implement `acl_sup` dataset for citation intent classification.

   Download `ACL-ARC` dataset (see [google drive](https://drive.google.com/drive/folders/1-8Ly4Jk12LHnkMKnxhuXieVNMA9cyORQ), choose `acl_sup` directory), then process it to be a classification dataset just like the above two datasets (with 'train' and 'test' datasets, ignore the 'dev' dataset).

   c. Implement `agnews_sup` dataset.

   The agnews dataset is a bit large, here we just use its provided test dataset.

   - remove the title, use the 'description' as input text, predict the label.
     - if you use `load_dataset` from huggingface, then you don't need to process 'text', since the title is already removed and the 'description' is 'text'.
   - random split the dataset into training and test set with ratio 9:1 using `train_test_split` API from huggingface `datasets` with `random seed = 2025`.

3. Implement the **few-shot version** of the above tasks: named `restaurant_fs`, `laptop_fs`, `acl_fs`, `agnews_fs`. To prepare a few-shot dataset with appropriate size, you can refer to [this paper](https://arxiv.org/pdf/2109.04332.pdf) for its experimental setup.

4. Implement the **aggregation** of dataset.

   - If we input the `get_dataset()` function with a list of `dataset_name`, e.g. `dataset_name = [’restaurant_fs’, ‘laptop_fs’, ‘acl_fs’]`, the returned dataset should contain all the dataset mentioned in the list.
   - You need to re-label the data samples to avoid label overlapping between different tasks. e.g. `restaurant_fs` use label 0,1,2, and `laptop_fs` use label 3,4,5.
   - You should prepare the dataset with the same format as others. It should be like:

   ```python
   DatasetDict{
   	'train': Dataset({'text':[], 'labels':[]}),
   	'test': Dataset({'text': [], 'labels':[]})
   }
   ```

## Task 2: Training Script

In this task, you have to write a train.py to train a transformer using the dataset prepared from the last task. Here is the [reference meterial](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py) which you can follow, but you should **clear up all the redundant code** if you copy the reference script as your template. We also provides a rough guide for you to complete the script step by step.

Please use all the libraries, functions or classes mentioned below in your training script. You may search for relevant documents or examples to learn their usages, and think about their advantages compared to some basic built-in libraries. There are no requirements about your detailed implementation (e.g. parameter settings).

1. `HfArgumentParser`. It's a command line argument parser. What are its advantages over `argparse`?
2. `logging`. It can output standard logs with specified settings (no specific requirements for your setting). What are its advantages over `print`?
3. `set_seed`. Why should we set random seed?
4. `get_dataset` defined in `dataHelper.py`.
5. `AutoConfig`, `AutoTokenizer`, `AutoModelFor???`. These auto classes are available for instantiating a base model class without a specific head. Different `AutoModel` are designed for different NLP tasks, and you should choose the proper one.
6. `datasets.map` and `AutoTokenizer` to process your dataset into tokenized IDs.
7. `evaluate`. You need to compute `accuracy`, `micro_f1`,`macro_f1` and `weight_f1` using this library.
8. `DataCollatorWithPadding`. Why you should use data collator before training?
9. `Trainer` and `TrainingArguments`. This is the main training engine of 🤗transformers.
10. `wandb`. It can track your experiments and easily interact with 🤗Huggingface. You may [sign up an account](http://wandb.ai) then read the [documents](https://docs.wandb.ai/guides/integrations/huggingface) to setup your recording. Use offline mode if you are unable to connect `wandb` server on your machine.

After you finish the script, run it with these settings:

- Use `bert-base-uncased`, `roberta-large` and `Qwen/Qwen2.5-0.5B` as your base model. Execute this command in your terminal before running to ensure you can download these models.

```sh
export HF_ENDPOINT=https://hf-mirror.com`
```

- Use `restaurant_sup`, `acl_sup` and `agnew_sup` as your datasets.
- To make the results reliable, you need to run the same experiments several times and report the standard deviation.
- Adjust the batch size, epoch number and learning rate to make your results converge stably. Hint: use extra-small learning rate and 3~5 epoches is enough!
- Write a shell script to avoid setting parameters every time it runs.

Write a small report to show the learning curves (metrics, loss during training, you can copy from wandb), results and configurations. Compare and analyze the results for different models and different datasets in your report.

## Task 3: PEFT

Parameter-efficient fine-tuning is a set of techniques to fine-tune a pre-trained model without having to update all of its parameters. PEFT uses significantly less computational resources, making it possible to fine-tune very large models on consumer-grade hardware. Now you need to implement  [Bottelneck Adapter](https://arxiv.org/pdf/1902.00751) and [LoRA](https://arxiv.org/pdf/2106.09685).

1. Re-run the datasets of Task 2 (`restaurant_sup`, `acl_sup` and `agnews_sup`), and also include the results in the report.
2. Try all the base models of Task 2 (`bert-base-uncased`, `roberta-large` and `Qwen/Qwen2.5-1.5B`).
3. Adjust the parameters of LoRA (3 sets of parameters is enough) and compare your experiment results. Hint: try $r=8,16,32$ as the initial setting, and keep $\alpha / r \in [1,2]$.
4. Answer the questions below:
   * How much VRAM do you need to full fine-tune a 3B model? (Just **estimate** it, and don't run the experiment.)
   * How much VRAM is saved when using PEFT methods? (Just **estimate** it, and don't run the experiment.)
   * Can PEFT speed up the training process? Can it speed up the reasoning process? Explain your reasons.

Hint: You can find most of the PEFT methods in [`peft`](https://huggingface.co/docs/peft/index/), but adapter is not inclulded. There's another add-on library [`adapters`](https://docs.adapterhub.ml/), which supports bottleneck adapters but less generic.

## Test and Submit

Once you have implemented the script, you should test it locally to ensure it works as expected.

**Remove all `raise NotImplementedError` and `pass` before testing your code!**

Navigate to the **root directory** of the repository and run the following command:

```
make test-hw2
```

Alternatively, you can run `pytest` directly on this directory:

```
pytest hw2_huggingface/
```

If your implementation is correct, you will see a message indicating that all tests have passed. If there are any failures, the output will help you diagnose what might be wrong with your code. However, the local test is **only for task 1 and parts of task 2**. For task 2 and 3, you should make sure that your script is runnable.

After finish all the tasks, you should submit your code and a report, which includes your experiment results and your answers to the questions. Any other observations or thoughts are welcome.