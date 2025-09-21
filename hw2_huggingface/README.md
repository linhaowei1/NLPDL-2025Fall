# Homework 2: Huggingface & PEFT

This assignment is designed to help you study [Huggingface🤗](https://huggingface.co/), which is the most popular tool for NLP researchers.

## Task Overview

1. Implement dataset loading script supporting few-shot and aggregation
2. Implement a training script using `🤗transformers` and train some models
3. Add PEFT methods in your script, then compare it with full fine-tuning.
4. Use LLaMA-Factory framework to train a LLM and compute metrics.

## Setup

### Environment and Unit Test

We use `uv` with a per-assignment virtual environment. Run commands from this `hw2_huggingface` directory (or use `--directory`).

```sh
# One-time setup for this assignment
uv sync

# One-time setup for this assignment (for Ascend users)
uv sync --extra npu

# Run Python or tests in this assignment's environment
uv run pytest . --ignore=LLaMA-Factory

# From repo root (alternative)
uv run --directory hw2_huggingface pytest . --ignore=LLaMA-Factory
```

`uv` will automatically resolve and activate the environment specified by this assignment’s `pyproject.toml`.

For task 4 (LLaMA-Factory), just follow its README.md.

```sh
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

uv sync --extra torch --extra metrics --extra deepspeed --prerelease=allow

# for HUAWEI Ascend
uv sync --extra torch-npu --extra metrics --extra deepspeed --prerelease=allow

# set FORCE_TORCHRUN to activate deepspeed
FORCE_TORCHRUN=1 uv run --prerelease=allow llamafactory-cli train examples/train_lora/llama3_lora_pretrain.yaml
```

If you encounter any problems about dependencies and environment setup in task 4, please turn to use conda environment and re-install LLaMA-Factory.

### Notice for All Students

- VPN is needed to connect [Huggingface🤗](https://huggingface.co/)
- Directly download datasets and models from [Huggingface🤗](https://huggingface.co/) is costly and slow. An alternative way is to use its mirror site [HF-Mirror](https://hf-mirror.com). Execute the command below before running your script.

```sh
export HF_ENDPOINT=https://hf-mirror.com
```

### Notice for HUAWEI Ascend Users

If you are using HUAWEI Ascend NPU:

- Choose docker image with `cann>=8.0.rc1`
- Import `torch_npu` package before importing your training script
- Replace `torch` to `torch-npu` when installing LLaMa-Factory in Task 4 (see [here](https://ascend.github.io/docs/sources/llamafactory/install.html))
- Open "Remote Development using SSH", and use `scp` commands to transmit your local dataset

## Statement on AI tools

Prompting LLMs such as ChatGPT is permitted for low-level programming questions or high-level conceptual questions about language models, but using it directly to solve the problem is not allowed.

We strongly encourage you to disable AI autocomplete (e.g., Cursor Tab, GitHub CoPilot) in your IDE when completing assignments (though non-AI autocomplete, e.g., autocompleting function names is totally fine). We have found that AI autocomplete makes it much harder to engage deeply with the material.

## Task 1: Build Your Dataset

[🤗Datasets](https://huggingface.co/docs/datasets/index) is a package for easily accessing and sharing datasets, and evaluation metrics for Natural Language Processing (NLP), computer vision, and audio tasks. You have to read the docs to learn how to use this package and finish the tasks below. In this task, you should hand in such a `dataHelper.py` file. The requirements are as follows.

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

   - Remove the title, use the 'description' as input text, predict the label.
     - if you use `load_dataset` from huggingface, then you don't need to process 'text', since the title is already removed and the 'description' is 'text'.
   - Random split the dataset into training and test set with **ratio 9:1** using `train_test_split` API from huggingface `datasets` with `random seed = 2025`.

3. Implement the **few-shot version** of the above tasks: named `restaurant_fs`, `laptop_fs`, `acl_fs`, `agnews_fs`. To prepare a few-shot dataset with appropriate size, you can refer to [this paper](https://arxiv.org/pdf/2109.04332.pdf) for its experimental setup.

4. Implement the **aggregation** of datasets.

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

[🤗Transformers](https://huggingface.co/docs/transformers/index) is one of the core package when you are using pre-train models for fine-tuning or inferences. In this task, you have to write a `train.py` to train a transformer using the dataset prepared from the last task. Here is the [reference meterial](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py) which you can follow, but you should **clear up all the redundant code** if you copy the reference script as your template. We also provides a rough guide for you to complete the script step by step.

Please use all the packages, functions or classes mentioned below in your training script. You may search for relevant documents or examples to learn their usages, and think about their advantages compared to some basic built-in packages. There are no requirements about your detailed implementation (e.g. parameter settings).

1. `HfArgumentParser`. It's a command line argument parser. What are its advantages over `argparse`?
2. `logging`. It can output standard logs with specified settings (no specific requirements for your setting). What are its advantages over `print`?
3. `set_seed`. Why should we set random seed?
4. `get_dataset` defined in `dataHelper.py`.
5. `AutoConfig`, `AutoTokenizer`, `AutoModelFor???`. These auto classes are available for instantiating a base model class without a specific head. Different `AutoModel` are designed for different NLP tasks, and you should choose the proper one.
6. `datasets.map` and `AutoTokenizer` to process your dataset into tokenized IDs.
7. `evaluate`. You need to compute `accuracy`, `micro_f1`,`macro_f1` and `weight_f1` using [this package](https://huggingface.co/docs/evaluate/index).
8. `DataCollatorWithPadding`. Why you should use data collator before training?
9. `Trainer` and `TrainingArguments`. This is the main training engine of 🤗transformers.
10. `wandb`. It can track your experiments and easily interact with 🤗Huggingface. You may [sign up an account](http://wandb.ai) then read the [documents](https://docs.wandb.ai/guides/integrations/huggingface) to setup your recording. Use offline mode if you are unable to connect `wandb` server on your machine.

After you finish the script,

- Use `bert-base-uncased`, `facebook/bart-base` and `Qwen/Qwen1.5-0.5B` as your base model.
- Use `restaurant_sup`, `acl_sup` and `agnew_sup` as your datasets.
- To make the results reliable, you need to run the same experiments several times and report the standard deviation.
- Adjust the batch size, epoch number and learning rate to make your results converge stably.
  > These models are highly capable and have been pre-trained on a large scale of data. Set a **small learning rate** and run **few epoches** to avoid overfitting or even catastrophic forgetting!
- It is recommended to write a shell script to avoid entering parameters every time. Don't forget to set `HF_ENDPONIT` in your shell script!

## Task 3: PEFT

Parameter-efficient fine-tuning is a set of techniques to fine-tune a pre-trained model without having to update all of its parameters. PEFT uses significantly less computational resources, making it possible to fine-tune very large models on consumer-grade hardware. Now you need to implement [Bottleneck Adapter](https://arxiv.org/pdf/1902.00751) and [LoRA](https://arxiv.org/pdf/2106.09685).

1. Re-run `agnews_sup` dataset with PEFT and include the results in the report.
2. Try all the base models of Task 2 (`bert-base-uncased`,`facebook/bart-large` and `Qwen/Qwen1.5-0.5B`).
3. Choose one of the models above, adjust the parameters of LoRA (3~5 sets of parameters is enough) and compare your experiment results.
4. Answer the questions below:
   - How much VRAM do you need to full fine-tune a **Llama-3-70B**? (Just **estimate** it, and don't run the experiment.)
   - How much VRAM is saved when using PEFT methods? (Just **estimate** it, and don't run the experiment.)
   - Can PEFT speed up the training process? Can it speed up the reasoning process? Explain your reasons.

Hint:

- You can find most of the PEFT methods in [`peft`](https://huggingface.co/docs/peft/index/), but adapter is not included. There's another add-on package [`adapters`](https://docs.adapterhub.ml/), which supports bottleneck adapters but less generic.
- For LoRA, it is reccomanded to keep $\alpha / r \in [1,2]$. $r = 16$ could be a good initial setting.
- Larger learning rate and more epoches are needed when using PEFT methods.

## Task 4: LLaMA-Factory

Nowadays, there are more easy-to-operate frameworks available for training or inferring large models based on the pytorch and transformers libraries, among which [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) is one of the most popular. In this task, you will try to fine-tune a model without writing any script, and evaluate your model with some classic metrics.

1. Setup `LLaMA-Factory` according to the documents.
   - Remember to add `deepspeed` dependencies!
2. Add `knkarthick/samsum` into `LLama-Factory/data/dataset_info.json`
3. Write a `yaml` file to fine-tune `Qwen/Qwen1.5-0.5B` on the training set with LoRA.
   - Set `template: qwen` to correctly load prompt template
   - Set `cutoff_len: 2048`
   - Set `lora_target: all`
   - Set `max_samples: 3000`
   - You can find example files in `LLama-Factory/examples/train_lora` or [here](https://ascend.github.io/docs/sources/llamafactory/quick_start.html).
4. Write a `yaml` file to merge your lora module model with the original model.
   - You can find example files in `LLama-Factory/examples/merge_lora`.
5. Start the engine to fine-tune your model.
6. Implement `eval.py` to calculate `BLEU`, `ROUGE-L` and `BERTScore-F1`, and run your evaluation on the test set.
   - Set `facebook/bart-large` as the scoring model of `BERTScore`
   - `bert_score` package supports batch process of multiple reference-candidate pairs, but `rouge` and `bleu` in nltk can only process one pair per call.

After finish the tasks, you should copy your written `yaml` files for training and merging to `hw2_huggingface` directory.

## Submission

You will submit the following files to Gradescope:

- `[Name_ID number_Report].pdf`: Record your experiment setup, results and your analysis. All the questions in the guidance or code template should be answered. For experiment results, you can copy images from wandb website.
- `code.zip`: Contains all the code you've written. (Please exclude large data files and model checkpoints)
