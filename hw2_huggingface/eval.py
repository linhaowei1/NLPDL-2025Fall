# hw2_huggingface/eval.py

import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
## TODO: import necessary libraries for evaluation metrics

def inference(model, tokenizer, inputs, max_length=512):
    '''Generate predictions from the model given inputs.'''
    model.eval()
    with torch.no_grad():
        ## TODO: generate predictions using model and tokenizer
        predictions = None
    return predictions


def compute_bleu(references, candidates):
    ## TODO: Compute average BLEU score for reference and candidate pairs.
    bleu_score = None
    return bleu_score


def compute_rouge(references, candidates):
    ## TODO: Compute average ROUGE-L score for reference and candidate pairs.
    rouge_l_score = None
    return rouge_l_score


def compute_bertscore(references, candidates, model="facebook/bart-large"):
    ## TODO: Compute average BERTScore for reference and candidate pairs.
    bertscore = None
    return bertscore


def evaluate_model(model_name, dataset_name, split="test"):
    '''Evaluate the model on the given dataset and split.'''

    ## TODO: Load model and tokenizer
    tokenizer = None
    model = None
    model.eval()

    ## TODO: Load dataset
    dataset = load_dataset(dataset_name, split=split)
    references = None
    candidates = None

    candidates = inference(model, tokenizer, dataset["text"])

    # Compute evaluation metrics
    bleu = compute_bleu(references, candidates)
    rouge_l = compute_rouge(references, candidates)
    bertscore = compute_bertscore(references, candidates)

    return {"BLEU": bleu, "ROUGE-L": rouge_l, "BERTScore": bertscore}


if __name__ == "__main__":
    model_name = "your-model-name"
    dataset_name = "your-dataset-name"

    results = evaluate_model(model_name, dataset_name)
    print("Evaluation Results:", results)
