# hw2_huggingface/test_eval.py

import pytest
from .eval import compute_bleu, compute_rouge, compute_bertscore


def test_compute_bleu():
    '''
    Test the compute_bleu function with dummy references and candidates.
    '''

    references = [
        "The quick brown fox jumps over the lazy dog.",
        "I love listening to music while I work."
    ]
    candidates = [
        "A fast brown fox jumps over a lazy dog.",
        "I enjoy listening to music as I do my work."
    ]
    bleu_score = compute_bleu(references, candidates)
    expected_bleu = (
        0.2854,  # `sacrebleu` BLEU score (default settings)
        0.1900,  # `nltk` BLEU score + `nltk.word_tokenize`, no smoothing
        0.1775,  # `nltk` BLEU score + `str.split()`, no smoothing
    )
    assert any(abs(b - e) < 1e-3 for b, e in zip(bleu_score, expected_bleu))


def test_compute_rouge():
    '''
    Test the compute_rouge function with dummy references and candidates.
    '''

    references = [
        "The quick brown fox jumps over the lazy dog.",
        "I love listening to music while I work."
    ]
    candidates = [
        "A fast brown fox jumps over a lazy dog.",
        "I enjoy listening to music as I do my work."
    ]
    rouge_l_score = compute_rouge(references, candidates)
    expected_rouge_l = (
        0.6667,  # `rouge-score` ROUGE-L score
        0.6458,  # `rouge` ROUGE-L score
    )
    assert any(abs(r - e) < 1e-3 for r, e in zip(rouge_l_score, expected_rouge_l))



def test_compute_bertscore():
    '''
    Test the compute_bertscore function with dummy references and candidates.
    '''

    references = [
        "The quick brown fox jumps over the lazy dog.",
        "I love listening to music while I work."
    ]
    candidates = [
        "A fast brown fox jumps over a lazy dog.",
        "I enjoy listening to music as I do my work."
    ]
    bertscore = compute_bertscore(references, candidates)
    expected_bertscore = 0.9391  # Expected BERTScore for the dummy data
    assert abs(bertscore - expected_bertscore) < 1e-2
