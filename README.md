# IMDB Movie Review Sentiment Classifier

A binary sentiment classifier (positive/negative) for movie reviews, built from scratch in PyTorch using an LSTM. Trained on the [Stanford Large Movie Review Dataset (aclImdb)](https://ai.stanford.edu/~amaas/data/sentiment/), achieving **85.42% accuracy** on the held-out test set.

## Overview

Given a movie review as raw text, the model predicts whether the sentiment is positive or negative. The pipeline covers the full path from raw `.txt` files to a trained neural network: cleaning and tokenizing text, building a vocabulary, handling variable-length sequences, training an LSTM, and evaluating on a true held-out test set.

- **Architecture:** Single-layer LSTM (embedding → LSTM → dropout → linear classifier)
- **Framework:** PyTorch
- **Dataset:** 50,000 labeled reviews (25,000 train, 25,000 test), split further into 20,000 train / 5,000 validation during development
- **Final test accuracy:** 85.42%

## Results

| Metric | Value |
|---|---|
| Vocabulary size | ~79,000 tokens |
| Max sequence length | 500 tokens |
| Embedding dimension | 64 |
| LSTM hidden dimension | 128 |
| Dropout | 0.5 |
| Best validation accuracy | 86.48% (epoch 8) |
| **Final test accuracy** | **85.42%** |

## What I Learned

This project started as a working end-to-end pipeline that trained without errors but performed no better than a coin flip (~50% validation accuracy). Getting it from there to ~85% test accuracy involved diagnosing and fixing several distinct issues, each of which taught me something concrete:

- **Padding isn't automatically ignored by an LSTM.** Setting `padding_idx=0` on the embedding layer only zeroes out the *embedding vector* for padding tokens — it does nothing to stop the LSTM from processing those zero vectors as real timesteps. Since I was padding every review to a fixed length of 500 tokens and taking the LSTM's final hidden state, the model was reading through potentially hundreds of meaningless padding steps *after* the actual review ended, which diluted or erased whatever the model had learned from the real content. This was the single biggest fix in the whole project.
- **`pack_padded_sequence` / `pad_packed_sequence` exist for exactly this reason.** By tracking the true (pre-padding) length of each review and passing it to `pack_padded_sequence`, the LSTM only processes real tokens and stops exactly where each review actually ends.
- **Vocabulary quality matters as much as model architecture.** My first tokenizer split naively on whitespace, which meant `"teachers"`, `"teachers."`, and `"teachers,"` were all treated as different words, and HTML `<br />` tags scattered throughout the dataset became vocabulary entries of their own. This inflated my vocabulary to over 200,000 tokens, most of which appeared only once or twice and contributed mostly noise. Adding basic text cleaning (stripping HTML, lowercasing, removing punctuation) cut this down dramatically and gave the model a much cleaner signal to learn from.
- **Training loss going down doesn't mean the model is getting better.** By the end of a 15-epoch run, training loss would often fall to near-zero while validation accuracy plateaued or even slightly declined — a textbook sign of overfitting. Tracking validation accuracy after *every* epoch (instead of just once at the end) made this visible and turned a vague "is my model good?" question into an actual measurable curve.
- **Dropout is a cheap, effective regularizer.** Adding a single `nn.Dropout(0.5)` layer between the LSTM output and the final classifier meaningfully reduced the gap between training and validation performance, and made the model's later epochs more stable instead of drifting downward.
- **CPU is not built for recurrent models at scale.** Training an LSTM over 500-token sequences on 20,000 examples was painfully slow on CPU — each epoch was taking long enough that iterating on the model felt impractical. Moving to a free GPU runtime in Google Colab (`model.to(device)`, moving batches to `device` in the training/validation loops) cut epoch time dramatically and made it realistic to actually experiment.
- **Validation and test sets serve different purposes.** I used a validation split (carved out of the training data) throughout development to tune things like dropout and epoch count. The real `test/` folder in the dataset was kept completely untouched until the very end, and evaluated exactly once, using the vocabulary built only from training data (never rebuilding it from test data, which would leak information the model shouldn't have access to). This is what makes the final 85.42% a trustworthy, unbiased number rather than a result quietly tuned to look good.

## Progression of Results

| Stage | Validation Accuracy | What Changed |
|---|---|---|
| Initial working pipeline | ~49.7% (chance) | Model trained without errors, but LSTM was reading through unmasked padding |
| Packed sequences added | ~86% | LSTM now stops at each review's true length instead of processing padding |
| Overfitting diagnosed | — | Added per-epoch validation tracking to see the training/val gap directly |
| Dropout added | ~86.5% peak, more stable across epochs | Reduced the gap between training and validation performance |
| Checkpointing added | — | Best-performing epoch's weights saved and used for final evaluation, instead of whichever epoch happened to be last |
| **Final test evaluation** | **85.42% (test set)** | Single, unbiased check on the true held-out test set |

The jump from ~50% to ~86% came almost entirely from fixing the padding/packing issue — a reminder that a model can train "successfully" (no errors, loss going down) while still being fundamentally broken in a way that only shows up in the accuracy numbers.

## Project Structure

```
sentiment-lstm/
├── sentiment_model.py    # Full training + evaluation script
├── README.md
└── aclImdb/               # Dataset (not committed — see Setup)
```

## Setup

1. Download the dataset from [ai.stanford.edu/~amaas/data/sentiment](https://ai.stanford.edu/~amaas/data/sentiment/) and extract it so `aclImdb/train` and `aclImdb/test` are available alongside the script.
2. Install dependencies:
   ```
   pip install torch scikit-learn
   ```
3. Run the script. Training on GPU (recommended) takes a few minutes for 15 epochs; CPU training is significantly slower due to the recurrent nature of LSTMs.

## Possible Next Steps

- Pretrained word embeddings (e.g., GloVe) instead of learning embeddings from scratch — typically the single biggest lever for improving accuracy on a dataset this size.
- Bidirectional LSTM to let the model use context from both directions of each review.
- Attention mechanism or a transformer-based approach for a further accuracy ceiling.
- Minimum word-frequency cutoff during vocabulary building, to further reduce noise from rare tokens.

  Data Citing:

  @InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.
               and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for
               Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
