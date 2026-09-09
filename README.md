# Sentiment Intelligence

An end-to-end natural language processing project for classifying customer-review sentiment across the Amazon, IMDb, and Yelp labelled-sentence datasets.

## Project overview

The workflow covers:

- loading and validating labelled review data;
- text cleaning and exploratory inspection;
- tokenization and padded-sequence preparation;
- a compact bidirectional LSTM classifier;
- training diagnostics and independent test evaluation; and
- reusable prediction helpers for new review text.

The portfolio case study reports an evaluation accuracy of **81.1%**, ROC AUC of **0.894**, and test loss of **0.426** from the original experiment. Exact results can vary with the TensorFlow version, random seed, and data split.

## Repository structure

```text
sentiment-intelligence/
├── data/
│   └── README.md
├── src/
│   └── train.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Run locally

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the UCI Sentiment Labelled Sentences dataset and follow the instructions in `data/README.md`.
4. Train and evaluate the model:

   ```bash
   python src/train.py --data-dir data/sentiment_labelled_sentences
   ```

The script writes the trained model, tokenizer, metrics, and training history to `artifacts/`.

## Model

The classifier uses an embedding layer, a bidirectional LSTM, dropout regularization, and a sigmoid output for binary sentiment classification. Evaluation includes accuracy, ROC AUC, loss, and a classification report.

## Data

This repository does not redistribute the review dataset. The original labelled sentences dataset is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences).

## License

Code in this repository is released under the MIT License. The source dataset remains subject to its own terms and citation requirements.
