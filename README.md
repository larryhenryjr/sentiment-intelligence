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

The completed experiment reports held-out accuracy of **79.33%**, ROC AUC of **0.8518**, and binary cross-entropy loss of **0.5925** using the restored best model. Exact results can vary with the TensorFlow version, random seed, and data split.

## Portfolio artifacts

- [Rendered analysis notebook](notebook/analysis.html)
- [Portfolio report](reports/sentiment-intelligence-report.pdf)

The public repository includes the rendered notebook and report while intentionally excluding the working `.ipynb` file, restricted source data, local paths, and academic grading material.
## Repository structure

```text
sentiment-intelligence/
├── data/
│   └── README.md
├── src/
│   ├── data.py
│   └── train.py
├── tests/
│   └── test_data.py
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

Run the data-quality tests with:

```bash
pytest tests/ -v --tb=short
```

## Model

The classifier uses an embedding layer, a bidirectional LSTM, dropout regularization, and a sigmoid output for binary sentiment classification. Evaluation includes accuracy, ROC AUC, loss, and a classification report.

## Data

This repository does not redistribute the review dataset. The original labelled sentences dataset is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences).

## License

Code in this repository is released under the MIT License. The source dataset remains subject to its own terms and citation requirements.
