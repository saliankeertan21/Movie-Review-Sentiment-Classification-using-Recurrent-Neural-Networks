# Movie Review Sentiment Analysis Using Deep Learning

An educational NLP project classifying IMDb movie reviews as **positive** or **negative**. It compares **SimpleRNN, LSTM, GRU, and Bidirectional LSTM** architectures and investigates overfitting, regularization, and classification errors.

## Dataset

The [Keras IMDb dataset](https://keras.io/api/datasets/imdb/) contains 25,000 labeled training reviews and 25,000 labeled test reviews. Reviews are integer-encoded; labels are `0` (negative) and `1` (positive). The experiments limit the vocabulary to 10,000 tokens and pad/truncate sequences to 200 positions.

## Workflow

1. Load and inspect the integer-encoded reviews.
2. Decode sample reviews, then pad/truncate token sequences.
3. Train SimpleRNN, LSTM, and GRU baselines with learned embeddings.
4. Compare validation performance and investigate overfitting.
5. Experiment with a smaller LSTM, Dropout, and Early Stopping.
6. Train a Bidirectional LSTM.
7. Evaluate using a confusion matrix, precision, recall, and F1-score.
8. Run inference on example reviews using an approximate tokenizer.

## Selected model: Bidirectional LSTM

| Metric | Result |
|---|---:|
| Validation accuracy | 85.50% |
| Validation loss | 0.3639 |
| Test accuracy | **84.42%** |
| Test loss | **0.3817** |
| Positive-class precision (validation) | 0.87 |
| Positive-class recall (validation) | 0.83 |
| Positive-class F1 (validation) | 0.85 |

### Validation confusion matrix

| Actual / predicted | Negative | Positive |
|---|---:|---:|
| Negative | 2,191 | 309 |
| Positive | 416 | 2,084 |

There were 725 errors among 5,000 validation reviews. More actual positive reviews were misclassified as negative (416) than vice versa (309).

## Sample inference results

| Example | Predicted class | Predicted positive probability |
|---|---|---:|
| Fantastic movie, brilliant acting | Positive | 0.9223 |
| Boring plot, disappointing ending | Negative | 0.0106 |
| Excellent acting, terrible story | Negative | 0.3627 |
| Expected bad, surprisingly good | Positive | 0.5943 |

These are outputs from four short examples, not evidence of real-world accuracy.

## Running the notebook

Open `notebooks/Sentiment_Analysis_RNN_LSTM_GRU.ipynb` in [Google Colab](https://colab.research.google.com/) and run all cells in order. Install the dependencies using:

```bash
pip install -r requirements.txt
```

The notebook and saved model files must be added to this repository after downloading them from Colab.

## Important limitations

- This is an educational model, not a production sentiment system.
- Inputs are truncated to 200 token positions, so longer reviews can lose information.
- The manual tokenizer for newly typed reviews only approximates the original IMDb preprocessing; this can affect predictions.
- The best validation results and final test results should be distinguished.
- Models were not all compared under identical regularization and architecture settings.

## Topics learned

Tokenization, embeddings, padding, SimpleRNN, LSTM, GRU, Bidirectional LSTM, Early Stopping, Dropout, train/validation/test evaluation, confusion matrix, precision, recall, and F1-score.

