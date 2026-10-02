# Sentiment analysis with BERT

Classifies a book review as positive or negative. BERT turns each review into an embedding, and a logistic regression classifies it.

## How it works

`Model Generator.ipynb`:

1. Loads a Kindle reviews dataset and keeps `rating` and `reviewText`.
2. Labels reviews with a rating of 3 or below as negative (0) and 4 or 5 as positive (1).
3. Embeds each review with `bert-base-uncased` (mean of the last hidden state), processing 12,000 reviews in batches.
4. Trains a logistic regression on the embeddings and saves it as `logistic_regression_model.pkl`.

Result on a held-out set of 2,400 reviews: **accuracy 0.85**, with precision and recall of 0.85 for both classes.

`app.py` is a Streamlit app: type a sentence, and it shows a green "Positive" or red "Negative" banner.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the BERT weights. The Kindle dataset is not included; the notebook expects the CSV at a local path you will need to change.
