## Fake News Detection using Logistic Regression & TF‑IDF

This project implements a **fake news detection** pipeline in Python using:

- **Scikit‑learn** (`TfidfVectorizer` + `LogisticRegression`)
- A combined dataset from **`True.csv`** and **`Fake.csv`**
- A trained and serialized model in **`FakeNews_detector.pkl`**

The core workflow is developed in the notebook `Fake_News_Detection.ipynb`.

---

## Project Structure

- **`Fake_News_Detection.ipynb`** – main notebook: data loading, cleaning, training, evaluation, and quick prediction.
- **`True.csv`** – labeled *real* news articles.
- **`Fake.csv`** – labeled *fake* news articles.
- **`FakeNews_detector.pkl`** – saved model package (`Pipeline` + cleaning function) for reuse.

These files are enough to reproduce the full training and evaluation pipeline after cloning the repository.

---

## Environment & Requirements

Recommended environment:

- **Python**: 3.9+ (developed and tested on Python 3.13)

Suggested `requirements.txt`:

```txt
numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
jupyter
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data & Labels

The project uses two CSV files:

- **`True.csv`** – real news articles
- **`Fake.csv`** – fake news articles

In the notebook:

- Each dataset gets a **`label`** column:
  - `1` → **Real**
  - `0` → **Fake**
- The two datasets are concatenated, shuffled, and reduced to:
  - `text` – article body
  - `label` – target

---

## Modeling Pipeline

The notebook builds the following pipeline:

- **Text cleaning** (`clean` function)
  - Lowercasing
  - Removing URLs, HTML tags, digits, punctuation
  - Normalizing whitespace
- **Vectorization**: `TfidfVectorizer`
  - `stop_words='english'`
  - `max_features=10000`
  - `ngram_range=(1, 1)` (unigrams only)
  - `sublinear_tf=True`
- **Classifier**: `LogisticRegression`
  - `solver='saga'`
  - `max_iter=2000`
  - `class_weight='balanced'` (handles slight label imbalance)

Hyperparameters are tuned via `GridSearchCV`:

- Parameter grid:
  - `clf__C = [0.01, 0.1, 0.5, 1.0]`
- Cross‑validation:
  - `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Scoring:
  - `f1` score

The best model is stored as:

- `best_model = grid.best_estimator_`

---

## Evaluation

On the held‑out test set (20% split, stratified), the model achieves:

- **Accuracy** ≈ **0.99**
- **Precision / Recall / F1** for both classes ≈ **0.99**

The notebook displays:

- A **classification report** (`classification_report`)
- A **confusion matrix** plotted with `seaborn.heatmap`

> **Note**: These metrics are specific to this dataset. Real‑world performance should be validated on external data before deployment.

---

## How to Run the Notebook

1. **Clone the repository**:

   ```bash
   git clone <your-repo-url>.git
   cd <your-repo-name>
   ```

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # macOS / Linux
   # source .venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Start Jupyter**:

   ```bash
   jupyter notebook
   ```

5. Open **`Fake_News_Detection.ipynb`** and run all cells (Kernel → Restart & Run All):
   - Load data
   - Clean text
   - Train the model with `GridSearchCV`
   - Evaluate on the test set
   - Save `FakeNews_detector.pkl`

---

## Using the Trained Model (Prediction)

In the notebook, a helper function is defined:

```python
def predict_news(text):
    cleaned = clean(text)
    pred = best_model.predict([cleaned])[0]
    prob = best_model.predict_proba([cleaned])[0][1]  # probability of Real (label=1)
    return pred, prob
```

Example usage:

```python
sample = (
    "Researchers at the University of Cambridge published a new study in the journal "
    "Science, revealing that global ocean temperatures have reached record highs for "
    "the third consecutive year."
)
label, prob = predict_news(sample)
print(f"Sample Prediction: {'Real' if label == 1 else 'Fake'} (prob={prob:.3f})")
```

To use the saved model in a standalone Python script:

```python
import joblib

package = joblib.load("FakeNews_detector.pkl")
model = package["model_pipeline"]
clean = package["cleaner"]

def predict_news(text: str):
    cleaned = clean(text)
    pred = model.predict([cleaned])[0]
    prob = model.predict_proba([cleaned])[0][1]
    return pred, prob
```

---

## Possible Improvements

If you want to extend this project further, consider:

- Adding a **separate validation set** (train / validation / test split) to better measure generalization.
- Implementing an **“uncertain” zone** (e.g. if probability is between 0.4 and 0.6, flag as uncertain instead of hard fake/real).
- Trying other models (e.g. **Linear SVM**, **Random Forest**, or modern **transformer‑based** text classifiers).
- Building a small **web API** (FastAPI / Flask) or **web UI** for interactive fake news detection.

