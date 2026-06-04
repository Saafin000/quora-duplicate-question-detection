# Quora Question Pairs

An NLP project that determines whether two given questions are semantically duplicate — i.e., whether they are asking the same thing, just worded differently.

Dataset: [Kaggle - Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs)

---

## Problem Statement

Quora has millions of questions, and many of them ask the same thing in different ways. Identifying duplicate question pairs helps improve search quality and reduce redundant content. This project builds a machine learning classifier to predict whether a pair of questions is duplicate or not.

---

## Project Structure

```
quora-question-pairs/
├── initial_EDA.ipynb                            # Exploratory Data Analysis
├── only-bow.ipynb                               # Baseline model using Bag of Words only
├── bow-with-basic-features.ipynb                # BOW + basic hand-crafted features
├── bow-with-preprocessing-and-advanced-features.ipynb  # Full pipeline with advanced features
└── streamlit-app/
    ├── app.py                                   # Streamlit web application
    ├── helper.py                                # Feature engineering & preprocessing logic
    ├── model.pkl                                # Trained ML model
    ├── cv.pkl                                   # Fitted CountVectorizer
    ├── requirements.txt                         # Python dependencies
    ├── Procfile                                 # For Heroku deployment
    └── setup.sh                                 # Streamlit server config for deployment
```

---

## Approach

### 1. EDA (`initial_EDA.ipynb`)
- Analyzed class distribution (duplicate vs non-duplicate)
- Explored question length distributions, word frequencies, and common patterns

### 2. Feature Engineering
Features are extracted from each question pair across three categories:

**Basic Features**
- Character length of each question
- Word count of each question
- Number of common words
- Total words combined
- Word share ratio (common / total)

**Token Features**
- Common non-stopword ratio (min & max normalized)
- Common stopword ratio (min & max normalized)
- Common token ratio (min & max normalized)
- First word match / last word match

**Length Features**
- Absolute difference in token count
- Average token length
- Longest common substring ratio

**Fuzzy Features** (via `fuzzywuzzy`)
- QRatio
- Partial ratio
- Token sort ratio
- Token set ratio

**BOW Features**
- Bag of Words vectors for each question using a pre-fitted `CountVectorizer`

### 3. Preprocessing (`helper.py`)
- Lowercasing and stripping
- Special character normalization (`%`, `$`, `₹`, `€`, `@`)
- Number shortening (`1000` → `1k`, etc.)
- Contraction expansion (`don't` → `do not`, etc.)
- HTML tag removal (via BeautifulSoup)
- Punctuation removal

### 4. Model
- Trained on the combined feature vector (22 hand-crafted features + BOW for q1 + BOW for q2)
- Saved as `model.pkl` for inference

---

## Notebooks

| Notebook | Description |
|---|---|
| `initial_EDA.ipynb` | Data exploration and visualization |
| `only-bow.ipynb` | Simple BOW baseline |
| `bow-with-basic-features.ipynb` | BOW + basic features |
| `bow-with-preprocessing-and-advanced-features.ipynb` | Full pipeline with all features |

---

## Streamlit Web App

A simple web interface to test the model live.

### Run Locally

1. Navigate to the app folder:
   ```bash
   cd streamlit-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   > If `sklearn` fails, install manually: `pip install scikit-learn fuzzywuzzy distance bs4 streamlit`

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser at `http://localhost:8501`, enter two questions, and click **Find**.

### Deploy to Heroku

The app includes a `Procfile` and `setup.sh` for Heroku deployment:
```
web: sh setup.sh && streamlit run app.py
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web app framework |
| `scikit-learn` | ML model & CountVectorizer |
| `fuzzywuzzy` | Fuzzy string matching features |
| `distance` | Longest common substring feature |
| `beautifulsoup4` | HTML tag removal in preprocessing |
| `numpy` | Array operations |

---

## Results

The model classifies a question pair as **Duplicate** or **Not Duplicate** based on the combined feature vector. Performance improves significantly when advanced features and preprocessing are added on top of the BOW baseline.
