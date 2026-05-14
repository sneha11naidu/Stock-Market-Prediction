# Stock Market Prediction from News Headlines

**Course:** INM434 NLP Coursework  
**Student:** Sneha Naidu (210048420)

## Overview

This project compares 5 NLP approaches to predict DJIA movement from news headlines:

1. **Bag of Words + Logistic Regression** – F1: 0.511
2. **TF-IDF + Logistic Regression** – F1: 0.529
3. **TF-IDF + Support Vector Machine** – F1: 0.538
4. **TF-IDF + Multinomial Naive Bayes** (Best) – F1: 0.663
5. **Fine-tuned FinBERT** – F1: 0.609

## Quick Start

### 1. Activate Virtual Environment

```bash
source nlp-env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Notebook

```bash
jupyter notebook Code/stock_market_prediction.ipynb
```

## Usage

- **Full training**: Run all cells to train models from scratch (~15-20 minutes)
- **Test evaluation only**: Run cells 1-14, then cells 35-36 to load pre-trained models and evaluate (~1-2 minutes)

## Project Structure

```
Code/
  stock_market_prediction.ipynb    # Main notebook
  requirements.txt                 # Dependencies
data/
  raw/
    Combined_News_DJIA.csv         # Dataset
models/                            # Pre-trained model files (.pkl)
outputs/                           # Results and metrics
Graphs/                            # Confusion matrices and visualizations
docs/                              # Supplementary documentation
```

## Documentation

See `docs/` folder for detailed explanations of BOW, TF-IDF, and visualization walkthroughs.
