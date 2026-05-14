# Bag of Words & Course Complexity Guide

## 1. WHAT IS BAG OF WORDS (BoW)?

### Conceptual Definition
A text representation that:
- Ignores **word order** (hence "bag" - order doesn't matter)
- Counts **word frequencies** in each document
- Represents each document as a vector of counts

### Mathematical View
```
Vocabulary V = {word1, word2, ..., wordN}
Document d = "the stock market rose today"

BoW(d) = [1, 1, 1, 1, 1, 0, 0, ..., 0]
          ↑  
        each position = count of that vocabulary word
```

### YOUR Code
```python
CountVectorizer(ngram_range=(1,1), max_features=5000)
```
Creates a sparse matrix where:
- Rows = documents
- Columns = top 5000 words (features)
- Values = word counts

---

## 2. DO YOU NEED TO KNOW HOW IT WORKS?

### For Your Coursework: **YES**

**Why:**
- **Lecture 2 explicitly covers**: "Text Classification and Distributional Semantics"
- **Required reading**: Chapter 4 (SLP3) on BoW-based features
- **Coursework marks grading criteria likely includes**: understanding text representation methods
- Your comparative study SHOULD explain WHY BoW is a baseline

### What You Should Know
✅ How to explain: "BoW converts text to frequency vectors"
✅ Implementation details: fit vocabulary on train, transform test with same vocabulary
✅ Limitations: loses word order (syntactic structure), can't capture "apple pie" as single concept
✅ Why it's useful: fast, interpretable, baseline for comparison

---

## 3. YOUR COURSEWORK COVERAGE vs LECTURE TOPICS

### ✅ WELL-COVERED
| Lecture | Topic | Your Implementation |
|---------|-------|-------------------|
| Lecture 1 | Text Preprocessing | `combine_headlines()`, text cleaning |
| Lecture 2 | BoW Features | Model 1: BoW + LogReg |
| Lecture 2 | Distributional Semantics (TF-IDF) | Model 2: TF-IDF + LogReg |
| Lecture 6 | Transformers | Model 4: DistilBERT |

### ⚠️ PARTIALLY-COVERED
| Lecture | Topic | Status |
|---------|-------|--------|
| Lecture 3 | Word Embeddings (Word2Vec, GloVe) | NOT IMPLEMENTED |
| Lecture 4 | Sequence Models (N-grams, Language Models) | MINIMAL (bigrams only in TF-IDF) |
| Lecture 5 | Neural Sequence Models (LSTM, GRU) | NOT IMPLEMENTED |

### ❌ NOT-COVERED
- Attention mechanisms visualization
- Fine-tuning strategies for transformers
- Error analysis (why did model fail?)

---

## 4. WHAT NLP TECHNIQUES ARE YOU ACTUALLY DOING?

### ✅ IMPLEMENTED:

**Tokenization:**
```python
re.sub("[^a-zA-Z]", " ", text)  # Remove non-alphabetic chars
text.lower()                      # Lowercase
' '.join(text.split())           # Split into tokens
```

**Feature Extraction (Multiple methods):**
- **BoW** (unigrams): `CountVectorizer(ngram_range=(1,1))`
- **TF-IDF** (unigrams + bigrams): `TfidfVectorizer(ngram_range=(1,2))`
- **Contextual Embeddings**: DistilBERT tokenizer (byte-pair encoding) + transformer

**Classification:**
- Logistic Regression (traditional ML)
- SVM (classical ML, Model 3)
- Transformer-based (neural)

### ❌ NOT IMPLEMENTED:

**Word Embeddings (Lecture 3):**
```python
# You could add this:
from gensim.models import Word2Vec
w2v = Word2Vec([tokens1, tokens2, ...], vector_size=100)
```

**Sequence Modeling (Lecture 4-5):**
```python
# LSTM example:
from tensorflow.keras.layers import LSTM
# model = Sequential([LSTM(...), Dense(...)])
```

**N-gram Language Models (Lecture 4):**
```python
# Basic trigram model - not needed but shows understanding
from nltk.lm import NgramLanguageModel
```

---

## 5. HOW TO DECIDE IF YOU NEED WORD EMBEDDINGS

### Questions to Ask:
1. **Does your coursework rubric mention Lecture 3?** 
   - If YES → should probably add Word2Vec/GloVe
   - If NO → BoW + TF-IDF + Transformer is sufficient

2. **Are you targeting 60+ marks or 75+?**
   - 60+: Current 4-model comparison is solid
   - 75+: Add analysis like:
     - Word2Vec embedding visualization (t-SNE/UMAP)
     - Compare word similarities: `w2v.similarity("stock", "market")`
     - Attention visualization from DistilBERT

3. **Did your class materials require specific models?**
   - Check your coursework brief PDF for approved model list

---

## 6. RECOMMENDED ACTION FOR YOUR COURSEWORK

### Current Status: ✅ GOOD FOUNDATION
- ✅ 4 different model families (classical ML + neural)
- ✅ Demonstrates progression in sophistication
- ✅ Compares BoW vs TF-IDF vs SVM vs Transformer
- ✅ Covers Lectures 1, 2, 6

### To Strengthen (Optional, depends on mark target):
**If aiming for 75+, consider adding:**
```python
# In new cell - Word2Vec analysis
from gensim.models import Word2Vec

# Train Word2Vec on your documents
w2v_model = Word2Vec(
    sentences=[doc.split() for doc in all_headlines],
    vector_size=100,
    window=5,
    min_count=2
)

# Show top words related to financial terms
print(w2v_model.most_similar("stock", topn=5))
print(w2v_model.most_similar("market", topn=5))

# Use as features
X_train_w2v = np.array([
    np.mean([w2v_model.wv[word] for word in doc.split() if word in w2v_model.wv.index_to_key],axis=0)
    for doc in X_train
])
```

This would:
- Implement Lecture 3 content (Word Embeddings)
- Show understanding of continuous vs discrete representations
- Give 5+ extra paragraphs for your report

---

## 7. KEY TAKEAWAY

**For the university marking scheme:**

| Component | Your Coverage |
|-----------|----------------|
| Understanding text preprocessing | ✅ Good |
| Implementing BoW correctly | ✅ Excellent |
| Comparative analysis | ✅ Good (4 models) |
| Theoretical depth | ⚠️ Need to explain WHY each method in your report |
| Visual/Attention analysis | ⚠️ Consider for DistilBERT |
| Alignment with course lectures | ⚠️ Missing Lecture 3-5 content in code (but can discuss in report) |

**Bottom line:** Your technical implementation is good. Your coursework report MUST explain the course concepts (BoW definition, TF-IDF theory, why transformers are more complex, etc.) to match course complexity expectations.
