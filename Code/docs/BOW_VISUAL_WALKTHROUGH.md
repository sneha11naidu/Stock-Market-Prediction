# Bag of Words: Visual Step-by-Step for Your Code

## Example with Simplified Data

### Your Raw Data (After Step 2 Preprocessing)
```
Doc 1: "stocks rise market good bull"
Doc 2: "stocks fall market bad bear"
Doc 3: "market up stocks gains bull"
```

---

## Step 1: TOKENIZATION (Inside CountVectorizer)
```
Doc 1: ["stocks", "rise", "market", "good", "bull"]
Doc 2: ["stocks", "fall", "market", "bad", "bear"]
Doc 3: ["market", "up", "stocks", "gains", "bull"]
```

---

## Step 2: BUILD VOCABULARY (fit_transform)
When you call: `bow.fit_transform(X_train)`

sklearn scans ALL training documents and creates:
```
Vocabulary (vocabulary index):
0: "stocks"
1: "rise"
2: "market"
3: "good"
4: "bull"
5: "fall"
6: "bad"
7: "bear"
8: "up"
9: "gains"
... (up to max_features=5000 in your code)
```

**Size = 5000 in your actual code** (or however many unique words exist if < 5000)

---

## Step 3: COUNT OCCURRENCES (vectorization)
Convert each document to a count vector using the vocabulary:

```
Doc 1: "stocks rise market good bull"
       ↓
Vector: [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, ...]
         ↑  ↑  ↑  ↑  ↑  positions in vocabulary
       stocks rise market good bull

Doc 2: "stocks fall market bad bear"
       ↓
Vector: [1, 0, 1, 0, 0, 1, 1, 1, 0, 0, ...]

Doc 3: "market up stocks gains bull"
       ↓
Vector: [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, ...]
```

---

## Step 4: FINAL MATRIX (what X_train_bow looks like)

```
              stocks rise market good bull fall bad bear up gains ...
Doc 1:          1     1     1     1    1    0    0   0   0   0
Doc 2:          1     0     1     0    0    1    1   1   0   0
Doc 3:          1     0     1     0    1    0    0   0   1   1
Doc 4:          ...
... (1500 docs)

Shape: (1500 documents, 5000 features)
Type: SPARSE matrix (most values are 0)
```

---

## WHY "SPARSE"?

Most documents don't contain most of the 5000 words, so the matrix is mostly zeros.

**Example:**
```
Actual document might have only 20-30 different words
But vector has 5000 positions
= 4970+ zeros per document!
= 99.5% zeros = SPARSE

Benefit: sklearn stores only non-zero values
Huge memory savings for 1500 × 5000 matrix
```

---

## YOUR CODE IN CONTEXT

### Line 1-2: Build vocabulary + transform training data
```python
bow = CountVectorizer(ngram_range=(1,1), max_features=5000)
X_train_bow = bow.fit_transform(X_train)
# X_train_bow = (1500, 5000) sparse matrix
```

### Line 3: Transform test data with SAME vocabulary
```python
X_test_bow = bow.transform(X_test)
# X_test_bow = (358, 5000) sparse matrix
# ⚠️ CRITICAL: uses vocabulary from training, NOT relearning
```

### Line 4-6: Train and predict
```python
model_bow_lr = LogisticRegression(max_iter=1000)
model_bow_lr.fit(X_train_bow, y_train)
# Learns: which words predict "Up" (positive coefficient)
#         which words predict "Down" (negative coefficient)

y_pred = model_bow_lr.predict(X_test_bow)
# For each test document's BoW vector, predict class
```

---

## WORD IMPORTANCE (from your code's last lines)

```python
feature_names = bow.get_feature_names_out()
coefficients = model_bow_lr.coef_[0]
# coefficients[0] = weight/importance of word 0 ("stocks")
# coefficients[15] = weight/importance of word 15 ("fall")
```

**Example output:**
```
Top 10 important words:
  market: 0.5234   ← positive coef = predicts "Up"
  rise: 0.4891
  gain: 0.4567
  bull: 0.3421
  fall: -0.6234   ← negative coef = predicts "Down"
  bear: -0.5891
  loss: -0.5123
  crash: -0.4891
```

The model learned: words like "rise", "gain", "bull" → predict market UP
                   words like "fall", "crash", "bear" → predict market DOWN

---

## HOW THIS DIFFERS FROM WORD EMBEDDINGS (Lecture 3)

### BoW (what you're doing)
```python
"stock market" 
→ ["stock", "market"] 
→ [1, 1, 0, 0, 0, ...]  ← Just counts, binary/frequency
```

### Word Embeddings (e.g., Word2Vec)
```python
"stock market"
→ ["stock", "market"]
→ stock_vector = [0.234, -0.891, 0.123, ..., 0.456]  (100D)
  market_vector = [0.245, -0.902, 0.134, ..., 0.467]  (100D)
→ Can compute: similarity("stock", "market") = 0.94
```

**Key difference:**
- BoW: **discrete** (words are separate, no notion of similarity)
- Embeddings: **continuous** (words are points in semantic space)

---

## YOUR COURSEWORK STRUCTURE

```
Step 2: Preprocessing
  ↓
Step 3: Train/Test Split
  ↓
Step 4: Model 1 - BoW + LogReg (DETAILED ABOVE)
  ↓
Step 5: Model 2 - TF-IDF + LogReg (similar, but weighted frequencies)
  ↓
Step X: Model 3 - TF-IDF + SVM
  ↓
Step Y: Model 4 - DistilBERT (transformer, not BoW-based)
```

**Report structure should explain:**
1. Why BoW is baseline (simple, interpretable, covered in Lecture 2)
2. How TF-IDF improves on BoW (weights by importance, Lecture 2)
3. Why transformer (DistilBERT) is more complex (Lecture 6, contextual embeddings)
4. Trade-offs: accuracy vs interpretability vs computational cost
