# Quora Question-pair classification

## **1. Introduction**

The objective of this project is to implement a model capable of detecting duplicate questions from the *Quora Question Pairs* dataset.  
Given two questions, we want to predict whether thwy have the same semantic meaning or not. Detecting duplicates is important in order to reduce redundancy, improve search engines and enhance user experience on question-answering platforms like *Quora* or *Stack Overflow*.  

This project includes two models: a baseline solution given at the start of the assignment and an improved solution incorporating additional features and user state-of-the-art techniques focused on improving roc-auc, precion and recall metrics.

## **2. Baseline Model**

The baseline model is implemented as a simple approach to the Quora duplicate question detection problem. The main steps are:

### **2.1. Data Preparation**

- Extract `question1` and `question2` columns from the dataset.
- Covert all entries into strings to ensure that all questions have the same data type.

### **2.2. Text Vectorization**

- Apply `CountVectorizer` from `Scikit-learn` with `ngram_range=(1, 1)` to create a **bag-of-words** representation.
- The vectorizer is fit only on training data to learn the vocabulary of unique words.
- Each question is transformed into a vector of **word counts**.

### **2.3. Feature Construction**

- Each pair of questions is represented by a concadenation of Q1 and Q2. E.g.
`Feature vector = [Vector of Q1 | Vector of Q2]`
- This generates only one vector per example that can be fed into the classifier.

### **2.4. Model Training**

- We use *Logistic Regression* to predict if the pair is duplicated or not (`is_duplicate` label).

### **2.5. Model Storage**

- Store both trained model and `CountVectorizer` to disk for later use in evaluation.

## **3. Limitations of the Baseline Model**

### **3.1. Missing semantic understanding**

The bag-of-words representation treats each word independently. This means that syhnonyms are not treated as words with the same meaning. E.g. `most inspiring movies` vs `most motivational movies` may be predicted as not duplicates, but they mean the same.

### **3.2. Ignores order**

Following the same example, `movies inspire people` vs `people inspire movies` are treated equally but they do not mean the same. The contextual meaning is lost because we do not preserve order.

### **3.3. Vocabulary limited by training data**

Words not seen in training are ignored. This limits generalizations in the test and validation splits.

### **3.4. High dimensional features**

Large vocabularies produces vectors with mostly zeros. This can lead to overfitting.
As seen in the model evaluation:

| Model        | Split | ROC AUC | Precision | Recall |
| ------------ | ----- | ------- | --------- | ------ |
| Simple Model | Train | 0.8899  | 0.7820    | 0.6867 |
| Simple Model | Val   | 0.8046  | 0.6773    | 0.6107 |
| Simple Model | Test  | 0.8137  | 0.6955    | 0.6188 |

The ROC-AUC is highest on the training ser and drops on validation and test. This indicates a bit of overfitting.  
Precision and recall are moderate across the three splits. Precision is slightly higher than recall.
Model tends to predict pairs as duplicates when unsure, but misses true duplicate pairs (false negatives).

