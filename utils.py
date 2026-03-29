import scipy.sparse
import numpy as np
import re
from collections import Counter


####################################################################
# Original helpers
####################################################################


def cast_list_as_strings(mylist):
    """
    Return a list of strings given a list

    :mylist: the list to cast as strings
    """
    return [str(x) for x in mylist]


def get_features_from_df(df, count_vectorizer):
    """
    Return the features of a dataframe given a count vectorizer

    :df: the dataframe containing the questions
    :count_vectorizer: the count vectorizer to use for vectorization
    """
    q1_casted = cast_list_as_strings(list(df["question1"]))
    q2_casted = cast_list_as_strings(list(df["question2"]))

    X_q1 = count_vectorizer.transform(q1_casted)
    X_q2 = count_vectorizer.transform(q2_casted)

    return scipy.sparse.hstack([X_q1, X_q2])


def get_mistakes(clf, X_q1q2, y):
    """
    Return the mistakes of a classifier given the features and the labels

    :clf: the classifier to evaluate
    :X_q1q2: the features of the questions pairs (after vectorization)
    :y: the true labels of the question pairs
    """
    predictions = clf.predict(X_q1q2)
    incorrect_predictions = predictions != y
    incorrect_indices = np.where(incorrect_predictions)[0]

    if np.sum(incorrect_predictions) == 0:
        print("No mistakes found.")
        return None, None
    else:
        return incorrect_indices, predictions


def print_mistake_k(train_df, k, mistake_indices, predictions):
    """
    Print the k first mistakes of a classifier given the mistake indices and the predictions

    :train_df: the dataframe containing the questions and labels
    :k: the index of the mistake to print (0 for the first mistake, 1 for the second, etc.)
    :mistake_indices: the indices of the mistakes in the original dataframe
    :predictions: the predicted labels of the classifier for the original dataframe
    """
    print(train_df.iloc[mistake_indices[k]].question1)
    print(train_df.iloc[mistake_indices[k]].question2)
    print("True label:", train_df.iloc[mistake_indices[k]].is_duplicate)
    print("Predicted label:", predictions[mistake_indices[k]])


####################################################################
# Text preprocessing
####################################################################

STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "it",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "and",
    "or",
    "but",
    "are",
    "was",
    "were",
    "be",
    "been",
    "do",
    "does",
    "did",
    "has",
    "have",
    "had",
    "will",
    "would",
    "can",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "this",
    "that",
    "these",
    "those",
    "with",
    "as",
    "by",
    "from",
    "up",
    "about",
    "into",
    "through",
    "during",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
}


def _tokenize(text, remove_stopwords=False):
    """
    Lowercase, remove punctuation, split into word tokens.
    Optionally remove stop words.

    :text: input text
    :remove_stopwords: if True, remove common stop words
    :returns: list of tokens
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # remove punctuation
    tokens = [t for t in text.split() if t]  # split and remove empty tokens
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


def _char_ngrams(text, n=3):
    """
    Counter of character n-grams in the text after lowercasing and removing spaces.

    :text: input text
    :n: n-gram length (default 3 for trigrams)
    :returns: Counter of n-grams
    """
    text = re.sub(r"\s+", "", str(text).lower().strip())  # remove spaces and lowercase
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


####################################################################
# Feature engineering
####################################################################


def jaccard_similarity(text1, text2, remove_stopwords=False):
    """
    Jaccard similarity between two sets of tokens.
    Jaccard (A, B) = |A ∩ B| / |A ∪ B|
    The stop-word variant captures overlap only.

    :text1: first question text
    :text2: second question text
    :remove_stopwords: if True, remove common stop words before computing similarity
    :returns: float in [0, 1]
    """
    set1 = set(_tokenize(text1, remove_stopwords))
    set2 = set(_tokenize(text2, remove_stopwords))
    if not set1 and not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def shared_word_ratio(text1, text2, remove_stopwords=False):
    """
    Sorensen-Dice coefficient on token bags (multisets)
    Dice(A, B) = 2 * |A ∩ B| / (|A| + |B|)
    Unlike Jaccard, repeated words count (bags, not sets)

    :text1: first question text
    :text2: second question text
    :remove_stopwords: if True, remove common stop words before computing ratio
    :returns: float in [0, 1]
    """
    tokens1 = _tokenize(text1, remove_stopwords)
    tokens2 = _tokenize(text2, remove_stopwords)
    total = len(tokens1) + len(tokens2)
    if total == 0:
        return 0.0
    bag1 = Counter(tokens1)
    bag2 = Counter(tokens2)
    common = sum((bag1 & bag2).values())
    return 2 * common / total


def char_ngram_similarity(text1, text2, n=3):
    """
    Jaccard similarity on character n-grams.
    Captures surface-level similarity, including typos.

    :text1: first question text
    :text2: second question text
    :n: n-gram length (default 3 for trigrams)
    :returns: float in [0, 1]
    """
    ngrams1 = set(_char_ngrams(text1, n).keys())
    ngrams2 = set(_char_ngrams(text2, n).keys())
    if not ngrams1 and not ngrams2:
        return 0.0
    return len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2)


def length_difference_ratio(text1, text2):
    """
    Absolute length difference ratio between two texts.
    Captures length-based similarity.

    :text1: first question text
    :text2: second question text
    :returns: float in [0, 1]
    """
    tokens1 = _tokenize(text1)
    tokens2 = _tokenize(text2)
    len1, len2 = len(tokens1), len(tokens2)
    if max(len1, len2) == 0:
        return 0.0
    return abs(len1 - len2) / max(len1, len2)


def tfidf_cosine_similarity(text1, text2, remove_stopwords=False):
    """
    TF-IDF cosine similarity between two texts, implemented from scratch.

    1. Tokenize both texts (optionally removing stop words)
    2. Compute TF (normalized term frequency) per document
    3. Compute smooth IDF over the two-document corpus:
        IDF(t) = log((N + 1) / (DF(t) + 1)) + 1, where N=2 and DF(t) is the number of documents containing term t
    4. Multiply TF x IDF to get TF-IDF vectors
    5. Compute cosine similarity: dot(A, B) / (||A|| * ||B||)

    Rare shared words are upweighted, common words are downweighted.
    :text1: first question text
    :text2: second question text
    :remove_stopwords: if True, remove common stop words before computing similarity
    :returns: float in [0, 1]
    """
    tokens1 = _tokenize(text1, remove_stopwords)
    tokens2 = _tokenize(text2, remove_stopwords)
    if not tokens1 and not tokens2:
        return 0.0

    def term_freq(tokens):
        counts = Counter(tokens)
        total = len(tokens)
        return {t: count / total for t, count in counts.items()}

    tf1 = term_freq(tokens1)
    tf2 = term_freq(tokens2)
    vocab = set(tf1) | set(tf2)

    N = 2
    idf = {t: np.log((N + 1) / (1 + (t in tf1) + (t in tf2))) + 1 for t in vocab}

    vec1 = np.array([tf1.get(t, 0.0) * idf[t] for t in vocab])
    vec2 = np.array([tf2.get(t, 0.0) * idf[t] for t in vocab])

    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def lcs_ratio(text1, text2, remove_stopwords=False):
    """
    Longest Common Subsequence (LCS) ratio between two texts.
    LCS is the longest sequence of characters that appear in both strings in the same order (not necessarily contiguous).
    The ratio is LCS length divided by the length of the longer string.

    :text1: first question text
    :text2: second question text
    :remove_stopwords: if True, remove common stop words before computing ratio
    :returns: float in [0, 1]
    """
    tokens1 = _tokenize(text1, remove_stopwords)
    tokens2 = _tokenize(text2, remove_stopwords)
    n, m = len(tokens1), len(tokens2)
    if n == 0 and m == 0:
        return 0.0

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if tokens1[i - 1] == tokens2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return min(dp[n][m] / ((n + m) / 2), 1.0)


def build_graph_features(train_df):
    """
    Build question-frequency and pair-frequency dictionaries from
    the training DataFrame.

    The question graph has one node per unique question string.
    An edge connects q1 and q2 for each pair.
    Node degree = how many times the question appears across all pairs.

    :train_df : training DataFrame (question1, question2 columns)
    :q_freq    : dict {question_string -> frequency in dataset}
    :pair_freq : dict {frozenset({q1, q2}) -> count of this exact pair}
    """
    q_freq = Counter()
    pair_freq = Counter()

    q1_list = cast_list_as_strings(list(train_df["question1"]))
    q2_list = cast_list_as_strings(list(train_df["question2"]))

    for q1, q2 in zip(q1_list, q2_list):
        q_freq[q1] += 1
        q_freq[q2] += 1
        pair_freq[(q1, q2)] += 1
        pair_freq[(q2, q1)] += 1

    return dict(q_freq), dict(pair_freq)


def _graph_features_single(q1, q2, q_freq, pair_freq):
    """
    Compute 6 graph-based features for a single pair.

    Features:
      q1_freq    : how many times q1 appears in training (node degree)
      q2_freq    : how many times q2 appears in training (node degree)
      freq_diff  : |q1_freq - q2_freq| — asymmetry signal
      freq_sum   : q1_freq + q2_freq — total "hubness"
      freq_min   : min(q1_freq, q2_freq) — low = at least one rare question
      pair_count : how many times this exact pair appears (>1 = strong dup)

    Unseen questions (val/test) default to 0.
    """
    f1 = q_freq.get(str(q1), 0)
    f2 = q_freq.get(str(q2), 0)
    pf = pair_freq.get(frozenset([str(q1), str(q2)]), 0)
    return [f1, f2, abs(f1 - f2), f1 + f2, min(f1, f2), pf]


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load once (global)
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")


def embedding_cosine_similarity(text1, text2):
    emb1 = EMBEDDER.encode([str(text1)])[0]
    emb2 = EMBEDDER.encode([str(text2)])[0]
    return float(cosine_similarity([emb1], [emb2])[0][0])


FEATURE_NAMES = [
    # Statistical (6)
    "jaccard",
    "jaccard_no_stopwords",
    "shared_word_ratio",
    "shared_word_ratio_no_stopwords",
    "char_trigram_similarity",
    "length_diff_ratio",
    # NLP (4)
    "tfidf_cosine",
    "tfidf_cosine_no_stopwords",
    "lcs_ratio",
    "lcs_ratio_no_stopwords",
    "embedding_cosine",
    # Graph (6)
    "q1_freq",
    "q2_freq",
    "freq_diff",
    "freq_sum",
    # "freq_min",
    "pair_count",
]


def get_features(df, q_freq, pair_freq):
    """
    Compute the full 16-feature enhanced matrix for a DataFrame.

    Parameters
    ----------
    df        : DataFrame with columns question1, question2
    q_freq    : dict from build_graph_features() — built on TRAIN only
    pair_freq : dict from build_graph_features() — built on TRAIN only

    Returns numpy array of shape (N, 16).
    Column order defined in FEATURE_NAMES.
    """
    q1_list = cast_list_as_strings(list(df["question1"]))
    q2_list = cast_list_as_strings(list(df["question2"]))
    n = len(q1_list)
    features = np.zeros((n, len(FEATURE_NAMES)))

    for i, (q1, q2) in enumerate(zip(q1_list, q2_list)):
        # Statistical
        features[i, 0] = jaccard_similarity(q1, q2, remove_stopwords=False)
        features[i, 1] = jaccard_similarity(q1, q2, remove_stopwords=True)
        features[i, 2] = shared_word_ratio(q1, q2, remove_stopwords=False)
        features[i, 3] = shared_word_ratio(q1, q2, remove_stopwords=True)
        features[i, 4] = char_ngram_similarity(q1, q2, n=3)
        features[i, 5] = length_difference_ratio(q1, q2)
        # NLP
        features[i, 6] = tfidf_cosine_similarity(q1, q2, remove_stopwords=False)
        features[i, 7] = tfidf_cosine_similarity(q1, q2, remove_stopwords=True)
        features[i, 8] = lcs_ratio(q1, q2, remove_stopwords=False)
        features[i, 9] = lcs_ratio(q1, q2, remove_stopwords=True)
        features[i, 10] = embedding_cosine_similarity(q1, q2)
        # Graph
        features[i, 11:] = _graph_features_single(q1, q2, q_freq, pair_freq)

    return features


####################################################################
# Evaluation
####################################################################


def evaluate_model(clf, X, y, split_name, use_proba=True):
    from sklearn.metrics import roc_auc_score, precision_score, recall_score

    y_pred = clf.predict(X)
    if use_proba and hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X)[:, 1]
    else:
        y_score = y_pred

    return {
        "split": split_name,
        "roc_auc": round(roc_auc_score(y, y_score), 4),
        "precision": round(precision_score(y, y_pred), 4),
        "recall": round(recall_score(y, y_pred), 4),
    }
