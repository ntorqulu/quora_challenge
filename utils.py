##################################
# Utils functions
##################################
import scipy.sparse
import numpy as np


def cast_list_as_strings(mylist):
    """
    Return a list of strings given a list
    """
    mylist_of_strings = []
    for x in mylist:
        mylist_of_strings.append(str(x))

    return mylist_of_strings


def get_features_from_df(df, count_vectorizer):
    """
    Return the features of a dataframe given a count vectorizer
    """
    q1_casted = cast_list_as_strings(list(df["question1"]))
    q2_casted = cast_list_as_strings(list(df["question2"]))

    X_q1 = count_vectorizer.transform(q1_casted)
    X_q2 = count_vectorizer.transform(q2_casted)
    X_q1q2 = scipy.sparse.hstack([X_q1, X_q2])

    return X_q1q2


def get_mistakes(clf, X_q1q2, y):
    """
    Return the mistakes of a classifier given the features and the labels
    """
    predictions = clf.predict(X_q1q2)
    incorrect_predictions = predictions != y
    incorrect_indices = np.where(incorrect_predictions)

    if np.sum(incorrect_predictions) == 0:
        print("No mistakes found.")
    else:
        return incorrect_indices, predictions


def print_mistake_k(train_df, k, mistake_indices, predictions):
    """
    Print the k first mistakes of a classifier given the mistake indices and the predictions
    """
    print(train_df.iloc[mistake_indices[k]].question1)
    print(train_df.iloc[mistake_indices[k]].question2)
    print("True label:", train_df.iloc[mistake_indices[k]].is_duplicate)
    print("Predicted label:", predictions[mistake_indices[k]])
