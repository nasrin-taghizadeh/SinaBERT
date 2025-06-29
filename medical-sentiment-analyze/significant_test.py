import json

import pandas as pd
from statsmodels.stats.contingency_tables import cochrans_q
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def load_sentiment_data():
    with open("sinabert_pred_2025.json", encoding='utf-8') as f:
        sina_results = json.load(f)

    with open("parsbert_pred_2025.json", "r", encoding="utf-8") as f:
        pars_results = json.load(f)

    main_data = pd.read_csv('medical-sentiment.csv', encoding='utf-8')
    main_data = main_data.to_dict(orient='list')
    # Output: {'comment': ['دکتر خوبی بود', 'خیلی بد بود'], 'sentiment': ['satisfied', 'unsatisfied']}

    all_labels = {}
    for sent, slabel in zip(main_data['comment'] , main_data['sentiment']):
        all_labels[sent] = slabel

    true_labels = []
    for sent in sina_results.keys():
        true_labels.append(all_labels[sent])

    labels = set(sina_results.values())
    print(labels)
    # {'no-idea', 'satisfied', 'unsatisfied'}


    return true_labels, list(sina_results.values()), list(pars_results.values())

# def sample_data():
#     # Example data: 3 treatments (A, B, C) applied to the same subjects,
#     # with outcomes 0 (failure) or 1 (success)
#     data = {
#         'Subject': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#         'Treatment_A': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
#         'Treatment_B': [1, 1, 1, 0, 0, 1, 0, 0, 1, 1],
#         'Treatment_C': [0, 1, 1, 1, 0, 0, 1, 1, 0, 0]
#     }
#     df = pd.DataFrame(data)
#
#     # Extract the treatment columns for Cochran's Q
#     treatments = df[['Treatment_A', 'Treatment_B', 'Treatment_C']]
#     return treatments
#
#
# def run_mcnemar_test(treatments):
#     # Perform Cochran's Q test
#     result = cochrans_q(treatments)
#
#     print(f"Cochran's Q statistic: {result.statistic:.3f}")
#     print(f"p-value: {result.pvalue:.3f}")
#
#     if result.pvalue < 0.05:
#         print("There is a significant difference among the treatments.")
#     else:
#         print("There is no significant difference among the treatments.")




def compare_classifiers_mcnemar_multiclass(true_labels, predictions_clf1, predictions_clf2):
    """
    Compares two classifiers with multi-class output using a McNemar-like approach
    by focusing on correct vs. incorrect predictions.

    Args:
        true_labels (np.array): Array of true class labels.
        predictions_clf1 (np.array): Array of predictions from classifier 1.
        predictions_clf2 (np.array): Array of predictions from classifier 2.

    Returns:
        tuple: A tuple containing the chi-squared statistic and p-value from McNemar's test.
    """

    # Determine correct/incorrect predictions for each classifier
    correct_clf1 = (predictions_clf1 == true_labels)
    correct_clf2 = (predictions_clf2 == true_labels)

    # Build the 2x2 contingency table for McNemar's test
    # a: both correct
    # b: clf1 correct, clf2 incorrect
    # c: clf1 incorrect, clf2 correct
    # d: both incorrect

    a = np.sum(correct_clf1 & correct_clf2)
    b = np.sum(correct_clf1 & ~correct_clf2)
    c = np.sum(~correct_clf1 & correct_clf2)
    d = np.sum(~correct_clf1 & ~correct_clf2)

    contingency_table = np.array([[a, b],
                                  [c, d]])

    # Perform McNemar's test
    result = mcnemar(contingency_table, exact=True)

    return result.statistic, result.pvalue

def example_usage():
    # Simulate true labels and predictions for 3 classes
    np.random.seed(42)
    n_samples = 100
    true_labels = np.random.randint(0, 3, n_samples) # 0, 1, or 2

    # Simulate predictions for two classifiers
    # Classifier 1 is slightly better
    predictions_clf1 = np.copy(true_labels)
    errors_clf1_indices = np.random.choice(n_samples, size=int(n_samples * 0.2), replace=False)
    for idx in errors_clf1_indices:
        predictions_clf1[idx] = (predictions_clf1[idx] + np.random.randint(1, 3)) % 3

    # Classifier 2 has more errors
    predictions_clf2 = np.copy(true_labels)
    errors_clf2_indices = np.random.choice(n_samples, size=int(n_samples * 0.3), replace=False)
    for idx in errors_clf2_indices:
        predictions_clf2[idx] = (predictions_clf2[idx] + np.random.randint(1, 3)) % 3

    chi2_stat, p_value = compare_classifiers_mcnemar_multiclass(true_labels, predictions_clf1, predictions_clf2)

    print(f"McNemar's Chi-squared statistic: {chi2_stat:.3f}")
    print(f"McNemar's p-value: {p_value:.3f}")

    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference in the overall error rates of the two classifiers.")
    else:
        print("Fail to reject the null hypothesis: No significant difference in the overall error rates of the two classifiers.")


if __name__=="__main__":
    true_labels, sina_labels, pars_labels = load_sentiment_data()
    chi2_stat, p_value = compare_classifiers_mcnemar_multiclass(true_labels, sina_labels, pars_labels)

    print(f"McNemar's Chi-squared statistic: {chi2_stat:.3f}")
    print(f"McNemar's p-value: {p_value:.3f}")

    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference in the overall error rates of the two classifiers.")
    else:
        print("Fail to reject the null hypothesis: No significant difference in the overall error rates of the two classifiers.")

