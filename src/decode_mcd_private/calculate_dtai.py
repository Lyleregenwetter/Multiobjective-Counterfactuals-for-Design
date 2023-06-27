import numpy as np


def calculateDTAI(actual_performance,
                  direction,
                  targets,
                  alpha_values,
                  beta_values,
                  smallest_allowed=1e-7):
    if actual_performance.shape[1] == 0:
        return np.zeros(actual_performance.shape[0])
    alpha_values, beta_values, actual_performance = convert_to_float_arrays(alpha_values,
                                                                            beta_values,
                                                                            actual_performance)
    actual = ensure_greater_than_smallest(actual_performance, smallest_allowed)

    scores = calculate_scores(alpha_values, beta_values, calculate_ratios(direction, targets, actual))

    scores_sum = np.sum(scores, axis=1)
    sum_s_max = np.sum(alpha_values / beta_values)
    sum_s_min = -np.sum(alpha_values)

    scores_sum_scaled = (scores_sum - sum_s_min) / (sum_s_max - sum_s_min)
    return scores_sum_scaled


def calculate_scores(alpha_values, beta_values, ratios):
    case_less_or_equal_to_one = np.multiply(alpha_values, ratios) - alpha_values
    alpha_over_beta = np.divide(alpha_values, beta_values)
    exponential = np.exp(np.multiply(beta_values, (1 - ratios)))
    case_greater_or_equal_to_one = np.multiply(alpha_over_beta, (1 - exponential))
    greater_than_one = np.greater(ratios, 1)
    greater_than_one = greater_than_one.astype("float32")
    scores = np.multiply(case_greater_or_equal_to_one, greater_than_one) \
             + np.multiply(case_less_or_equal_to_one, (1 - greater_than_one))
    return scores


def ensure_greater_than_smallest(actual_performance, tolerance):
    return np.maximum(actual_performance, tolerance)


def calculate_ratios(direction, targets, y):
    if direction == "maximize":
        return np.divide(y, targets)
    if direction == "minimize":
        return np.divide(targets, y)
    raise Exception("Unknown optimization direction, expected maximize or minimize")


def convert_to_float_arrays(alpha_values, beta_values, y_eval):
    alpha_values = alpha_values.astype("float32")
    beta_values = beta_values.astype("float32")
    y_eval = y_eval.astype("float32")
    return alpha_values, beta_values, y_eval
