import numpy as np
import util
import sys

### NOTE : You need to complete logreg implementation first!
from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'

def main_posonly(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    
    NOTE: You need to complete logreg implementation first (see class above)!!!
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    plot_path = save_path.replace('.txt', '.png')
    plot_path_true = plot_path.replace(WILDCARD, 'true')
    plot_path_naive = plot_path.replace(WILDCARD, 'naive')
    plot_path_adjusted = plot_path.replace(WILDCARD, 'adjusted')

    # Problem (2a): Train and test on true labels (t)
    full_predictions = fully_observed_predictions(train_path, test_path, output_path_true, plot_path_true)

    # Problem (2b): Train on y-labels and test on true labels
    naive_predictions, clf = naive_partial_labels_predictions(train_path, test_path, output_path_naive, plot_path_naive)

    # Problem (2f): Apply correction factor using validation set and test on true labels
    alpha = find_alpha_and_plot_correction(clf, valid_path, test_path, output_path_adjusted, plot_path_adjusted, naive_predictions)

    return

def fully_observed_predictions(train_path, test_path, output_path_true, plot_path_true):
    """
    Problem (2a): Fully Observable Binary Classification Helper Function

    Args:
        train_path: Path to CSV file containing dataset for training.
        test_path: Path to CSV file containing dataset for testing.
        output_path_true: Path to save observed predictions
        plot_path_true: Path to save the plot using plot_posonly util function
    Return:
        full_predictions: tensor of predictions returned from applied LogReg classifier prediction
    """
    full_predictions = None
    # Problem (2a): Train and test on true labels (t)
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    # *** START CODE HERE ***
    # *** END CODE HERE ***
    return full_predictions

def naive_partial_labels_predictions(train_path, test_path, output_path_naive, plot_path_naive):
    """
    Problem (2b): Naive Partial Labels Binary Classification Helper Function

    Args:
        train_path: Path to CSV file containing dataset for training.
        test_path: Path to CSV file containing dataset for testing.
        output_path_naive: Path to save observed predictions
        plot_path_naive: Path to save the plot using plot_posonly util function
    Return:
        naive_predictions: tensor of predictions returned from applied LogReg prediction
        clf: Logistic Regression classifier (will be reused for 2f)
    """
    naive_predictions = None
    clf = None
    # Problem (2b): Train on y-labels and test on true labels
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    # *** START CODE HERE ***
    # *** END CODE HERE ***
    return naive_predictions, clf

def find_alpha_and_plot_correction(clf, valid_path, test_path, output_path_adjusted, plot_path_adjusted, naive_predictions):
    """
    Problem (2f): Alpha Correction Binary Classification Helper Function

    Args:
        clf: Logistic regression classifier from part 2b
        valid_path: Path to CSV file containing dataset for validation.
        test_path: Path to CSV file containing dataset for testing.
        output_path_adjusted: Path to save observed predictions
        plot_path_adjusted: Path to save the plot using plot_posonly util function
        naive_predictions: tensor of predictions returned from applied LogReg prediction from 2b
    Return:
        alpha: corrected alpha value
    """
    alpha = None
    # Problem (2f): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to output_path_adjusted
    # *** START CODE HERE ***
    # *** END CODE HERE ***
    return alpha

if __name__ == "__main__":
    '''
    Start of Problem 2: Incomplete, Positive-Only Labels
    '''
    main_posonly(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')