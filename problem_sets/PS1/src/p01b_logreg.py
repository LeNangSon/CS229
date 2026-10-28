import numpy as np
import util

from linear_model import LinearModel

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_eval, y_val = util.load_dataset(eval_path, add_intercept=True)
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    res = clf.predict(x_eval)
    print(sum(res == y_val) / len(y_val))
    np.savetxt(pred_path, res, delimiter=",")
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        theta = np.zeros(x.shape[1])

        epsilon = 1e-5
        diff = 10
        while(diff > epsilon):
            old_theta = theta.copy()
            y_hat = sigmoid(x @ theta)
            W = np.diag(y_hat * (1 - y_hat))
            theta -= np.linalg.inv(x.T @ W @ x) @ x.T @ (y_hat - y)
            diff = np.linalg.norm(theta - old_theta)
        self.weight = theta
        return self
        # *** END CODE HERE ***
    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return x @ self.weight >=0
        # *** END CODE HERE ***

if __name__ == "__main__":
    main("/Volumes/DEV/Stanford Course/CS229 Autumn 2018 Machine Learning/problem_sets/PS1/data/ds1_train.csv", "/Volumes/DEV/Stanford Course/CS229 Autumn 2018 Machine Learning/problem_sets/PS1/data/ds1_valid.csv", "/Volumes/DEV/Stanford Course/CS229 Autumn 2018 Machine Learning/problem_sets/PS1/data/ds1_res.csv")