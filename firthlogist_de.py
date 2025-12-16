
import warnings
from copy import deepcopy
from math import sqrt
import numpy as np
from scipy.linalg import lapack
from scipy.special import expit
from scipy.stats import chi2, norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from tabulate import tabulate

class FirthLogisticRegression(BaseEstimator, ClassifierMixin):
    # ... [保留原有的 __init__ 和其他方法] ...

    def _validate_input(self, X, y):
        if self.max_iter < 0:
            raise ValueError(
                f"Maximum number of iterations must be positive; "
                f"got max_iter={self.max_iter}"
            )
        if self.max_halfstep < 0:
            raise ValueError(
                f"Maximum number of step-halvings must >= 0; "
                f"got max_halfstep={self.max_iter}"
            )
        if self.tol < 0:
            raise ValueError(
                f"Tolerance for stopping criteria must be positive; got tol={self.tol}"
            )

        # 手动验证和转换数据（替代 self._validate_data）
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        # 确保最小样本数
        if X.shape[0] < 2:
            raise ValueError("Number of samples must be at least 2.")

        check_classification_targets(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"Got {len(self.classes_)} - only 2 classes supported.")

        y = LabelEncoder().fit_transform(y).astype(X.dtype, copy=False)
        return X, y

    def summary(self, xname=None, tablefmt="simple"):
        # 替换 check_is_fitted 为自定义检查
        if not hasattr(self, "coef_"):
            raise ValueError("Model is not fitted yet.")
        # ... [其余代码不变] ...

    def decision_function(self, X):
        # 替换 check_is_fitted 为自定义检查
        if not hasattr(self, "coef_"):
            raise ValueError("Model is not fitted yet.")

        # 手动转换 X（替代 self._validate_data）
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != len(self.coef_):
            raise ValueError(
                f"X has {X.shape[1]} features, but model expects {len(self.coef_)}."
            )

        scores = X @ self.coef_ + self.intercept_
        return scores

    # ... [其余方法保持不变] ...