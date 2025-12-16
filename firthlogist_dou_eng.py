import warnings
from copy import deepcopy
from importlib.resources import open_text
from math import sqrt
import numpy as np
from scipy.linalg import lapack
from scipy.special import expit
from scipy.stats import chi2, norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

# Remove tabulate import (no longer needed)


class FirthLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic regression with Firth's bias reduction method.
    
    This implementation resolves separation issues in standard logistic regression and 
    is compatible with older scikit-learn versions (no dependency on `_validate_data`).
    Based on the `logistf` R package and Heinze & Schemper (2002).

    Parameters
    ----------
    max_iter : int, default=25
        Maximum number of Newton-Raphson iterations for model fitting.
    max_halfstep : int, default=0
        Maximum number of step-halvings per Newton-Raphson iteration (for stability).
    max_stepsize : float, default=5
        Maximum step size for coefficient updates (prevents extreme values).
    pl_max_iter : int, default=100
        Maximum iterations for profile likelihood confidence interval (CI) calculation.
    pl_max_halfstep : int, default=0
        Maximum step-halvings for profile likelihood CI calculation.
    pl_max_stepsize : float, default=5
        Maximum step size for profile likelihood CI updates.
    tol : float, default=0.0001
        Convergence tolerance (stop if coefficient change < tol).
    fit_intercept : bool, default=True
        Whether to include an intercept term in the model.
    skip_pvals : bool, default=False
        If True, skip p-value calculation (speeds up fitting).
    skip_ci : bool, default=False
        If True, skip confidence interval calculation (speeds up fitting).
    alpha : float, default=0.05
        Significance level for confidence intervals (1-alpha = confidence level).
    wald : bool, default=False
        If True, use Wald method for p-values/CI (faster but less robust than profile likelihood).
    test_vars : int/list/None, default=None
        Indices of variables to calculate p-values/CI for. None = all variables.

    Attributes
    ----------
    bse_ : ndarray of shape (n_features + [1 if fit_intercept else 0],)
        Standard errors of the coefficients (including intercept if fitted).
    classes_ : ndarray of shape (2,)
        Unique class labels (binary classification only).
    ci_ : ndarray of shape (n_params, 2)
        Profile likelihood/Wald confidence intervals (lower, upper) for each parameter.
    coef_ : ndarray of shape (n_features,)
        Firth-corrected coefficients for the features.
    intercept_ : float
        Firth-corrected intercept term (0 if fit_intercept=False).
    loglik_ : float
        Penalized log-likelihood of the fitted model.
    n_iter_ : int
        Number of Newton-Raphson iterations completed.
    pvals_ : ndarray of shape (n_params,)
        p-values for coefficient significance tests.

    References
    ----------
    Firth D (1993). Bias reduction of maximum likelihood estimates. Biometrika 80, 27–38.
    Heinze G, Schemper M (2002). A solution to the problem of separation in logistic regression.
    Statistics in Medicine 21: 2409-2419.
    """

    def __init__(
        self,
        max_iter=25,
        max_halfstep=0,
        max_stepsize=5,
        pl_max_iter=100,
        pl_max_halfstep=0,
        pl_max_stepsize=5,
        tol=0.0001,
        fit_intercept=True,
        skip_pvals=False,
        skip_ci=False,
        alpha=0.05,
        wald=False,
        test_vars=None,
    ):
        self.max_iter = max_iter
        self.max_stepsize = max_stepsize
        self.max_halfstep = max_halfstep
        self.pl_max_iter = pl_max_iter
        self.pl_max_halfstep = pl_max_halfstep
        self.pl_max_stepsize = pl_max_stepsize
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.skip_pvals = skip_pvals
        self.skip_ci = skip_ci
        self.alpha = alpha
        self.wald = wald
        self.test_vars = test_vars

    def _more_tags(self):
        """Tag for scikit-learn compatibility (binary classification only)."""
        return {"binary_only": True}

    def _validate_input(self, X, y):
        """
        Manually validate and preprocess input data (replaces sklearn's _validate_data).
        Ensures compatibility with older scikit-learn versions without modifying packages.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix (raw input: list, numpy array, pandas DataFrame).
        y : array-like of shape (n_samples,)
            Target vector (binary labels: 0/1).

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Validated float64 feature matrix.
        y : ndarray of shape (n_samples,)
            Validated float64 target vector (encoded to 0/1).

        Raises
        ------
        ValueError
            If input data is invalid (empty, wrong dimensions, mismatched samples, non-binary labels).
        """
        # 1. Check for empty input
        if X is None or y is None:
            raise ValueError("X and y must not be None (empty input)")
        
        # 2. Convert to numpy float64 array (supports list/DataFrame input)
        try:
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to convert X/y to float64 array: {str(e)}")
        
        # 3. Validate dimensions (X=2D, y=1D)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got {X.ndim}D")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D (n_samples,), got {y.ndim}D")
        
        # 4. Ensure sample count matches between X and y
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Mismatched sample count: X has {X.shape[0]} samples, y has {y.shape[0]}"
            )
        
        # 5. Ensure minimum sample size (at least 2 samples required)
        if X.shape[0] < 2:
            raise ValueError(f"At least 2 samples required, got {X.shape[0]}")
        
        # 6. Validate hyperparameters (non-negative values)
        if self.max_iter < 0:
            raise ValueError(f"max_iter must be positive (got {self.max_iter})")
        if self.max_halfstep < 0:
            raise ValueError(f"max_halfstep must be ≥ 0 (got {self.max_halfstep})")
        if self.tol < 0:
            raise ValueError(f"tol must be positive (got {self.tol})")
        
        # 7. Validate classification targets (binary only)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        
        if len(self.classes_) != 2:
            raise ValueError(f"Only binary classification supported (got {len(self.classes_)} classes)")
        
        # 8. Encode labels to 0/1 (compatible with logistic regression)
        y = LabelEncoder().fit_transform(y).astype(X.dtype, copy=False)

        return X, y

    def fit(self, X, y):
        """
        Fit Firth's bias-reduced logistic regression model to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Training binary target vector (0/1).

        Returns
        -------
        self : FirthLogisticRegression
            Fitted model instance (supports chaining: fit().predict()).

        Raises
        ------
        ConvergenceWarning
            If Newton-Raphson iterations fail to converge.
        """
        # Validate input data (replace _validate_data)
        X, y = self._validate_input(X, y)
        
        # Add intercept column if enabled (append column of 1s to X)
        if self.fit_intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        # Fit model via Newton-Raphson (Firth's method)
        self.coef_, self.loglik_, self.n_iter_ = _firth_newton_raphson(
            X, y, self.max_iter, self.max_stepsize, self.max_halfstep, self.tol
        )

        # Calculate standard errors (SE) of coefficients
        self.bse_ = _bse(X, self.coef_)

        # Calculate confidence intervals (CI)
        if not self.skip_ci:
            if not self.wald:
                # Profile likelihood CI (more robust for small samples/separation)
                self.ci_ = _profile_likelihood_ci(
                    X=X,
                    y=y,
                    fitted_coef=self.coef_,
                    full_loglik=self.loglik_,
                    max_iter=self.pl_max_iter,
                    max_stepsize=self.pl_max_stepsize,
                    max_halfstep=self.pl_max_halfstep,
                    tol=self.tol,
                    alpha=self.alpha,
                    test_vars=self.test_vars,
                )
            else:
                # Wald CI (faster but less robust)
                self.ci_ = _wald_ci(self.coef_, self.bse_, self.alpha)
        else:
            # Fill CI with NaN if skipped
            self.ci_ = np.full((self.coef_.shape[0], 2), np.nan)

        # Calculate p-values (significance tests)
        if not self.skip_pvals:
            if not self.wald:
                # Penalized likelihood ratio test (LRT)
                self.pvals_ = _penalized_lrt(
                    self.loglik_,
                    X,
                    y,
                    self.max_iter,
                    self.max_stepsize,
                    self.max_halfstep,
                    self.tol,
                    self.test_vars,
                )
            else:
                # Wald test (faster but less robust)
                self.pvals_ = _wald_test(self.coef_, self.bse_)
        else:
            # Fill p-values with NaN if skipped
            self.pvals_ = np.full(self.coef_.shape[0], np.nan)

        # Split intercept from coefficients (for scikit-learn compatibility)
        if self.fit_intercept:
            self.intercept_ = self.coef_[-1]
            self.coef_ = self.coef_[:-1]
        else:
            self.intercept_ = 0.0

        return self

    def summary(self, xname=None):
        """
        Print model summary (replaces tabulate with pure Python string formatting).
        Displays coefficients, standard errors, confidence intervals, and p-values.

        Parameters
        ----------
        xname : list of str, optional
            Custom names for features (must match number of features). 
            Default: ["x1", "x2", ..., "Intercept"].

        Returns
        -------
        None (prints summary to console)
        """
        check_is_fitted(self)  # Ensure model is fitted
        
        # Set default feature names if not provided
        if xname and len(xname) != len(self.coef_):
            raise ValueError(
                f"xname length ({len(xname)}) must match number of features ({len(self.coef_)})"
            )
        if not xname:
            xname = [f"x{i}" for i in range(1, len(self.coef_) + 1)]

        # Combine coefficients + intercept (if fitted)
        coef_vals = list(self.coef_)
        if self.fit_intercept:
            xname.append("Intercept")
            coef_vals.append(self.intercept_)

        # Prepare summary data (align columns for readability)
        header = (
            f"{'Variable':<10} {'Coef':<10} {'Std Err':<10} "
            f"[{self.alpha/2:.3f}    {1-self.alpha/2:.3f}]    {'P-value':<10}"
        )
        separator = "-" * len(header)
        rows = []

        # Build each row of the summary table
        for i, var_name in enumerate(xname):
            coef = coef_vals[i]
            se = self.bse_[i]
            ci_lower = self.ci_[i, 0]
            ci_upper = self.ci_[i, 1]
            pval = self.pvals_[i]

            # Format values (handle NaN for skipped CI/p-values)
            coef_str = f"{coef:.4f}" if not np.isnan(coef) else "NaN"
            se_str = f"{se:.4f}" if not np.isnan(se) else "NaN"
            ci_str = f"{ci_lower:.4f}    {ci_upper:.4f}" if not (np.isnan(ci_lower) or np.isnan(ci_upper)) else "NaN    NaN"
            pval_str = f"{pval:.4f}" if not np.isnan(pval) else "NaN"

            # Align columns (fixed width for readability)
            row = f"{var_name:<10} {coef_str:<10} {se_str:<10} {ci_str:<20} {pval_str:<10}"
            rows.append(row)

        # Print summary (no tabulate required)
        print("\n=== Firth Logistic Regression Summary ===")
        print(header)
        print(separator)
        for row in rows:
            print(row)
        
        # Print additional model info
        print(f"\nModel Metrics:")
        print(f"  Penalized Log-Likelihood: {self.loglik_:.4f}")
        print(f"  Newton-Raphson Iterations: {self.n_iter_}")
        print(f"  Confidence Level: {100*(1-self.alpha):.1f}%")
        print(f"  CI Method: {'Wald' if self.wald else 'Profile Likelihood'}")
        
        # Restore xname (remove intercept if added)
        if self.fit_intercept:
            xname.pop()

    def decision_function(self, X):
        """
        Compute linear decision function (z = X @ coef + intercept).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix for prediction.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Linear decision scores (higher = higher probability of class 1).
        """
        check_is_fitted(self)
        
        # Validate prediction input (consistent with training)
        try:
            X = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to convert X to float64 array: {str(e)}")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got {X.ndim}D")
        
        # Compute linear scores
        scores = X @ self.coef_ + self.intercept_
        return scores

    def predict(self, X):
        """
        Predict binary class labels (0/1) for input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix for prediction.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        decision = self.decision_function(X)
        # Threshold at 0 (sigmoid(0) = 0.5)
        indices = (decision > 0).astype(int)
        return self.classes_[indices]

    def predict_proba(self, X):
        """
        Predict class probabilities (0 and 1) for input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix for prediction.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Probability of class 0 (column 0) and class 1 (column 1).
        """
        decision = self.decision_function(X)
        # Convert linear scores to probabilities via sigmoid
        proba_1 = expit(decision)
        proba_0 = 1 - proba_1
        return np.column_stack([proba_0, proba_1])


# ------------------------------
# Core Helper Functions (Firth's Method)
# ------------------------------
def _firth_newton_raphson(X, y, max_iter, max_stepsize, max_halfstep, tol, mask=None):
    """
    Newton-Raphson optimization for Firth's penalized likelihood.
    Implements step-halving and step-size limits for stability.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_params)
        Feature matrix (including intercept if fitted).
    y : ndarray of shape (n_samples,)
        Binary target vector (0/1).
    max_iter : int
        Maximum Newton-Raphson iterations.
    max_stepsize : float
        Maximum step size for coefficient updates.
    max_halfstep : int
        Maximum step-halvings per iteration.
    tol : float
        Convergence tolerance (coefficient change < tol).
    mask : int/list/None
        Indices of coefficients to mask (for null model in LRT).

    Returns
    -------
    coef : ndarray of shape (n_params,)
        Fitted Firth-corrected coefficients.
    loglik : float
        Penalized log-likelihood of the fitted model.
    n_iter : int
        Number of iterations completed.

    Raises
    ------
    ConvergenceWarning
        If step-halving fails or iterations are exhausted.
    """
    # Initialize coefficients to 0
    coef = np.zeros(X.shape[1])
    
    for iter in range(1, max_iter + 1):
        # Compute predicted probabilities (sigmoid)
        preds = expit(X @ coef)
        
        # Weighted feature matrix (X * sqrt(p*(1-p)))
        XW = _get_XW(X, preds, mask)

        # Fisher information matrix (XW^T @ XW)
        fisher_info_mtx = XW.T @ XW
        
        # Hat matrix diagonal (leverage scores)
        hat = _hat_diag(XW)
        
        # Firth's corrected score vector (U*)
        U_star = np.matmul(X.T, y - preds + np.multiply(hat, 0.5 - preds))
        
        # Compute update step (solve Fisher @ step = U*)
        step_size = np.linalg.lstsq(fisher_info_mtx, U_star, rcond=None)[0]

        # Limit step size to max_stepsize (prevent extreme updates)
        mx = np.max(np.abs(step_size)) / max_stepsize
        if mx > 1:
            step_size = step_size / mx
        
        # Update coefficients
        coef_new = coef + step_size
        preds_new = expit(X @ coef_new)
        
        # Compute penalized log-likelihood (current and new)
        loglike = _loglikelihood(X, y, preds)
        loglike_new = _loglikelihood(X, y, preds_new)
        steps = 0

        # Step-halving (ensure log-likelihood increases)
        while loglike < loglike_new:
            step_size *= 0.5
            coef_new = coef + step_size
            preds_new = expit(X @ coef_new)
            loglike_new = _loglikelihood(X, y, preds_new)
            steps += 1
            
            # Step-halving failed (max halfsteps reached)
            if steps == max_halfstep:
                warnings.warn(
                    "Step-halving failed to converge (max halfsteps reached)",
                    ConvergenceWarning,
                    stacklevel=2
                )
                return coef_new, -loglike_new, iter

        # Check convergence (coefficient change < tol)
        if iter > 1 and np.linalg.norm(coef_new - coef) < tol:
            return coef_new, -loglike_new, iter

        # Update coefficients for next iteration
        coef += step_size

    # Iterations exhausted (convergence failed)
    warnings.warn(
        f"Newton-Raphson failed to converge (max_iter={max_iter} reached). Try increasing max_iter.",
        ConvergenceWarning,
        stacklevel=2
    )
    return coef, -loglike_new, max_iter


def _loglikelihood(X, y, preds):
    """
    Compute Firth's penalized log-likelihood.
    Formula: logL = logL_standard + 0.5 * log(det(Fisher))

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_params)
        Feature matrix (including intercept if fitted).
    y : ndarray of shape (n_samples,)
        Binary target vector (0/1).
    preds : ndarray of shape (n_samples,)
        Predicted probabilities (sigmoid(X @ coef)).

    Returns
    -------
    penalized_loglik : float
        Negative penalized log-likelihood (for minimization).
    """
    # Weighted feature matrix
    XW = _get_XW(X, preds)
    
    # Fisher information matrix (regularized to avoid singular determinant)
    fisher_info_mtx = XW.T @ XW
    fisher_info_mtx += 1e-10 * np.eye(fisher_info_mtx.shape[0])  # Numerical stability
    
    # Penalty term (0.5 * log(determinant of Fisher matrix))
    penalty = 0.5 * np.log(np.linalg.det(fisher_info_mtx))
    
    # Clip predictions to avoid log(0) (numerical stability)
    preds_clipped = np.clip(preds, 1e-15, 1 - 1e-15)
    
    # Standard logistic regression log-likelihood
    standard_loglik = np.sum(y * np.log(preds_clipped) + (1 - y) * np.log(1 - preds_clipped))
    
    # Penalized log-likelihood (negative for minimization)
    return -1 * (standard_loglik + penalty)


def _get_XW(X, preds, mask=None):
    """
    Compute weighted feature matrix (XW = X * sqrt(p*(1-p))).
    Applies mask (zero out specified columns) for null model fitting.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_params)
        Feature matrix (including intercept if fitted).
    preds : ndarray of shape (n_samples,)
        Predicted probabilities (sigmoid(X @ coef)).
    mask : int/list/None
        Indices of columns to mask (zero out).

    Returns
    -------
    XW : ndarray of shape (n_samples, n_params)
        Weighted feature matrix.
    """
    # Weight vector (sqrt(p*(1-p)))
    rootW = np.sqrt(preds * (1 - preds))
    
    # Weighted X matrix
    XW = rootW[:, np.newaxis] * X
    
    # Apply mask (zero out specified columns)
    if mask is not None:
        XW[:, mask] = 0.0
    
    return XW


def _get_aug_XW(X, preds, hats):
    """
    Compute augmented weighted feature matrix for profile likelihood CI.
    Formula: XW_aug = X * sqrt(p*(1-p)*(1+hat))

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_params)
        Feature matrix (including intercept if fitted).
    preds : ndarray of shape (n_samples,)
        Predicted probabilities (sigmoid(X @ coef)).
    hats : ndarray of shape (n_samples,)
        Diagonal elements of the hat matrix.

    Returns
    -------
    XW_aug : ndarray of shape (n_samples, n_params)
        Augmented weighted feature matrix.
    """
    rootW = np.sqrt(preds * (1 - preds) * (1 + hats))
    XW_aug = rootW[:, np.newaxis] * X
    return XW_aug


def _hat_diag(XW):
    """
    Compute diagonal elements of the hat matrix (leverage scores) via QR decomposition.
    Hat matrix: H = XW (XW^T XW)^-1 XW^T → diagonal elements = sum(Q^2) where Q is orthogonal.

    Parameters
    ----------
    XW : ndarray of shape (n_samples, n_params)
        Weighted feature matrix.

    Returns
    -------
    hat : ndarray of shape (n_samples,)
        Diagonal elements of the hat matrix.
    """
    # QR decomposition (scipy lapack interface for efficiency)
    qr, tau, _, _ = lapack.dgeqrf(XW, overwrite_a=True)
    Q, _, _ = lapack.dorgqr(qr, tau, overwrite_a=True)
    
    # Diagonal elements = sum of squares of Q rows
    hat = np.einsum("ij,ij->i", Q, Q)
    return hat


def _bse(X, coefs):
    """
    Compute standard errors (SE) of coefficients (sqrt of covariance matrix diagonal).
    Covariance matrix = inverse of Fisher information matrix (regularized for stability).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_params)
        Feature matrix (including intercept if fitted).
    coefs : ndarray of shape (n_params,)
        Fitted Firth-corrected coefficients.

    Returns
    -------
    bse : ndarray of shape (n_params,)
        Standard errors of the coefficients.
    """
    # Predicted probabilities (sigmoid)
    preds = expit(X @ coefs)
    
    # Weighted feature matrix
    XW = _get_XW(X, preds)
    
    # Fisher information matrix (regularized to avoid singularity)
    fisher_info_mtx = XW.T @ XW
    fisher_info_mtx += 1e-10 * np.eye(fisher_info_mtx.shape[0])
    
    # Covariance matrix (pseudo-inverse for numerical stability)
    cov_matrix = np.linalg.pinv(fisher_info_mtx)
    
    # Standard errors = sqrt(variance) (diagonal of covariance matrix)
    return np.sqrt(np.diag(cov_matrix))


def _penalized_lrt(
    full_loglik, X, y, max_iter, max_stepsize, max_halfstep, tol, test_vars
):
    """
    Penalized likelihood ratio test (LRT) for coefficient significance.
    Compares full model (all coefficients) vs null model (masked coefficients).

    Parameters
    ----------
    full_loglik : float
        Penalized log-likelihood of the full model.
    X : ndarray of shape (n_samples, n_params)
        Feature matrix (including intercept if fitted).
    y : ndarray of shape (n_samples,)
        Binary target vector (0/1).
    max_iter/max_stepsize/max_halfstep/tol : float/int
        Newton-Raphson hyperparameters (same as full model).
    test_vars : int/list/None
        Indices of variables to test (None = all variables).

    Returns
    -------
    pvals : ndarray of shape (n_params,)
        p-values for each tested variable (NaN for untested variables).
    """
    # Determine variables to test
    if test_vars is None:
        test_var_indices = range(X.shape[1])
    elif isinstance(test_vars, int):
        test_var_indices = [test_vars]
    else:
        test_var_indices = sorted(test_vars)

    # Calculate p-value for each tested variable
    pvals = []
    for mask in test_var_indices:
        # Fit null model (mask current variable)
        _, null_loglik, _ = _firth_newton_raphson(
            X, y, max_iter, max_stepsize, max_halfstep, tol, mask
        )
        # Compute LRT p-value
        pvals.append(_lrt(full_loglik, null_loglik))

    # Handle partial testing (fill untested variables with NaN)
    if len(pvals) < X.shape[1]:
        pval_array = np.full(X.shape[1], np.nan)
        for idx, test_var_idx in enumerate(test_var_indices):
            pval_array[test_var_idx] = pvals[idx]
        return pval_array
    
    return np.array(pvals)


def _lrt(full_loglik, null_loglik):
    """
    Compute likelihood ratio test (LRT) p-value.
    LRT statistic = 2*(full_loglik - null_loglik) ~ Chi2(df=1).

    Parameters
    ----------
    full_loglik : float
        Penalized log-likelihood of the full model.
    null_loglik : float
        Penalized log-likelihood of the null model.

    Returns
    -------
    pval : float
        LRT p-value (probability of observing statistic under null hypothesis).
    """
    lr_stat = 2 * (full_loglik - null_loglik)
    pval = chi2.sf(lr_stat, df=1)  # Survival function (1 - CDF)
    return pval


def _predict(X, coef):
    """
    Predict probabilities (sigmoid) with clipping for numerical stability.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_params)
        Feature matrix (including intercept if fitted).
    coef : ndarray of shape (n_params,)
        Fitted coefficients.

    Returns
    -------
    preds : ndarray of shape (n_samples,)
        Clipped predicted probabilities (1e-15 ≤ p ≤ 1-1e-15).
    """
    preds = expit(X @ coef)
    np.clip(preds, a_min=1e-15, a_max=1 - 1e-15, out=preds)
    return preds


def _profile_likelihood_ci(
    X,
    y,
    fitted_coef,
    full_loglik,
    max_iter,
    max_stepsize,
    max_halfstep,
    tol,
    alpha,
    test_vars,
):
    """
    Compute profile likelihood confidence intervals (CI) for coefficients.
    More robust than Wald CI for small samples/separation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_params)
        Feature matrix (including intercept if fitted).
    y : ndarray of shape (n_samples,)
        Binary target vector (0/1).
    fitted_coef : ndarray of shape (n_params,)
        Fitted coefficients from full model.
    full_loglik : float
        Penalized log-likelihood of full model.
    max_iter/max_stepsize/max_halfstep/tol : float/int
        Newton-Raphson hyperparameters for CI fitting.
    alpha : float
        Significance level (1-alpha = confidence level).
    test_vars : int/list/None
        Indices of variables to compute CI for (None = all variables).

    Returns
    -------
    ci : ndarray of shape (n_params, 2)
        Profile likelihood CI (lower, upper) for each coefficient (NaN if non-converged).
    """
    # Target log-likelihood for CI (LL0 = full_loglik - Chi2(1-alpha)/2)
    LL0 = full_loglik - chi2.ppf(1 - alpha, 1) / 2
    
    # Variables to compute CI for
    if test_vars is None:
        test_var_indices = range(fitted_coef.shape[0])
    elif isinstance(test_vars, int):
        test_var_indices = [test_vars]
    else:
        test_var_indices = sorted(test_vars)

    # Store lower/upper bounds
    lower_bound = []
    upper_bound = []

    # Compute lower (-1) and upper (+1) bounds
    for side in [-1, 1]:
        for coef_idx in test_var_indices:
            coef = deepcopy(fitted_coef)
            converged = False

            # Iterate to find CI bound
            for iter in range(1, max_iter + 1):
                preds = _predict(X, coef)
                loglike = -_loglikelihood(X, y, preds)
                
                # Weighted feature matrix and hat matrix
                XW = _get_XW(X, preds)
                hat = _hat_diag(XW)
                XW_aug = _get_aug_XW(X, preds, hat)
                
                # Fisher information matrix and corrected score
                fisher_info_mtx = XW_aug.T @ XW_aug
                U_star = np.matmul(X.T, y - preds + np.multiply(hat, 0.5 - preds))

                # Lambda adjustment for profile likelihood
                inv_fisher = np.linalg.pinv(fisher_info_mtx)
                tmp1x1 = U_star @ np.negative(inv_fisher) @ U_star
                underRoot = (
                    -2 * ((LL0 - loglike) + 0.5 * tmp1x1)
                    / (inv_fisher[coef_idx, coef_idx] + 1e-10)  # Avoid division by zero
                )
                lambda_ = 0 if underRoot < 0 else side * sqrt(underRoot)
                U_star[coef_idx] += lambda_

                # Compute update step (limit step size)
                step_size = np.linalg.lstsq(fisher_info_mtx, U_star, rcond=None)[0]
                mx = np.max(np.abs(step_size)) / max_stepsize
                if mx > 1:
                    step_size = step_size / mx
                
                # Update coefficients
                coef += step_size
                loglike_old = deepcopy(loglike)

                # Step-halving (ensure log-likelihood approaches LL0)
                for halfs in range(1, max_halfstep + 1):
                    preds = _predict(X, coef)
                    loglike = -_loglikelihood(X, y, preds)
                    if (abs(loglike - LL0) < abs(loglike_old - LL0)) and loglike > LL0:
                        break
                    step_size *= 0.5
                    coef -= step_size

                # Check convergence (log-likelihood close to LL0)
                if abs(loglike - LL0) <= tol:
                    if side == -1:
                        lower_bound.append(coef[coef_idx])
                    else:
                        upper_bound.append(coef[coef_idx])
                    converged = True
                    break

            # Handle non-convergence (fill with NaN)
            if not converged:
                if side == -1:
                    lower_bound.append(np.nan)
                else:
                    upper_bound.append(np.nan)
                warnings.warn(
                    f"Profile likelihood CI failed to converge for variable {coef_idx} (max_iter={max_iter})",
                    ConvergenceWarning,
                    stacklevel=2
                )

    # Combine bounds into CI matrix (handle partial testing)
    bounds = np.column_stack([lower_bound, upper_bound])
    if len(lower_bound) < fitted_coef.shape[0]:
        ci = np.full([fitted_coef.shape[0], 2], np.nan)
        for idx, test_var_idx in enumerate(test_var_indices):
            ci[test_var_idx] = bounds[idx]
        return ci

    return bounds


def _wald_ci(coef, bse, alpha):
    """
    Compute Wald confidence intervals (fast but less robust than profile likelihood).
    Formula: coef ± z_(1-alpha/2) * SE

    Parameters
    ----------
    coef : ndarray of shape (n_params,)
        Fitted coefficients.
    bse : ndarray of shape (n_params,)
        Standard errors of coefficients.
    alpha : float
        Significance level (1-alpha = confidence level).

    Returns
    -------
    ci : ndarray of shape (n_params, 2)
        Wald CI (lower, upper) for each coefficient.
    """
    # Z-score for confidence level (e.g., 1.96 for 95% CI)
    z = norm.ppf(1 - alpha / 2)
    
    # Compute lower/upper bounds
    lower_ci = coef - z * bse
    upper_ci = coef + z * bse
    
    return np.column_stack([lower_ci, upper_ci])


def _wald_test(coef, bse):
    """
    Compute Wald test p-values for coefficient significance.
    Formula: z = coef / SE → p-value = Chi2(z², df=1)

    Parameters
    ----------
    coef : ndarray of shape (n_params,)
        Fitted coefficients.
    bse : ndarray of shape (n_params,)
        Standard errors of coefficients.

    Returns
    -------
    pvals : ndarray of shape (n_params,)
        Wald test p-values (NaN if SE=0 to avoid division by zero).
    """
    # Z-statistic (avoid division by zero)
    z_stat = coef / (bse + 1e-10)
    
    # Wald test p-value (Chi2 distribution with df=1)
    pvals = chi2.sf(z_stat ** 2, df=1)
    
    return pvals


# ------------------------------
# Example Dataset Loaders (Optional)
# ------------------------------
def load_sex2():
    """
    Load the `sex2` dataset (from logistf R package) for testing.
    Binary classification: case/control (target) vs clinical features.

    Returns
    -------
    X : ndarray of shape (n_samples, 6)
        Feature matrix (age, oc, vic, vicl, vis, dia).
    y : ndarray of shape (n_samples,)
        Target vector (case=1, control=0).
    feature_names : list of str
        Names of features (["age", "oc", "vic", "vicl", "vis", "dia"]).
    """
    try:
        with open_text("firthlogist.datasets", "sex2.csv") as f:
            X = np.loadtxt(f, skiprows=1, delimiter=",")
        y = X[:, 0]
        X = X[:, 1:]
        feature_names = ["age", "oc", "vic", "vicl", "vis", "dia"]
        return X, y, feature_names
    except FileNotFoundError:
        warnings.warn("sex2 dataset not found (optional for testing)", stacklevel=2)
        return None, None, None


def load_endometrial():
    """
    Load the endometrial cancer dataset (Heinze & Schemper 2002) for testing.
    Binary classification: high-grade (HG=1) vs low-grade (HG=0) endometrial cancer.

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        Feature matrix (NV, PI, EH).
    y : ndarray of shape (n_samples,)
        Target vector (HG=1, HG=0).
    feature_names : list of str
        Names of features (["NV", "PI", "EH"]).
    """
    try:
        with open_text("firthlogist.datasets", "endometrial.csv") as f:
            X = np.loadtxt(f, skiprows=1, delimiter=",")
        y = X[:, -1]
        X = X[:, :-1]
        feature_names = ["NV", "PI", "EH"]
        return X, y, feature_names
    except FileNotFoundError:
        warnings.warn("endometrial dataset not found (optional for testing)", stacklevel=2)
        return None, None, None


# ------------------------------
# Test Code (Run to Validate)
# ------------------------------
if __name__ == "__main__":
    # Simulate small-sample data with quasi-complete separation (Firth's use case)
    np.random.seed(42)  # Reproducibility
    n = 50  # Small sample size
    X = np.random.normal(0, 1, (n, 2))  # 2 features
    y = (X[:, 0] > 0).astype(int)       # Quasi-separation (x1 predicts y)
    y[np.random.choice(n, 5)] = 1 - y[np.random.choice(n, 5)]  # Add noise

    # Fit Firth logistic regression model
    model = FirthLogisticRegression(
        fit_intercept=True,
        wald=False,  # Use profile likelihood CI (more robust)
        alpha=0.05,  # 95% CI
        skip_pvals=False,
        skip_ci=False
    )
    model.fit(X, y)

    # Print model summary (no tabulate required)
    model.summary(xname=["Feature 1", "Feature 2"])

    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    # Print prediction examples
    print("\n=== Prediction Examples (First 5 Samples) ===")
    print(f"Predicted Labels:   {y_pred[:5]}")
    print(f"Predicted Probabilities (Class 1): {y_pred_proba[:5, 1].round(4)}")
    print(f"True Labels:        {y[:5]}")