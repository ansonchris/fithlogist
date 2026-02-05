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
import pandas as pd  # Supplement necessary dependency (for time formatting)

class FirthLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Firth's bias-reduced logistic regression (compatible with statsmodels Logit + SAS proc logistic functionalities)
    
    Core Features:
    - Interface fully aligned with statsmodels Logit: Attributes like params, bse, pvalues, zvalues, ci
    - New SAS-style weight/freq/offset options, behavior exactly consistent with proc logistic
    - Support dynamic offset input in prediction (model.predict(X, offset=offsets))
    - Retains advantages of Firth's method (resolves separation issues, corrects small-sample bias)
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
        # New: SAS proc logistic-style weight/freq/offset options
        weight=None,  # Corresponds to SAS weight=var: Observation weights (non-negative values)
        freq=None,    # Corresponds to SAS freq=var: Observation frequencies (non-negative integers, equivalent to repeated observations)
        offset=None,  # Corresponds to SAS offset=var: Offset term with fixed coefficient 1 (not involved in fitting)
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
        # New instance variables (SAS-compatible)
        self.weight = weight
        self.freq = freq
        self.offset = offset

    def _more_tags(self):
        return {"binary_only": True}

    def _validate_input(self, X, y):
        # Original input validation logic
        if X is None or y is None:
            raise ValueError("X and y must not be None (empty input)")
        try:
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to convert X/y to float64 array: {str(e)}")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got {X.ndim}D")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D (n_samples,), got {y.ndim}D")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatched sample count: X has {X.shape[0]} samples, y has {y.shape[0]}")
        if X.shape[0] < 2:
            raise ValueError(f"At least 2 samples required, got {X.shape[0]}")
        if self.max_iter < 0:
            raise ValueError(f"max_iter must be positive (got {self.max_iter})")
        if self.max_halfstep < 0:
            raise ValueError(f"max_halfstep must be ≥ 0 (got {self.max_halfstep})")
        if self.tol < 0:
            raise ValueError(f"tol must be positive (got {self.tol})")
        
        # Validate classification targets (binary only)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"Only binary classification supported (got {len(self.classes_)} classes)")
        y = LabelEncoder().fit_transform(y).astype(X.dtype, copy=False)
        
        self.n_samples_, self.n_features_ = X.shape
        
        # --------------------------
        # New: SAS-style weight/freq/offset validation (strictly matches proc logistic behavior)
        # --------------------------
        # 1. weight and freq are mutually exclusive (SAS core rule)
        if self.weight is not None and self.freq is not None:
            raise ValueError("weight and freq cannot be used simultaneously (SAS proc logistic compatible)")
        
        # 2. Process effective weights (choose either weight or freq, default to all 1s)
        self.effective_weight_ = np.ones(self.n_samples_, dtype=np.float64)
        if self.weight is not None:
            # SAS weight: Non-negative values for weighted likelihood estimation
            weight = np.asarray(self.weight, dtype=np.float64)
            if weight.ndim != 1 or weight.shape[0] != self.n_samples_:
                raise ValueError(f"weight must be 1D array with length {self.n_samples_} (match sample count)")
            if np.any(weight < 0):
                raise ValueError("weight cannot contain negative values (SAS proc logistic rule)")
            self.effective_weight_ = weight
        elif self.freq is not None:
            # SAS freq: Non-negative integers (equivalent to repeated observations, warning for non-integers)
            freq = np.asarray(self.freq, dtype=np.float64)
            if freq.ndim != 1 or freq.shape[0] != self.n_samples_:
                raise ValueError(f"freq must be 1D array with length {self.n_samples_} (match sample count)")
            if np.any(freq < 0):
                raise ValueError("freq cannot contain negative values (SAS proc logistic rule)")
            if not np.allclose(freq, np.round(freq)):
                warnings.warn("freq should be non-negative integers (SAS proc logistic expects integers)", UserWarning, stacklevel=2)
            self.effective_weight_ = freq
        
        # 3. Process offset (SAS offset: Fixed coefficient 1, not involved in model estimation)
        self.offset_ = np.zeros(self.n_samples_, dtype=np.float64)
        if self.offset is not None:
            offset = np.asarray(self.offset, dtype=np.float64)
            if offset.ndim != 1 or offset.shape[0] != self.n_samples_:
                raise ValueError(f"offset must be 1D array with length {self.n_samples_} (match sample count)")
            self.offset_ = offset
        return X, y
    
    # --------------------------
    # Core Fix 1: Correct _validate_prediction_offset logic
    # --------------------------
    def _validate_prediction_offset(self, X, offset):
        """Validate offset input during prediction (helper function)"""
        n_pred_samples = X.shape[0]
        if offset is None:
            # Fix: Return all zeros with prediction sample count (not training offset)
            return np.zeros(n_pred_samples, dtype=np.float64)
        
        # Validate custom prediction offset
        try:
            offset = np.asarray(offset, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to convert prediction offset to float64 array: {str(e)}")
        if offset.ndim != 1:
            raise ValueError(f"Prediction offset must be 1D (n_samples,), got {offset.ndim}D")
        if offset.shape[0] != n_pred_samples:
            raise ValueError(f"Prediction offset length ({offset.shape[0]}) must match X sample count ({n_pred_samples})")
        return offset

    def decision_function(self, X, offset=None):
        """
        Align with SAS offset behavior: Linear predictor = X@coef + intercept + offset
        Supports dynamic offset input during prediction (overrides training offset if provided)
        
        Parameters:
            X: 2D array of shape (n_samples, n_features) - Prediction features
            offset: 1D array of shape (n_samples,) - Custom offset for prediction (optional)
        """
        check_is_fitted(self)
        try:
            X = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to convert X to float64 array: {str(e)}")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got {X.ndim}D")
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X feature count ({X.shape[1]}) must match training feature count ({self.n_features_})")
        
        # Validate and get prediction offset (custom > zero)
        pred_offset = self._validate_prediction_offset(X, offset)
        
        # Linear predictor with dynamic offset
        return X @ self.coef_ + self.intercept_ + pred_offset

    # --------------------------
    # Core Fix 2: Correct typo (offset-offset → offset=offset)
    # --------------------------
    def predict(self, X, offset=None):
        """
        Predict class labels for X with optional custom offset
        
        Parameters:
            X: 2D array of shape (n_samples, n_features) - Prediction features
            offset: 1D array of shape (n_samples,) - Custom offset for prediction (optional)
        """
        # Fix: Typo from offset-offset to offset=offset
        decision = self.decision_function(X, offset=offset)
        indices = (decision > 0).astype(int)
        return self.classes_[indices]

    def predict_proba(self, X, offset=None):
        """
        Predict class probabilities for X with optional custom offset
        
        Parameters:
            X: 2D array of shape (n_samples, n_features) - Prediction features
            offset: 1D array of shape (n_samples,) - Custom offset for prediction (optional)
        """
        decision = self.decision_function(X, offset=offset)
        proba_1 = expit(decision)
        return np.column_stack([1 - proba_1, proba_1])

    def fit(self, X, y):
        X, y = self._validate_input(X, y)
        
        # Record variable names (align with statsmodels)
        self.endog_name_ = "y"
        self.exog_name_ = [f"x{i+1}" for i in range(self.n_features_)]
        
        # Add intercept term (Intercept first, align with statsmodels)
        if self.fit_intercept:
            X = np.hstack((np.ones((self.n_samples_, 1)), X))
            self.exog_name_.insert(0, "Intercept")
        
        # Fit model (pass effective weights and offset, compatible with Firth's method)
        coef, self.loglik, self.n_iter_ = _firth_newton_raphson(
            X, y, self.max_iter, self.max_stepsize, self.max_halfstep, self.tol,
            effective_weight=self.effective_weight_,
            offset=self.offset_
        )
        
        # Calculate standard errors (weighted version)
        self.bse = _bse(X, coef, self.effective_weight_, self.offset_)
        self.params = coef
        
        # Calculate confidence intervals (weighted version)
        if not self.skip_ci:
            if not self.wald:
                self.ci = _profile_likelihood_ci(
                    X=X, y=y, fitted_coef=coef, full_loglik=self.loglik,
                    max_iter=self.pl_max_iter, max_stepsize=self.pl_max_stepsize,
                    max_halfstep=self.pl_max_halfstep, tol=self.tol, alpha=self.alpha,
                    test_vars=self.test_vars,
                    effective_weight=self.effective_weight_,
                    offset=self.offset_
                )
            else:
                self.ci = _wald_ci(coef, self.bse, self.alpha)
        else:
            self.ci = np.full((self.params.shape[0], 2), np.nan)
        
        # Calculate p-values and z-values (weighted version)
        if not self.skip_pvals:
            if not self.wald:
                self.pvalues = _penalized_lrt(
                    self.loglik, X, y, self.max_iter, self.max_stepsize,
                    self.max_halfstep, self.tol, self.test_vars,
                    effective_weight=self.effective_weight_,
                    offset=self.offset_
                )
            else:
                self.pvalues = _wald_test(coef, self.bse)
            self.zvalues = self.params / (self.bse + 1e-10)  # Avoid division by zero
        else:
            self.pvalues = np.full(self.params.shape[0], np.nan)
            self.zvalues = np.full(self.params.shape[0], np.nan)
        
        # Separate intercept and feature coefficients (compatible with sklearn)
        if self.fit_intercept:
            self.intercept_ = self.params[0]
            self.coef_ = self.params[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self.params
        
        # Calculate statsmodels-style statistics (weighted version)
        self.n_params_ = self.params.shape[0]
        self.AIC = 2 * self.n_params_ - 2 * self.loglik
        self.BIC = self.n_params_ * np.log(self.n_samples_) - 2 * self.loglik
        # Weighted null model (for McFadden R²)
        null_X = X[:, [0]] if self.fit_intercept else X[:, :0]
        null_model = _firth_newton_raphson(
            null_X, y, self.max_iter, self.max_stepsize, self.max_halfstep, self.tol,
            effective_weight=self.effective_weight_,
            offset=self.offset_
        )
        self.null_loglik = null_model[1]
        self.mcfadden_r2 = 1 - (self.loglik / self.null_loglik)
        
        return self

    def summary(self, xname=None, title=None):
        """Fully aligned with statsmodels Logit format, with new SAS option usage status display"""
        check_is_fitted(self)
        
        # Process feature names
        if xname is not None:
            if len(xname) != self.n_features_:
                raise ValueError(f"xname length ({len(xname)}) must match number of features ({self.n_features_})")
            exog_names = xname.copy()
            if self.fit_intercept:
                exog_names.insert(0, "Intercept")
        else:
            exog_names = self.exog_name_
        
        # 1. Basic model information (with new SAS option status)
        title = title or "Firth Logistic Regression Results (SAS proc logistic compatible)"
        print("=" * len(title))
        print(title)
        print("=" * len(title))
        print(f"Dep. Variable:        {self.endog_name_}")
        print(f"Model:                FirthLogistic")
        print(f"Method:               ML (Firth bias-reduced)")
        print(f"Date:                 {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Time:                 {pd.Timestamp.now().strftime('%H:%M:%S')}")
        print(f"No. Observations:     {self.n_samples_}")
        print(f"Weight Used:          {self.weight is not None} (SAS style)")
        print(f"Freq Used:            {self.freq is not None} (SAS style)")
        print(f"Training Offset Used: {self.offset is not None} (SAS style)")
        print(f"Prediction Offset:    Supported (dynamic input via predict(X, offset=...))")
        print(f"Df Residuals:         {self.n_samples_ - self.n_params_}")
        print(f"Df Model:             {self.n_params_ - (1 if self.fit_intercept else 0)}")
        print(f"Pseudo R-squ. (McF):  {self.mcfadden_r2:.4f}")
        print(f"Log-Likelihood:       {self.loglik:.4f}")
        print(f"AIC:                  {self.AIC:.4f}")
        print(f"BIC:                  {self.BIC:.4f}")
        print(f"Converged:            {self.n_iter_ < self.max_iter}")
        print(f"Max Iterations:       {self.max_iter}")
        print()
        
        # 2. Coefficient table (retains statsmodels format)
        ci_level = 100 * (1 - self.alpha)
        header = (
            f"{'':<2} {'Variable':<12} {'Coef.':<10} {'Std. Err.':<10} "
            f"{'z':<8} {'P>|z|':<8} [{ci_level:.1f}% Conf. Int.]"
        )
        separator = "-" * len(header)
        print(header)
        print(separator)
        
        for i, (var_name, coef, se, z, pval, ci_lower, ci_upper) in enumerate(
            zip(exog_names, self.params, self.bse, self.zvalues, self.pvalues, self.ci[:, 0], self.ci[:, 1])
        ):
            coef_str = f"{coef:.4f}" if not np.isnan(coef) else "."
            se_str = f"{se:.4f}" if not np.isnan(se) else "."
            z_str = f"{z:.4f}" if not np.isnan(z) else "."
            pval_str = f"{pval:.4f}" if not np.isnan(pval) else "."
            # Significance markers
            sig_mark = ""
            if not np.isnan(pval):
                sig_mark = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "." if pval < 0.1 else ""
            ci_str = f"{ci_lower:.4f}  {ci_upper:.4f}" if not (np.isnan(ci_lower) or np.isnan(ci_upper)) else ".        ."
            print(f"{i+1:<2} {var_name:<12} {coef_str:<10} {se_str:<10} {z_str:<8} {pval_str:<8} {ci_str} {sig_mark}")
        
        # 3. Significance legend and notes
        print(separator)
        print("Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1")
        print()
        print(f"Note: Confidence interval method = {'Wald' if self.wald else 'Profile Likelihood'}")
        print(f"Note: SAS proc logistic compatible options: weight={self.weight is not None}, freq={self.freq is not None}, offset={self.offset is not None}")
        print(f"Note: Dynamic prediction offset is supported via predict(X, offset=...) and predict_proba(X, offset=...)")
        if self.wald:
            print("Note: Wald intervals are faster but less robust for small samples/separation.")

# ------------------------------
# Core Helper Functions (unchanged)
# ------------------------------
def _firth_newton_raphson(X, y, max_iter, max_stepsize, max_halfstep, tol, mask=None, effective_weight=None, offset=None):
    effective_weight = np.ones(X.shape[0]) if effective_weight is None else effective_weight
    offset = np.zeros(X.shape[0]) if offset is None else offset
    
    coef = np.zeros(X.shape[1])
    for iter in range(1, max_iter + 1):
        # Linear predictor = X@coef + offset (SAS offset core logic)
        linear_pred = X @ coef + offset
        preds = expit(linear_pred)
        
        # Weighted feature matrix (SAS weight/freq acts on likelihood)
        rootW = np.sqrt(preds * (1 - preds) * effective_weight)
        XW = rootW[:, np.newaxis] * X
        if mask is not None:
            XW[:, mask] = 0.0
        fisher_info_mtx = XW.T @ XW
        
        # Weighted corrected score vector (Firth's method + SAS weights)
        hat = _hat_diag(XW)
        U_star = np.matmul(X.T, effective_weight * (y - preds + np.multiply(hat, 0.5 - preds)))
        
        # Step size calculation and limitation
        step_size = np.linalg.lstsq(fisher_info_mtx, U_star, rcond=None)[0]
        mx = np.max(np.abs(step_size)) / max_stepsize
        if mx > 1:
            step_size /= mx
        
        # Coefficient update and step halving (ensure likelihood increase)
        coef_new = coef + step_size
        linear_pred_new = X @ coef_new + offset
        preds_new = expit(linear_pred_new)
        
        loglike = _loglikelihood(X, y, preds, effective_weight)
        loglike_new = _loglikelihood(X, y, preds_new, effective_weight)
        steps = 0
        
        while loglike < loglike_new:
            step_size *= 0.5
            coef_new = coef + step_size
            linear_pred_new = X @ coef_new + offset
            preds_new = expit(linear_pred_new)
            loglike_new = _loglikelihood(X, y, preds_new, effective_weight)
            steps += 1
            if steps == max_halfstep:
                warnings.warn("Step-halving failed to converge (max halfsteps reached)", ConvergenceWarning, stacklevel=2)
                return coef_new, -loglike_new, iter
        
        # Convergence check
        if iter > 1 and np.linalg.norm(coef_new - coef) < tol:
            return coef_new, -loglike_new, iter
        coef = coef_new
    
    warnings.warn(f"Newton-Raphson failed to converge (max_iter={max_iter} reached)", ConvergenceWarning, stacklevel=2)
    return coef, -loglike_new, max_iter

def _loglikelihood(X, y, preds, effective_weight=None):
    """Weighted Firth likelihood function (adapted for SAS weight/freq)"""
    effective_weight = np.ones(X.shape[0]) if effective_weight is None else effective_weight
    
    XW = _get_XW(X, preds, effective_weight=effective_weight)
    fisher_info_mtx = XW.T @ XW + 1e-10 * np.eye(XW.shape[1])  # Numerical stability
    penalty = 0.5 * np.log(np.linalg.det(fisher_info_mtx))
    
    # Weighted standard likelihood (core action point of SAS weight/freq)
    preds_clipped = np.clip(preds, 1e-15, 1 - 1e-15)
    standard_loglik = np.sum(effective_weight * (y * np.log(preds_clipped) + (1 - y) * np.log(1 - preds_clipped)))
    
    return -1 * (standard_loglik + penalty)

def _get_XW(X, preds, mask=None, effective_weight=None):
    """Weighted feature matrix (adapted for SAS weight/freq)"""
    effective_weight = np.ones(X.shape[0]) if effective_weight is None else effective_weight
    rootW = np.sqrt(preds * (1 - preds) * effective_weight)
    XW = rootW[:, np.newaxis] * X
    if mask is not None:
        XW[:, mask] = 0.0
    return XW

def _get_aug_XW(X, preds, hats, effective_weight=None):
    """Weighted augmented feature matrix (adapted for SAS weight/freq)"""
    effective_weight = np.ones(X.shape[0]) if effective_weight is None else effective_weight
    rootW = np.sqrt(preds * (1 - preds) * (1 + hats) * effective_weight)
    XW_aug = rootW[:, np.newaxis] * X
    return XW_aug

def _hat_diag(XW):
    qr, tau, _, _ = lapack.dgeqrf(XW, overwrite_a=True)
    Q, _, _ = lapack.dorgqr(qr, tau, overwrite_a=True)
    return np.einsum("ij,ij->i", Q, Q)

def _bse(X, coefs, effective_weight=None, offset=None):
    """Weighted standard errors (adapted for SAS weight/offset)"""
    effective_weight = np.ones(X.shape[0]) if effective_weight is None else effective_weight
    offset = np.zeros(X.shape[0]) if offset is None else offset
    
    linear_pred = X @ coefs + offset
    preds = expit(linear_pred)
    XW = _get_XW(X, preds, effective_weight=effective_weight)
    
    fisher_info_mtx = XW.T @ XW + 1e-10 * np.eye(XW.shape[1])
    cov_matrix = np.linalg.pinv(fisher_info_mtx)
    return np.sqrt(np.diag(cov_matrix))

def _penalized_lrt(
    full_loglik, X, y, max_iter, max_stepsize, max_halfstep, tol, test_vars,
    effective_weight=None, offset=None
):
    """Weighted likelihood ratio test (adapted for SAS weight/offset)"""
    effective_weight = np.ones(X.shape[0]) if effective_weight is None else effective_weight
    offset = np.zeros(X.shape[0]) if offset is None else offset
    
    test_var_indices = range(X.shape[1]) if test_vars is None else (
        [test_vars] if isinstance(test_vars, int) else sorted(test_vars)
    )
    
    pvals = []
    for mask in test_var_indices:
        _, null_loglik, _ = _firth_newton_raphson(
            X, y, max_iter, max_stepsize, max_halfstep, tol, mask,
            effective_weight=effective_weight, offset=offset
        )
        pvals.append(_lrt(full_loglik, null_loglik))
    
    if len(pvals) < X.shape[1]:
        pval_array = np.full(X.shape[1], np.nan)
        for idx, test_var_idx in enumerate(test_var_indices):
            pval_array[test_var_idx] = pvals[idx]
        return pval_array
    return np.array(pvals)

def _lrt(full_loglik, null_loglik):
    lr_stat = 2 * (full_loglik - null_loglik)
    return chi2.sf(lr_stat, df=1)

def _predict(X, coef, offset=None):
    """Prediction probabilities with offset (adapted for SAS offset)"""
    offset = np.zeros(X.shape[0]) if offset is None else offset
    linear_pred = X @ coef + offset
    preds = expit(linear_pred)
    np.clip(preds, a_min=1e-15, a_max=1 - 1e-15, out=preds)
    return preds

def _profile_likelihood_ci(
    X, y, fitted_coef, full_loglik, max_iter, max_stepsize, max_halfstep, tol, alpha, test_vars,
    effective_weight=None, offset=None
):
    """Weighted profile likelihood confidence intervals (adapted for SAS weight/offset)"""
    effective_weight = np.ones(X.shape[0]) if effective_weight is None else effective_weight
    offset = np.zeros(X.shape[0]) if offset is None else offset
    
    LL0 = full_loglik - chi2.ppf(1 - alpha, 1) / 2
    test_var_indices = range(fitted_coef.shape[0]) if test_vars is None else (
        [test_vars] if isinstance(test_vars, int) else sorted(test_vars)
    )
    
    lower_bound, upper_bound = [], []
    for side in [-1, 1]:
        for coef_idx in test_var_indices:
            coef = deepcopy(fitted_coef)
            converged = False
            for iter in range(1, max_iter + 1):
                preds = _predict(X, coef, offset=offset)
                loglike = -_loglikelihood(X, y, preds, effective_weight=effective_weight)
                
                XW = _get_XW(X, preds, effective_weight=effective_weight)
                hat = _hat_diag(XW)
                XW_aug = _get_aug_XW(X, preds, hat, effective_weight=effective_weight)
                
                fisher_info_mtx = XW_aug.T @ XW_aug
                U_star = np.matmul(X.T, effective_weight * (y - preds + np.multiply(hat, 0.5 - preds)))
                
                inv_fisher = np.linalg.pinv(fisher_info_mtx)
                tmp1x1 = U_star @ np.negative(inv_fisher) @ U_star
                underRoot = -2 * ((LL0 - loglike) + 0.5 * tmp1x1) / (inv_fisher[coef_idx, coef_idx] + 1e-10)
                lambda_ = 0 if underRoot < 0 else side * sqrt(underRoot)
                U_star[coef_idx] += lambda_
                
                step_size = np.linalg.lstsq(fisher_info_mtx, U_star, rcond=None)[0]
                mx = np.max(np.abs(step_size)) / max_stepsize
                if mx > 1:
                    step_size /= mx
                
                coef += step_size
                loglike_old = deepcopy(loglike)
                for halfs in range(1, max_halfstep + 1):
                    preds = _predict(X, coef, offset=offset)
                    loglike = -_loglikelihood(X, y, preds, effective_weight=effective_weight)
                    if (abs(loglike - LL0) < abs(loglike_old - LL0)) and loglike > LL0:
                        break
                    step_size *= 0.5
                    coef -= step_size
                
                if abs(loglike - LL0) <= tol:
                    (lower_bound if side == -1 else upper_bound).append(coef[coef_idx])
                    converged = True
                    break
            if not converged:
                (lower_bound if side == -1 else upper_bound).append(np.nan)
                warnings.warn(f"Profile likelihood CI failed to converge for variable {coef_idx}", ConvergenceWarning, stacklevel=2)
    
    bounds = np.column_stack([lower_bound, upper_bound])
    if len(lower_bound) < fitted_coef.shape[0]:
        ci = np.full([fitted_coef.shape[0], 2], np.nan)
        for idx, test_var_idx in enumerate(test_var_indices):
            ci[test_var_idx] = bounds[idx]
        return ci
    return bounds

def _wald_ci(coef, bse, alpha):
    z = norm.ppf(1 - alpha / 2)
    return np.column_stack([coef - z * bse, coef + z * bse])

def _wald_test(coef, bse):
    z_stat = coef / (bse + 1e-10)
    return chi2.sf(z_stat ** 2, df=1)

# ------------------------------
# Test Code (verify fixed dynamic offset)
# ------------------------------
if __name__ == "__main__":
    # Simulate data
    np.random.seed(42)
    n_train = 50
    n_pred = 20
    X_train = np.random.normal(0, 1, (n_train, 2))
    y_train = (X_train[:, 0] > 0).astype(int)
    y_train[np.random.choice(n_train, 5)] = 1 - y_train[np.random.choice(n_train, 5)]
    X_pred = np.random.normal(0, 1, (n_pred, 2))
    
    # Construct offsets (training + custom prediction offset)
    train_offset = np.random.normal(0, 0.5, n_train)
    pred_offset = np.random.normal(0, 0.5, n_pred)  # Custom offset for prediction
    
    # 1. Fit model with training offset
    model = FirthLogisticRegression(fit_intercept=True, offset=train_offset)
    model.fit(X_train, y_train)
    model.summary(xname=["FeatureA", "FeatureB"])
    
    # 2. Predict with custom offset (core new functionality)
    print("\n=== Test Dynamic Prediction Offset (Fixed) ===")
    y_pred_with_offset = model.predict(X_pred, offset=pred_offset)
    y_pred_without_offset = model.predict(X_pred)  # Use zero offset (correct behavior)
    proba_pred_with_offset = model.predict_proba(X_pred, offset=pred_offset)
    
    print(f"Prediction results with custom offset (first 5 samples): {y_pred_with_offset[:5]}")
    print(f"Prediction results without custom offset (first 5 samples): {y_pred_without_offset[:5]}")
    print(f"Probability results with custom offset (first 5 samples): \n{proba_pred_with_offset[:5].round(4)}")


# 构造预测用 offset
pred_offset = np.random.normal(0, 0.5, 20)

# 调用带动态 offset 的 predict
y_pred = model.predict(X_pred, offset=pred_offset)

# 调用带动态 offset 的 predict_proba（可选）
proba_pred = model.predict_proba(X_pred, offset=pred_offset)