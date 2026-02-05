#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from sklearn.utils.validation import check_is_fitted

class FirthLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic regression with Firth's bias reduction method.
    Includes support for Weight, Freq, and Offset (SAS style).
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
        fit_const=True,
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
        self.fit_const = fit_const
        self.skip_pvals = skip_pvals
        self.skip_ci = skip_ci
        self.alpha = alpha
        self.wald = wald
        self.test_vars = test_vars

    def _more_tags(self):
        return {"binary_only": True}

    def _validate_input(self, X, y, weights, freq, offset):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples = X.shape[0]

        # SAS Logic: Effective Weight = weight * freq
        w = np.ones(n_samples) if weights is None else np.asarray(weights, dtype=np.float64)
        f = np.ones(n_samples) if freq is None else np.asarray(freq, dtype=np.float64)
        off = np.zeros(n_samples) if offset is None else np.asarray(offset, dtype=np.float64)

        if w.shape[0] != n_samples or f.shape[0] != n_samples or off.shape[0] != n_samples:
            raise ValueError("weights, freq, and offset must match the length of X.")

        eff_w = w * f
        
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"Only binary classification supported (got {len(self.classes_)} classes)")
        
        y = LabelEncoder().fit_transform(y).astype(X.dtype, copy=False)
        return X, y, eff_w, off

    def fit(self, X, y, weights=None, freq=None, offset=None):
        X, y, self.eff_w_, self.offset_ = self._validate_input(X, y, weights, freq, offset)
        
        if self.fit_const:
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        self.coef_, self.loglik_, self.n_iter_ = _firth_newton_raphson(
            X, y, self.max_iter, self.max_stepsize, self.max_halfstep, self.tol,
            eff_w=self.eff_w_, offset=self.offset_
        )

        self.bse_ = _bse(X, self.coef_, self.eff_w_, self.offset_)

        if not self.skip_ci:
            if not self.wald:
                self.ci_ = _profile_likelihood_ci(
                    X, y, self.coef_, self.loglik_, self.pl_max_iter, 
                    self.pl_max_stepsize, self.pl_max_halfstep, self.tol, self.alpha, 
                    self.test_vars, self.eff_w_, self.offset_
                )
            else:
                self.ci_ = _wald_ci(self.coef_, self.bse_, self.alpha)
        else:
            self.ci_ = np.full((self.coef_.shape[0], 2), np.nan)

        if not self.skip_pvals:
            if not self.wald:
                self.pvals_ = _penalized_lrt(
                    self.loglik_, X, y, self.max_iter, self.max_stepsize, 
                    self.max_halfstep, self.tol, self.test_vars, self.eff_w_, self.offset_
                )
            else:
                self.pvals_ = _wald_test(self.coef_, self.bse_)
        else:
            self.pvals_ = np.full(self.coef_.shape[0], np.nan)

        if self.fit_const:
            self.const_ = self.coef_[-1]
            self.coef_ = self.coef_[:-1]
        else:
            self.const_ = 0.0

        return self

    def summary(self, xname=None):
        check_is_fitted(self)
        if not xname:
            xname = [f"x{i}" for i in range(1, len(self.coef_) + 1)]
        
        var_names = ["const"] + xname if self.fit_const else xname
        coef_vals = [self.const_] + list(self.coef_) if self.fit_const else list(self.coef_)

        print("\n=== Firth Logistic Regression Summary ===")
        header = f"{'Variable':<10} {'Coef':<10} {'Std Err':<10} [{self.alpha/2:.3f}    {1-self.alpha/2:.3f}]    {'P-value':<10}"
        print(header + "\n" + "-" * len(header))
        for i, var in enumerate(var_names):
            ci_str = f"{self.ci_[i,0]:.4f}    {self.ci_[i,1]:.4f}" if not np.isnan(self.ci_[i,0]) else "NaN    NaN"
            print(f"{var:<10} {coef_vals[i]:<10.4f} {self.bse_[i]:<10.4f} {ci_str:<20} {self.pvals_[i]:<10.4f}")

    def decision_function(self, X, offset=None):
        check_is_fitted(self)
        X = np.asarray(X, dtype=np.float64)
        off = np.zeros(X.shape[0]) if offset is None else np.asarray(offset)
        return X @ self.coef_ + self.const_ + off

    def predict(self, X, offset=None):
        decision = self.decision_function(X, offset=offset)
        return self.classes_[(decision > 0).astype(int)]

    def predict_class(self, X, offset=None):
        """Standard alias for predict."""
        return self.predict(X, offset=offset)

    def predict_proba(self, X, offset=None):
        prob_1 = expit(self.decision_function(X, offset=offset))
        return np.column_stack([1 - prob_1, prob_1])

# --- Helper Functions ---

def _firth_newton_raphson(X, y, max_iter, max_stepsize, max_halfstep, tol, eff_w, offset, mask=None):
    coef = np.zeros(X.shape[1])
    for iter in range(1, max_iter + 1):
        preds = _predict(X, coef, offset)
        XW = _get_XW(X, preds, mask, eff_w)
        fisher = XW.T @ XW
        hat = _hat_diag(XW)
        
        U_star = np.matmul(X.T, eff_w * (y - preds + np.multiply(hat, 0.5 - preds)))
        step = np.linalg.lstsq(fisher, U_star, rcond=None)[0]
        
        mx = np.max(np.abs(step)) / max_stepsize
        if mx > 1: step /= mx
        
        coef_new = coef + step
        loglike = _loglikelihood(X, y, preds, eff_w, offset)
        loglike_new = _loglikelihood(X, y, _predict(X, coef_new, offset), eff_w, offset)
        
        halfs = 0
        while loglike < loglike_new and halfs < max_halfstep:
            step *= 0.5
            coef_new = coef + step
            loglike_new = _loglikelihood(X, y, _predict(X, coef_new, offset), eff_w, offset)
            halfs += 1

        if iter > 1 and np.linalg.norm(coef_new - coef) < tol:
            return coef_new, -loglike_new, iter
        coef = coef_new
    return coef, -loglike_new, max_iter

def _loglikelihood(X, y, preds, eff_w, offset):
    XW = _get_XW(X, preds, None, eff_w)
    fisher = XW.T @ XW + 1e-10 * np.eye(X.shape[1])
    penalty = 0.5 * np.log(np.linalg.det(fisher))
    preds_c = np.clip(preds, 1e-15, 1 - 1e-15)
    standard_ll = np.sum(eff_w * (y * np.log(preds_c) + (1 - y) * np.log(1 - preds_c)))
    return -1 * (standard_ll + penalty)

def _get_XW(X, preds, mask=None, eff_w=1.0):
    rootW = np.sqrt(eff_w * preds * (1 - preds))
    XW = rootW[:, np.newaxis] * X
    if mask is not None: XW[:, mask] = 0.0
    return XW

def _get_aug_XW(X, preds, hats, eff_w=1.0):
    rootW = np.sqrt(eff_w * preds * (1 - preds) * (1 + hats))
    return rootW[:, np.newaxis] * X

def _hat_diag(XW):
    qr, tau, _, _ = lapack.dgeqrf(XW, overwrite_a=True)
    Q, _, _ = lapack.dorgqr(qr, tau, overwrite_a=True)
    return np.einsum("ij,ij->i", Q, Q)

def _bse(X, coefs, eff_w, offset):
    preds = _predict(X, coefs, offset)
    XW = _get_XW(X, preds, None, eff_w)
    cov = np.linalg.pinv(XW.T @ XW + 1e-10 * np.eye(X.shape[1]))
    return np.sqrt(np.diag(cov))

def _penalized_lrt(full_ll, X, y, max_iter, max_stepsize, max_halfstep, tol, test_vars, eff_w, offset):
    indices = range(X.shape[1]) if test_vars is None else sorted(test_vars)
    pvals = np.full(X.shape[1], np.nan)
    for mask in indices:
        _, null_ll, _ = _firth_newton_raphson(X, y, max_iter, max_stepsize, max_halfstep, tol, eff_w, offset, mask)
        pvals[mask] = _lrt(full_ll, null_ll)
    return pvals

def _lrt(full_loglik, null_loglik):
    return chi2.sf(2 * (full_loglik - null_loglik), df=1)

def _predict(X, coef, offset=0):
    preds = expit(X @ coef + offset)
    return np.clip(preds, 1e-15, 1 - 1e-15)

def _profile_likelihood_ci(X, y, fitted_coef, full_ll, max_iter, max_stepsize, max_halfstep, tol, alpha, test_vars, eff_w, offset):
    LL0 = full_ll - chi2.ppf(1 - alpha, 1) / 2
    indices = range(fitted_coef.shape[0]) if test_vars is None else sorted(test_vars)
    ci = np.full((fitted_coef.shape[0], 2), np.nan)

    for side_idx, side in enumerate([-1, 1]):
        for coef_idx in indices:
            coef = deepcopy(fitted_coef)
            for _ in range(max_iter):
                preds = _predict(X, coef, offset)
                loglike = -_loglikelihood(X, y, preds, eff_w, offset)
                XW = _get_XW(X, preds, None, eff_w)
                hat = _hat_diag(XW)
                XW_aug = _get_aug_XW(X, preds, hat, eff_w)
                
                fisher = XW_aug.T @ XW_aug
                U_star = np.matmul(X.T, eff_w * (y - preds + np.multiply(hat, 0.5 - preds)))
                inv_f = np.linalg.pinv(fisher)
                
                u_val = -2 * ((LL0 - loglike) + 0.5 * (U_star @ np.negative(inv_f) @ U_star)) / (inv_f[coef_idx, coef_idx] + 1e-10)
                U_star[coef_idx] += side * sqrt(max(0, u_val))

                step = np.linalg.lstsq(fisher, U_star, rcond=None)[0]
                coef += step / max(1, np.max(np.abs(step)) / max_stepsize)
                if abs(-_loglikelihood(X, y, _predict(X, coef, offset), eff_w, offset) - LL0) <= tol:
                    ci[coef_idx, side_idx] = coef[coef_idx]
                    break
    return ci

def _wald_ci(coef, bse, alpha):
    z = norm.ppf(1 - alpha / 2)
    return np.column_stack([coef - z * bse, coef + z * bse])

def _wald_test(coef, bse):
    return chi2.sf((coef / (bse + 1e-10)) ** 2, df=1)


# In[ ]:


# 1. Create a dataset with separation (X1 perfectly predicts Y)
# Firth's method is designed specifically for this scenario.
X = np.array([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9], [6, 10]])
y = np.array([0, 0, 0, 1, 1, 1])

# 2. Define weights, frequencies, and offsets (SAS style)
weights = np.array([1.0, 1.0, 1.2, 0.8, 1.0, 1.0])
freq = np.array([1, 1, 2, 1, 1, 1]) # Doubling importance of the 3rd sample
offset = np.zeros(6) # No fixed adjustment for this example

# 3. Initialize and fit the model
model = FirthLogisticRegression(fit_const=True, wald=False)
model.fit(X, y, weights=weights, freq=freq, offset=offset)

# 4. Print Summary (using the restored summary function)
model.summary(xname=["Feature_A", "Feature_B"])

# 5. Make Predictions (using the restored predict_class and predict_proba)
new_data = np.array([[1.5, 5.5], [5.5, 9.5]])
classes = model.predict_class(new_data)
probs = model.predict_proba(new_data)

print("\n=== Predictions on New Data ===")
for i, (cls, pr) in enumerate(zip(classes, probs)):
    print(f"Sample {i+1}: Predicted Class = {cls}, Probabilities = {pr.round(4)}")


# In[ ]:




