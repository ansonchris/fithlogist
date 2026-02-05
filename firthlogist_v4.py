#!/usr/bin/env python
# coding: utf-8

# In[16]:


import warnings
from copy import deepcopy
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
    Firth Logistic Regression with Weight, Freq, and Offset (SAS PROC LOGISTIC style). [cite: 2]
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
        fit_const=True,  # Renamed from fit_intercept [cite: 9]
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

    def _validate_input(self, X, y, weights, freq, offset):
        """Validates input and computes effective weights and offset. [cite: 28, 33]"""
        X = np.asarray(X, dtype=np.float64) 
        y = np.asarray(y, dtype=np.float64) 
        n_samples = X.shape[0]

        # Handle SAS-style optional parameters
        w = np.ones(n_samples) if weights is None else np.asarray(weights, dtype=np.float64)
        f = np.ones(n_samples) if freq is None else np.asarray(freq, dtype=np.float64)
        off = np.zeros(n_samples) if offset is None else np.asarray(offset, dtype=np.float64)

        if w.shape[0] != n_samples or f.shape[0] != n_samples or off.shape[0] != n_samples:
            raise ValueError("weights, freq, and offset must match the number of samples in X.") 

        # Compute effective weight: w * freq
        eff_w = w * f
        
        check_classification_targets(y) 
        self.classes_ = np.unique(y) 
        y = LabelEncoder().fit_transform(y).astype(X.dtype, copy=False) 

        return X, y, eff_w, off

    def fit(self, X, y, weights=None, freq=None, offset=None):
        """Fits model with weight, freq, and offset. [cite: 40, 45]"""
        X, y, self.eff_w_, self.offset_ = self._validate_input(X, y, weights, freq, offset)
        
        if self.fit_const: 
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        # Fit model via Newton-Raphson [cite: 46]
        self.coef_, self.loglik_, self.n_iter_ = _firth_newton_raphson(
            X, y, self.max_iter, self.max_stepsize, self.max_halfstep, self.tol, 
            eff_w=self.eff_w_, offset=self.offset_
        )

        self.bse_ = _bse(X, self.coef_, self.eff_w_, self.offset_) 

        # Profile Likelihood CI [cite: 47]
        if not self.skip_ci:
            if not self.wald:
                self.ci_ = _profile_likelihood_ci(
                    X, y, self.coef_, self.loglik_, self.pl_max_iter, 
                    self.pl_max_stepsize, self.pl_max_halfstep, self.tol, self.alpha, 
                    self.test_vars, self.eff_w_, self.offset_
                )
            else:
                self.ci_ = _wald_ci(self.coef_, self.bse_, self.alpha)
        
        # Penalized LRT [cite: 51]
        if not self.skip_pvals:
            if not self.wald:
                self.pvals_ = _penalized_lrt(
                    self.loglik_, X, y, self.max_iter, self.max_stepsize, 
                    self.max_halfstep, self.tol, self.test_vars, self.eff_w_, self.offset_
                )
            else:
                self.pvals_ = _wald_test(self.coef_, self.bse_) 

        if self.fit_const: 
            self.const_ = self.coef_[-1]
            self.coef_ = self.coef_[:-1]
        else:
            self.const_ = 0.0

        return self

    def decision_function(self, X, offset=None):
        """Linear predictor with optional offset. [cite: 65]"""
        check_is_fitted(self)
        X = np.asarray(X, dtype=np.float64)
        off = np.zeros(X.shape[0]) if offset is None else np.asarray(offset)
        return X @ self.coef_ + self.const_ + off 

    def predict_proba(self, X, offset=None):
        """Probability prediction with optional offset. [cite: 72]"""
        scores = self.decision_function(X, offset=offset)
        p1 = expit(scores) 
        return np.column_stack([1 - p1, p1])

    def predict(self, X, offset=None):
        """Class prediction. [cite: 69]"""
        return self.classes_[(self.decision_function(X, offset=offset) > 0).astype(int)]

# --- Modified Helper Functions ---

def _firth_newton_raphson(X, y, max_iter, max_stepsize, max_halfstep, tol, eff_w, offset, mask=None):
    """Newton-Raphson with weights and offset. [cite: 76]"""
    coef = np.zeros(X.shape[1]) 
    
    for iter in range(1, max_iter + 1):
        # Sigmoid probability includes offset [cite: 87, 137]
        preds = expit(X @ coef + offset)
        
        # Weighted feature matrix [cite: 88, 102]
        XW = _get_XW(X, preds, eff_w, mask)
        fisher_info_mtx = XW.T @ XW
        hat = _hat_diag(XW) 
        
        # Weighted Score Vector (U*) with Firth correction [cite: 89]
        U_star = np.matmul(X.T, eff_w * (y - preds) + eff_w * np.multiply(hat, 0.5 - preds))
        
        step_size = np.linalg.lstsq(fisher_info_mtx, U_star, rcond=None)[0]
        mx = np.max(np.abs(step_size)) / max_stepsize 
        if mx > 1: step_size /= mx
        
        coef_new = coef + step_size 
        loglike = _loglikelihood(X, y, preds, eff_w, offset) 
        loglike_new = _loglikelihood(X, y, expit(X @ coef_new + offset), eff_w, offset) 
        
        # Step halving [cite: 91]
        steps = 0
        while loglike < loglike_new and steps < max_halfstep:
            step_size *= 0.5
            coef_new = coef + step_size
            loglike_new = _loglikelihood(X, y, expit(X @ coef_new + offset), eff_w, offset)
            steps += 1 

        if iter > 1 and np.linalg.norm(coef_new - coef) < tol:
            return coef_new, -loglike_new, iter
        coef = coef_new

    return coef, -loglike_new, max_iter 

def _loglikelihood(X, y, preds, eff_w, offset):
    """Penalized log-likelihood with effective weights. [cite: 96]"""
    XW = _get_XW(X, preds, eff_w) 
    fisher_info_mtx = XW.T @ XW + 1e-10 * np.eye(X.shape[1]) 
    penalty = 0.5 * np.log(np.linalg.det(fisher_info_mtx)) 
    
    preds_clipped = np.clip(preds, 1e-15, 1 - 1e-15) 
    # Weighted log-likelihood [cite: 101]
    standard_loglik = np.sum(eff_w * (y * np.log(preds_clipped) + (1 - y) * np.log(1 - preds_clipped)))
    return -1 * (standard_loglik + penalty) 

def _get_XW(X, preds, eff_w, mask=None):
    """Weighted feature matrix calculation. [cite: 102]"""
    # XW = X * sqrt(eff_w * p * (1-p)) 
    rootW = np.sqrt(eff_w * preds * (1 - preds))
    XW = rootW[:, np.newaxis] * X
    if mask is not None: XW[:, mask] = 0.0
    return XW

def _bse(X, coefs, eff_w, offset):
    """Standard error computation with weights. [cite: 117, 122]"""
    preds = expit(X @ coefs + offset)
    XW = _get_XW(X, preds, eff_w)
    cov_matrix = np.linalg.pinv(XW.T @ XW + 1e-10 * np.eye(X.shape[1]))
    return np.sqrt(np.diag(cov_matrix))

# --- Supporting Logic ---

def _penalized_lrt(full_loglik, X, y, max_iter, max_stepsize, max_halfstep, tol, test_vars, eff_w, offset):
    """LRT with weights and offset support. [cite: 123, 131]"""
    test_var_indices = range(X.shape[1]) if test_vars is None else sorted(test_vars)
    pvals = np.full(X.shape[1], np.nan)
    for mask in test_var_indices:
        _, null_loglik, _ = _firth_newton_raphson(X, y, max_iter, max_stepsize, max_halfstep, tol, eff_w, offset, mask)
        pvals[mask] = chi2.sf(2 * (full_loglik - null_loglik), df=1) 
    return pvals

def _profile_likelihood_ci(X, y, fitted_coef, full_loglik, max_iter, max_stepsize, max_halfstep, tol, alpha, test_vars, eff_w, offset):
    """Profile Likelihood CI with weights and offset support. [cite: 141, 149]"""
    LL0 = full_loglik - chi2.ppf(1 - alpha, 1) / 2 
    test_var_indices = range(fitted_coef.shape[0]) if test_vars is None else sorted(test_vars)
    ci = np.full((fitted_coef.shape[0], 2), np.nan)

    for side_idx, side in enumerate([-1, 1]):
        for coef_idx in test_var_indices:
            coef = deepcopy(fitted_coef)
            for _ in range(max_iter):
                preds = expit(X @ coef + offset) 
                loglike = -_loglikelihood(X, y, preds, eff_w, offset)
                XW = _get_XW(X, preds, eff_w) 
                hat = _hat_diag(XW) 
                
                fisher_info_mtx = XW.T @ XW
                U_star = np.matmul(X.T, eff_w * (y - preds) + eff_w * np.multiply(hat, 0.5 - preds))
                inv_fisher = np.linalg.pinv(fisher_info_mtx) 
                
                underRoot = -2 * ((LL0 - loglike) + 0.5 * (U_star @ np.negative(inv_fisher) @ U_star)) / (inv_fisher[coef_idx, coef_idx] + 1e-10)
                lambda_ = 0 if underRoot < 0 else side * np.sqrt(underRoot) 
                U_star[coef_idx] += lambda_ 

                step_size = np.linalg.lstsq(fisher_info_mtx, U_star, rcond=None)[0] 
                coef += step_size 
                if abs(-_loglikelihood(X, y, expit(X @ coef + offset), eff_w, offset) - LL0) <= tol: 
                    ci[coef_idx, side_idx] = coef[coef_idx]
                    break
    return ci

def _hat_diag(XW):
    """Diagonal elements of weighted hat matrix. [cite: 113, 116]"""
    qr, tau, _, _ = lapack.dgeqrf(XW, overwrite_a=True)
    Q, _, _ = lapack.dorgqr(qr, tau, overwrite_a=True)
    return np.einsum("ij,ij->i", Q, Q)

def _wald_ci(coef, bse, alpha):
    z = norm.ppf(1 - alpha / 2)
    return np.column_stack([coef - z * bse, coef + z * bse])

def _wald_test(coef, bse): 
    return chi2.sf((coef / (bse + 1e-10))**2, df=1)


# In[17]:


import numpy as np
import pandas as pd
from scipy.special import expit

# --- 1. Generate Synthetic Aggregated Data ---
data = {
    'training_score': [10, 20, 40, 60, 80, 90],  # Predictor (X)
    'injuries':       [1,  0,  1,  0,  1,  0],   # Outcome (y)
    'employee_count': [5,  10, 8,  12, 5,  15],  # FREQ (How many people have this profile)
    'sampling_wt':    [1.1, 0.9, 1.0, 1.0, 1.2, 0.8], # WEIGHT (Importance)
    'exposure_yrs':   [1,  2,  5,  5,  10, 10]   # For OFFSET (Log of exposure)
}

df = pd.DataFrame(data)

# Pre-processing for the model
X = df[['training_score']].values
y = df['injuries'].values
freq = df['employee_count'].values
weights = df['sampling_wt'].values

# In SAS/Statistics, offset is usually the log of the exposure/time variable
offset = np.log(df['exposure_yrs'].values)

# --- 2. Initialize and Fit the Model ---
# (Using the class provided in the previous response)
model = FirthLogisticRegression(
    fit_const=True, 
    wald=False, 
    alpha=0.05
)

model.fit(X, y, weights=weights, freq=freq, offset=offset)

# --- 3. Interpretation & Predictions ---
print("--- Model Results ---")
print(f"Intercept (Const): {model.const_:.4f}")
print(f"Coefficient (Score): {model.coef_[0]:.4f}")

# Prediction: What is the probability for a new person?
# New employee: Score=50, Exposure=3 years.
X_new = np.array([[50]])
new_offset = np.log([3]) # Must provide the same scale of offset used in training

prob = model.predict_proba(X_new, offset=new_offset)

print("\n--- Prediction for New Data ---")
print(f"Employee with Score 50 and 3 years exposure:")
print(f"Probability of Injury: {prob[0][1]:.2%}")

# Comparison: If the same employee had 10 years exposure
higher_offset = np.log([10])
prob_long_term = model.predict_proba(X_new, offset=higher_offset)
print(f"Probability if exposure was 10 years: {prob_long_term[0][1]:.2%}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




