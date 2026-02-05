#!/usr/bin/env python
# coding: utf-8

# # GE

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


# In[2]:


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


# # DOU

# In[3]:


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


class FirthLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Firth's bias-reduced logistic regression (对齐 statsmodels Logit 接口)
    
    核心特性：
    - 属性命名与 statsmodels 一致：params, bse, pvalues, zvalues, ci, loglik, AIC, BIC
    - summary() 输出格式完全模仿 statsmodels Logit (含模型信息、系数表格、拟合统计量)
    - 保留 Firth 方法核心优势（解决分离问题、小样本偏倚校正）
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
        fit_intercept=True,  # 恢复 statsmodels 命名（原 fit_const）
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
        self.fit_intercept = fit_intercept  # 对齐 statsmodels：fit_intercept
        self.skip_pvals = skip_pvals
        self.skip_ci = skip_ci
        self.alpha = alpha
        self.wald = wald
        self.test_vars = test_vars

    def _more_tags(self):
        return {"binary_only": True}

    def _validate_input(self, X, y):
        # 原有输入验证逻辑（保持不变）
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
        
        # 分类目标验证（二进制）
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"Only binary classification supported (got {len(self.classes_)} classes)")
        y = LabelEncoder().fit_transform(y).astype(X.dtype, copy=False)
        
        self.n_samples_, self.n_features_ = X.shape  # 新增：样本数/特征数（用于统计量计算）
        return X, y

    def fit(self, X, y):
        X, y = self._validate_input(X, y)
        
        # 新增：记录因变量名称（对齐 statsmodels）
        self.endog_name_ = "y"  # 可通过参数自定义，默认 "y"
        self.exog_name_ = [f"x{i+1}" for i in range(self.n_features_)]  # 特征默认名称
        
        # 添加截距项（对齐 statsmodels：Intercept 在前）
        if self.fit_intercept:
            X = np.hstack((np.ones((self.n_samples_, 1)), X))  # Intercept 列（全1）在前
            self.exog_name_.insert(0, "Intercept")  # 截距项名称对齐 statsmodels
        
        # 拟合模型
        coef, self.loglik, self.n_iter_ = _firth_newton_raphson(
            X, y, self.max_iter, self.max_stepsize, self.max_halfstep, self.tol
        )
        
        # 计算标准误、p值、置信区间（对齐 statsmodels 属性命名）
        self.bse = _bse(X, coef)  # 对齐 statsmodels：bse（无下划线）
        self.params = coef  # 对齐 statsmodels：params（截距+特征系数，顺序一致）
        
        # 计算置信区间（对齐 statsmodels：ci 为 (n_params, 2) 数组）
        if not self.skip_ci:
            if not self.wald:
                self.ci = _profile_likelihood_ci(  # 对齐 statsmodels：ci（无下划线）
                    X=X, y=y, fitted_coef=coef, full_loglik=self.loglik,
                    max_iter=self.pl_max_iter, max_stepsize=self.pl_max_stepsize,
                    max_halfstep=self.pl_max_halfstep, tol=self.tol, alpha=self.alpha,
                    test_vars=self.test_vars
                )
            else:
                self.ci = _wald_ci(coef, self.bse, self.alpha)
        else:
            self.ci = np.full((self.params.shape[0], 2), np.nan)
        
        # 计算 p值和 z值（对齐 statsmodels：pvalues/zvalues）
        if not self.skip_pvals:
            if not self.wald:
                self.pvalues = _penalized_lrt(  # 对齐 statsmodels：pvalues（无下划线）
                    self.loglik, X, y, self.max_iter, self.max_stepsize,
                    self.max_halfstep, self.tol, self.test_vars
                )
            else:
                self.pvalues = _wald_test(coef, self.bse)
            # 计算 z值（statsmodels 核心输出：z = coef / bse）
            self.zvalues = self.params / (self.bse + 1e-10)  # 避免除以零
        else:
            self.pvalues = np.full(self.params.shape[0], np.nan)
            self.zvalues = np.full(self.params.shape[0], np.nan)
        
        # 分离截距和特征系数（兼容 sklearn 同时不破坏 statsmodels 接口）
        if self.fit_intercept:
            self.intercept_ = self.params[0]  # sklearn 风格（备用）
            self.coef_ = self.params[1:]      # sklearn 风格（备用）
        else:
            self.intercept_ = 0.0
            self.coef_ = self.params
        
        # 新增：计算 statsmodels 标志性统计量（AIC/BIC/McFadden R²）
        self.n_params_ = self.params.shape[0]  # 参数个数（截距+特征）
        self.AIC = 2 * self.n_params_ - 2 * self.loglik  # AIC 公式（对齐 statsmodels）
        self.BIC = self.n_params_ * np.log(self.n_samples_) - 2 * self.loglik  # BIC 公式
        # McFadden 伪 R²（参考 statsmodels 实现）
        null_model = _firth_newton_raphson(
            X[:, [0]] if self.fit_intercept else X[:, :0],  # 仅截距项（空模型）
            y, max_iter=self.max_iter, max_stepsize=self.max_stepsize,
            max_halfstep=self.max_halfstep, tol=self.tol
        )
        self.null_loglik = null_model[1]
        self.mcfadden_r2 = 1 - (self.loglik / self.null_loglik)  # McFadden R²

        return self

    def summary(self, xname=None, title=None):
        """
        完全对齐 statsmodels Logit 的 summary() 输出格式
        参数：
            xname: 特征名称列表（覆盖默认的 x1/x2/...）
            title: 摘要标题（默认 "Firth Logistic Regression Results"）
        """
        check_is_fitted(self)
        
        # 处理特征名称（优先使用用户输入 xname，否则用默认名）
        if xname is not None:
            if len(xname) != self.n_features_:
                raise ValueError(f"xname length ({len(xname)}) must match number of features ({self.n_features_})")
            exog_names = xname.copy()
            if self.fit_intercept:
                exog_names.insert(0, "Intercept")  # 截距项名称固定为 "Intercept"（对齐 statsmodels）
        else:
            exog_names = self.exog_name_
        
        # 1. 打印模型基本信息（完全模仿 statsmodels 格式）
        title = title or "Firth Logistic Regression Results"
        print("=" * len(title))
        print(title)
        print("=" * len(title))
        print(f"Dep. Variable:        {self.endog_name_}")
        print(f"Model:                FirthLogistic")
        print(f"Method:               ML (Firth bias-reduced)")
        print(f"Date:                 {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}" if 'pandas' in locals() else "")
        print(f"Time:                 {pd.Timestamp.now().strftime('%H:%M:%S')}" if 'pandas' in locals() else "")
        print(f"No. Observations:     {self.n_samples_}")
        print(f"Df Residuals:         {self.n_samples_ - self.n_params_}")
        print(f"Df Model:             {self.n_params_ - (1 if self.fit_intercept else 0)}")
        print(f"Pseudo R-squ. (McF):  {self.mcfadden_r2:.4f}")
        print(f"Log-Likelihood:       {self.loglik:.4f}")
        print(f"AIC:                  {self.AIC:.4f}")
        print(f"BIC:                  {self.BIC:.4f}")
        print(f"Converged:            {self.n_iter_ < self.max_iter}")
        print(f"Max Iterations:       {self.max_iter}")
        print()
        
        # 2. 打印系数表格（完全对齐 statsmodels 列名和格式）
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
            # 格式处理（NaN 显示为 "."，p值显著性标记）
            coef_str = f"{coef:.4f}" if not np.isnan(coef) else "."
            se_str = f"{se:.4f}" if not np.isnan(se) else "."
            z_str = f"{z:.4f}" if not np.isnan(z) else "."
            pval_str = f"{pval:.4f}" if not np.isnan(pval) else "."
            # 显著性标记（***p<0.001, **p<0.01, *p<0.05, .p<0.1）
            sig_mark = ""
            if not np.isnan(pval):
                if pval < 0.001:
                    sig_mark = "***"
                elif pval < 0.01:
                    sig_mark = "**"
                elif pval < 0.05:
                    sig_mark = "*"
                elif pval < 0.1:
                    sig_mark = "."
            ci_str = f"{ci_lower:.4f}  {ci_upper:.4f}" if not (np.isnan(ci_lower) or np.isnan(ci_upper)) else ".        ."
            # 打印行（对齐格式）
            print(f"{i+1:<2} {var_name:<12} {coef_str:<10} {se_str:<10} {z_str:<8} {pval_str:<8} {ci_str} {sig_mark}")
        
        # 3. 打印显著性说明（模仿 statsmodels）
        print(separator)
        print("Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1")
        print()
        print(f"Note: Confidence interval method = {'Wald' if self.wald else 'Profile Likelihood'}")
        if self.wald:
            print("Note: Wald intervals are faster but less robust for small samples/separation.")

    def decision_function(self, X):
        check_is_fitted(self)
        try:
            X = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to convert X to float64 array: {str(e)}")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got {X.ndim}D")
        # 对齐 statsmodels：线性预测值 = X@coef + intercept
        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        decision = self.decision_function(X)
        indices = (decision > 0).astype(int)
        return self.classes_[indices]

    def predict_proba(self, X):
        decision = self.decision_function(X)
        proba_1 = expit(decision)
        return np.column_stack([1 - proba_1, proba_1])


# ------------------------------
# 核心辅助函数（保持不变，仅适配参数名称）
# ------------------------------
def _firth_newton_raphson(X, y, max_iter, max_stepsize, max_halfstep, tol, mask=None):
    coef = np.zeros(X.shape[1])
    for iter in range(1, max_iter + 1):
        preds = expit(X @ coef)
        XW = _get_XW(X, preds, mask)
        fisher_info_mtx = XW.T @ XW
        hat = _hat_diag(XW)
        U_star = np.matmul(X.T, y - preds + np.multiply(hat, 0.5 - preds))
        
        step_size = np.linalg.lstsq(fisher_info_mtx, U_star, rcond=None)[0]
        mx = np.max(np.abs(step_size)) / max_stepsize
        if mx > 1:
            step_size /= mx
        
        coef_new = coef + step_size
        preds_new = expit(X @ coef_new)
        loglike = _loglikelihood(X, y, preds)
        loglike_new = _loglikelihood(X, y, preds_new)
        steps = 0
        
        while loglike < loglike_new:
            step_size *= 0.5
            coef_new = coef + step_size
            preds_new = expit(X @ coef_new)
            loglike_new = _loglikelihood(X, y, preds_new)
            steps += 1
            if steps == max_halfstep:
                warnings.warn("Step-halving failed to converge (max halfsteps reached)", ConvergenceWarning, stacklevel=2)
                return coef_new, -loglike_new, iter
        
        if iter > 1 and np.linalg.norm(coef_new - coef) < tol:
            return coef_new, -loglike_new, iter
        coef = coef_new
    
    warnings.warn(f"Newton-Raphson failed to converge (max_iter={max_iter} reached)", ConvergenceWarning, stacklevel=2)
    return coef, -loglike_new, max_iter


def _loglikelihood(X, y, preds):
    XW = _get_XW(X, preds)
    fisher_info_mtx = XW.T @ XW + 1e-10 * np.eye(XW.shape[1])
    penalty = 0.5 * np.log(np.linalg.det(fisher_info_mtx))
    preds_clipped = np.clip(preds, 1e-15, 1 - 1e-15)
    standard_loglik = np.sum(y * np.log(preds_clipped) + (1 - y) * np.log(1 - preds_clipped))
    return -1 * (standard_loglik + penalty)


def _get_XW(X, preds, mask=None):
    rootW = np.sqrt(preds * (1 - preds))
    XW = rootW[:, np.newaxis] * X
    if mask is not None:
        XW[:, mask] = 0.0
    return XW


def _get_aug_XW(X, preds, hats):
    rootW = np.sqrt(preds * (1 - preds) * (1 + hats))
    XW_aug = rootW[:, np.newaxis] * X
    return XW_aug


def _hat_diag(XW):
    qr, tau, _, _ = lapack.dgeqrf(XW, overwrite_a=True)
    Q, _, _ = lapack.dorgqr(qr, tau, overwrite_a=True)
    return np.einsum("ij,ij->i", Q, Q)


def _bse(X, coefs):
    preds = expit(X @ coefs)
    XW = _get_XW(X, preds)
    fisher_info_mtx = XW.T @ XW + 1e-10 * np.eye(XW.shape[1])
    cov_matrix = np.linalg.pinv(fisher_info_mtx)
    return np.sqrt(np.diag(cov_matrix))


def _penalized_lrt(full_loglik, X, y, max_iter, max_stepsize, max_halfstep, tol, test_vars):
    test_var_indices = range(X.shape[1]) if test_vars is None else (
        [test_vars] if isinstance(test_vars, int) else sorted(test_vars)
    )
    pvals = []
    for mask in test_var_indices:
        _, null_loglik, _ = _firth_newton_raphson(X, y, max_iter, max_stepsize, max_halfstep, tol, mask)
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


def _predict(X, coef):
    preds = expit(X @ coef)
    np.clip(preds, a_min=1e-15, a_max=1 - 1e-15, out=preds)
    return preds


def _profile_likelihood_ci(X, y, fitted_coef, full_loglik, max_iter, max_stepsize, max_halfstep, tol, alpha, test_vars):
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
                preds = _predict(X, coef)
                loglike = -_loglikelihood(X, y, preds)
                XW = _get_XW(X, preds)
                hat = _hat_diag(XW)
                XW_aug = _get_aug_XW(X, preds, hat)
                fisher_info_mtx = XW_aug.T @ XW_aug
                U_star = np.matmul(X.T, y - preds + np.multiply(hat, 0.5 - preds))
                
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
                    preds = _predict(X, coef)
                    loglike = -_loglikelihood(X, y, preds)
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
# 示例数据集和测试代码（验证 statsmodels 兼容性）
# ------------------------------
def load_sex2():
    try:
        with open_text("firthlogist.datasets", "sex2.csv") as f:
            X = np.loadtxt(f, skiprows=1, delimiter=",")
        y = X[:, 0]
        X = X[:, 1:]
        return X, y, ["age", "oc", "vic", "vicl", "vis", "dia"]
    except FileNotFoundError:
        warnings.warn("sex2 dataset not found (optional for testing)", stacklevel=2)
        return None, None, None


def load_endometrial():
    try:
        with open_text("firthlogist.datasets", "endometrial.csv") as f:
            X = np.loadtxt(f, skiprows=1, delimiter=",")
        y = X[:, -1]
        X = X[:, :-1]
        return X, y, ["NV", "PI", "EH"]
    except FileNotFoundError:
        warnings.warn("endometrial dataset not found (optional for testing)", stacklevel=2)
        return None, None, None


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


class FirthLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Firth's bias-reduced logistic regression (对齐 statsmodels Logit 接口)
    
    核心特性：
    - 属性命名与 statsmodels 一致：params, bse, pvalues, zvalues, ci, loglik, AIC, BIC
    - summary() 输出格式完全模仿 statsmodels Logit (含模型信息、系数表格、拟合统计量)
    - 保留 Firth 方法核心优势（解决分离问题、小样本偏倚校正）
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
        fit_intercept=True,  # 恢复 statsmodels 命名（原 fit_const）
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
        self.fit_intercept = fit_intercept  # 对齐 statsmodels：fit_intercept
        self.skip_pvals = skip_pvals
        self.skip_ci = skip_ci
        self.alpha = alpha
        self.wald = wald
        self.test_vars = test_vars

    def _more_tags(self):
        return {"binary_only": True}

    def _validate_input(self, X, y):
        # 原有输入验证逻辑（保持不变）
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
        
        # 分类目标验证（二进制）
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"Only binary classification supported (got {len(self.classes_)} classes)")
        y = LabelEncoder().fit_transform(y).astype(X.dtype, copy=False)
        
        self.n_samples_, self.n_features_ = X.shape  # 新增：样本数/特征数（用于统计量计算）
        return X, y

    def fit(self, X, y):
        X, y = self._validate_input(X, y)
        
        # 新增：记录因变量名称（对齐 statsmodels）
        self.endog_name_ = "y"  # 可通过参数自定义，默认 "y"
        self.exog_name_ = [f"x{i+1}" for i in range(self.n_features_)]  # 特征默认名称
        
        # 添加截距项（对齐 statsmodels：Intercept 在前）
        if self.fit_intercept:
            X = np.hstack((np.ones((self.n_samples_, 1)), X))  # Intercept 列（全1）在前
            self.exog_name_.insert(0, "Intercept")  # 截距项名称对齐 statsmodels
        
        # 拟合模型
        coef, self.loglik, self.n_iter_ = _firth_newton_raphson(
            X, y, self.max_iter, self.max_stepsize, self.max_halfstep, self.tol
        )
        
        # 计算标准误、p值、置信区间（对齐 statsmodels 属性命名）
        self.bse = _bse(X, coef)  # 对齐 statsmodels：bse（无下划线）
        self.params = coef  # 对齐 statsmodels：params（截距+特征系数，顺序一致）
        
        # 计算置信区间（对齐 statsmodels：ci 为 (n_params, 2) 数组）
        if not self.skip_ci:
            if not self.wald:
                self.ci = _profile_likelihood_ci(  # 对齐 statsmodels：ci（无下划线）
                    X=X, y=y, fitted_coef=coef, full_loglik=self.loglik,
                    max_iter=self.pl_max_iter, max_stepsize=self.pl_max_stepsize,
                    max_halfstep=self.pl_max_halfstep, tol=self.tol, alpha=self.alpha,
                    test_vars=self.test_vars
                )
            else:
                self.ci = _wald_ci(coef, self.bse, self.alpha)
        else:
            self.ci = np.full((self.params.shape[0], 2), np.nan)
        
        # 计算 p值和 z值（对齐 statsmodels：pvalues/zvalues）
        if not self.skip_pvals:
            if not self.wald:
                self.pvalues = _penalized_lrt(  # 对齐 statsmodels：pvalues（无下划线）
                    self.loglik, X, y, self.max_iter, self.max_stepsize,
                    self.max_halfstep, self.tol, self.test_vars
                )
            else:
                self.pvalues = _wald_test(coef, self.bse)
            # 计算 z值（statsmodels 核心输出：z = coef / bse）
            self.zvalues = self.params / (self.bse + 1e-10)  # 避免除以零
        else:
            self.pvalues = np.full(self.params.shape[0], np.nan)
            self.zvalues = np.full(self.params.shape[0], np.nan)
        
        # 分离截距和特征系数（兼容 sklearn 同时不破坏 statsmodels 接口）
        if self.fit_intercept:
            self.intercept_ = self.params[0]  # sklearn 风格（备用）
            self.coef_ = self.params[1:]      # sklearn 风格（备用）
        else:
            self.intercept_ = 0.0
            self.coef_ = self.params
        
        # 新增：计算 statsmodels 标志性统计量（AIC/BIC/McFadden R²）
        self.n_params_ = self.params.shape[0]  # 参数个数（截距+特征）
        self.AIC = 2 * self.n_params_ - 2 * self.loglik  # AIC 公式（对齐 statsmodels）
        self.BIC = self.n_params_ * np.log(self.n_samples_) - 2 * self.loglik  # BIC 公式
        # McFadden 伪 R²（参考 statsmodels 实现）
        null_model = _firth_newton_raphson(
            X[:, [0]] if self.fit_intercept else X[:, :0],  # 仅截距项（空模型）
            y, max_iter=self.max_iter, max_stepsize=self.max_stepsize,
            max_halfstep=self.max_halfstep, tol=self.tol
        )
        self.null_loglik = null_model[1]
        self.mcfadden_r2 = 1 - (self.loglik / self.null_loglik)  # McFadden R²

        return self

    def summary(self, xname=None, title=None):
        """
        完全对齐 statsmodels Logit 的 summary() 输出格式
        参数：
            xname: 特征名称列表（覆盖默认的 x1/x2/...）
            title: 摘要标题（默认 "Firth Logistic Regression Results"）
        """
        check_is_fitted(self)
        
        # 处理特征名称（优先使用用户输入 xname，否则用默认名）
        if xname is not None:
            if len(xname) != self.n_features_:
                raise ValueError(f"xname length ({len(xname)}) must match number of features ({self.n_features_})")
            exog_names = xname.copy()
            if self.fit_intercept:
                exog_names.insert(0, "Intercept")  # 截距项名称固定为 "Intercept"（对齐 statsmodels）
        else:
            exog_names = self.exog_name_
        
        # 1. 打印模型基本信息（完全模仿 statsmodels 格式）
        title = title or "Firth Logistic Regression Results"
        print("=" * len(title))
        print(title)
        print("=" * len(title))
        print(f"Dep. Variable:        {self.endog_name_}")
        print(f"Model:                FirthLogistic")
        print(f"Method:               ML (Firth bias-reduced)")
        print(f"Date:                 {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}" if 'pandas' in locals() else "")
        print(f"Time:                 {pd.Timestamp.now().strftime('%H:%M:%S')}" if 'pandas' in locals() else "")
        print(f"No. Observations:     {self.n_samples_}")
        print(f"Df Residuals:         {self.n_samples_ - self.n_params_}")
        print(f"Df Model:             {self.n_params_ - (1 if self.fit_intercept else 0)}")
        print(f"Pseudo R-squ. (McF):  {self.mcfadden_r2:.4f}")
        print(f"Log-Likelihood:       {self.loglik:.4f}")
        print(f"AIC:                  {self.AIC:.4f}")
        print(f"BIC:                  {self.BIC:.4f}")
        print(f"Converged:            {self.n_iter_ < self.max_iter}")
        print(f"Max Iterations:       {self.max_iter}")
        print()
        
        # 2. 打印系数表格（完全对齐 statsmodels 列名和格式）
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
            # 格式处理（NaN 显示为 "."，p值显著性标记）
            coef_str = f"{coef:.4f}" if not np.isnan(coef) else "."
            se_str = f"{se:.4f}" if not np.isnan(se) else "."
            z_str = f"{z:.4f}" if not np.isnan(z) else "."
            pval_str = f"{pval:.4f}" if not np.isnan(pval) else "."
            # 显著性标记（***p<0.001, **p<0.01, *p<0.05, .p<0.1）
            sig_mark = ""
            if not np.isnan(pval):
                if pval < 0.001:
                    sig_mark = "***"
                elif pval < 0.01:
                    sig_mark = "**"
                elif pval < 0.05:
                    sig_mark = "*"
                elif pval < 0.1:
                    sig_mark = "."
            ci_str = f"{ci_lower:.4f}  {ci_upper:.4f}" if not (np.isnan(ci_lower) or np.isnan(ci_upper)) else ".        ."
            # 打印行（对齐格式）
            print(f"{i+1:<2} {var_name:<12} {coef_str:<10} {se_str:<10} {z_str:<8} {pval_str:<8} {ci_str} {sig_mark}")
        
        # 3. 打印显著性说明（模仿 statsmodels）
        print(separator)
        print("Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1")
        print()
        print(f"Note: Confidence interval method = {'Wald' if self.wald else 'Profile Likelihood'}")
        if self.wald:
            print("Note: Wald intervals are faster but less robust for small samples/separation.")

    def decision_function(self, X):
        check_is_fitted(self)
        try:
            X = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to convert X to float64 array: {str(e)}")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got {X.ndim}D")
        # 对齐 statsmodels：线性预测值 = X@coef + intercept
        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        decision = self.decision_function(X)
        indices = (decision > 0).astype(int)
        return self.classes_[indices]

    def predict_proba(self, X):
        decision = self.decision_function(X)
        proba_1 = expit(decision)
        return np.column_stack([1 - proba_1, proba_1])


# ------------------------------
# 核心辅助函数（保持不变，仅适配参数名称）
# ------------------------------
def _firth_newton_raphson(X, y, max_iter, max_stepsize, max_halfstep, tol, mask=None):
    coef = np.zeros(X.shape[1])
    for iter in range(1, max_iter + 1):
        preds = expit(X @ coef)
        XW = _get_XW(X, preds, mask)
        fisher_info_mtx = XW.T @ XW
        hat = _hat_diag(XW)
        U_star = np.matmul(X.T, y - preds + np.multiply(hat, 0.5 - preds))
        
        step_size = np.linalg.lstsq(fisher_info_mtx, U_star, rcond=None)[0]
        mx = np.max(np.abs(step_size)) / max_stepsize
        if mx > 1:
            step_size /= mx
        
        coef_new = coef + step_size
        preds_new = expit(X @ coef_new)
        loglike = _loglikelihood(X, y, preds)
        loglike_new = _loglikelihood(X, y, preds_new)
        steps = 0
        
        while loglike < loglike_new:
            step_size *= 0.5
            coef_new = coef + step_size
            preds_new = expit(X @ coef_new)
            loglike_new = _loglikelihood(X, y, preds_new)
            steps += 1
            if steps == max_halfstep:
                warnings.warn("Step-halving failed to converge (max halfsteps reached)", ConvergenceWarning, stacklevel=2)
                return coef_new, -loglike_new, iter
        
        if iter > 1 and np.linalg.norm(coef_new - coef) < tol:
            return coef_new, -loglike_new, iter
        coef = coef_new
    
    warnings.warn(f"Newton-Raphson failed to converge (max_iter={max_iter} reached)", ConvergenceWarning, stacklevel=2)
    return coef, -loglike_new, max_iter


def _loglikelihood(X, y, preds):
    XW = _get_XW(X, preds)
    fisher_info_mtx = XW.T @ XW + 1e-10 * np.eye(XW.shape[1])
    penalty = 0.5 * np.log(np.linalg.det(fisher_info_mtx))
    preds_clipped = np.clip(preds, 1e-15, 1 - 1e-15)
    standard_loglik = np.sum(y * np.log(preds_clipped) + (1 - y) * np.log(1 - preds_clipped))
    return -1 * (standard_loglik + penalty)


def _get_XW(X, preds, mask=None):
    rootW = np.sqrt(preds * (1 - preds))
    XW = rootW[:, np.newaxis] * X
    if mask is not None:
        XW[:, mask] = 0.0
    return XW


def _get_aug_XW(X, preds, hats):
    rootW = np.sqrt(preds * (1 - preds) * (1 + hats))
    XW_aug = rootW[:, np.newaxis] * X
    return XW_aug


def _hat_diag(XW):
    qr, tau, _, _ = lapack.dgeqrf(XW, overwrite_a=True)
    Q, _, _ = lapack.dorgqr(qr, tau, overwrite_a=True)
    return np.einsum("ij,ij->i", Q, Q)


def _bse(X, coefs):
    preds = expit(X @ coefs)
    XW = _get_XW(X, preds)
    fisher_info_mtx = XW.T @ XW + 1e-10 * np.eye(XW.shape[1])
    cov_matrix = np.linalg.pinv(fisher_info_mtx)
    return np.sqrt(np.diag(cov_matrix))


def _penalized_lrt(full_loglik, X, y, max_iter, max_stepsize, max_halfstep, tol, test_vars):
    test_var_indices = range(X.shape[1]) if test_vars is None else (
        [test_vars] if isinstance(test_vars, int) else sorted(test_vars)
    )
    pvals = []
    for mask in test_var_indices:
        _, null_loglik, _ = _firth_newton_raphson(X, y, max_iter, max_stepsize, max_halfstep, tol, mask)
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


def _predict(X, coef):
    preds = expit(X @ coef)
    np.clip(preds, a_min=1e-15, a_max=1 - 1e-15, out=preds)
    return preds


def _profile_likelihood_ci(X, y, fitted_coef, full_loglik, max_iter, max_stepsize, max_halfstep, tol, alpha, test_vars):
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
                preds = _predict(X, coef)
                loglike = -_loglikelihood(X, y, preds)
                XW = _get_XW(X, preds)
                hat = _hat_diag(XW)
                XW_aug = _get_aug_XW(X, preds, hat)
                fisher_info_mtx = XW_aug.T @ XW_aug
                U_star = np.matmul(X.T, y - preds + np.multiply(hat, 0.5 - preds))
                
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
                    preds = _predict(X, coef)
                    loglike = -_loglikelihood(X, y, preds)
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
# 示例数据集和测试代码（验证 statsmodels 兼容性）
# ------------------------------
def load_sex2():
    try:
        with open_text("firthlogist.datasets", "sex2.csv") as f:
            X = np.loadtxt(f, skiprows=1, delimiter=",")
        y = X[:, 0]
        X = X[:, 1:]
        return X, y, ["age", "oc", "vic", "vicl", "vis", "dia"]
    except FileNotFoundError:
        warnings.warn("sex2 dataset not found (optional for testing)", stacklevel=2)
        return None, None, None


def load_endometrial():
    try:
        with open_text("firthlogist.datasets", "endometrial.csv") as f:
            X = np.loadtxt(f, skiprows=1, delimiter=",")
        y = X[:, -1]
        X = X[:, :-1]
        return X, y, ["NV", "PI", "EH"]
    except FileNotFoundError:
        warnings.warn("endometrial dataset not found (optional for testing)", stacklevel=2)
        return None, None, None


# In[4]:


# 模拟数据（验证接口兼容性）
np.random.seed(42)
n = 50
X = np.random.normal(0, 1, (n, 2))
y = (X[:, 0] > 0).astype(int)
y[np.random.choice(n, 5)] = 1 - y[np.random.choice(n, 5)]

# 1. 拟合模型（完全模仿 statsmodels 用法）
model = FirthLogisticRegression(fit_intercept=True, wald=False, alpha=0.05)
model.fit(X, y)

# 2. 访问属性（与 statsmodels 完全一致）
print("=== 对齐 statsmodels 属性访问 ===")
print(f"params (系数): {model.params.round(4)}")
print(f"bse (标准误): {model.bse.round(4)}")
print(f"zvalues (z统计量): {model.zvalues.round(4)}")
print(f"pvalues (p值): {model.pvalues.round(4)}")
print(f"ci (95%置信区间): \n{model.ci.round(4)}")
print(f"AIC: {model.AIC.round(4)}")
print(f"BIC: {model.BIC.round(4)}")
print()

# 3. 打印 summary（与 statsmodels 格式一致）
print("=== 对齐 statsmodels summary 输出 ===")
model.summary(xname=["FeatureA", "FeatureB"], title="Firth Logistic Regression (Statsmodels Compatible)")

# 4. 对比 statsmodels Logit（可选验证）
try:
    import statsmodels.api as sm
    X_sm = sm.add_constant(X)  # statsmodels 需手动加截距
    sm_model = sm.Logit(y, X_sm).fit(disp=0)
    print("\n=== statsmodels Logit Summary（对比用）===")
    sm_model.summary()
except ImportError:
    print("\n提示：未安装 statsmodels，跳过对比验证")


# In[5]:


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
import pandas as pd  # 补充必要依赖（用于时间格式化）

class FirthLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Firth's bias-reduced logistic regression（对齐 statsmodels Logit + SAS proc logistic 功能）
    
    核心特性：
    - 接口完全对齐 statsmodels Logit：params, bse, pvalues, zvalues, ci 等属性
    - 新增 SAS 风格 weight/freq/offset 选项，行为与 proc logistic 完全一致
    - 保留 Firth 方法优势（解决分离问题、小样本偏倚校正）
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
        # 新增：SAS proc logistic 风格 weight/freq/offset 选项
        weight=None,  # 对应 SAS weight=var：观测权重（非负数值）
        freq=None,    # 对应 SAS freq=var：观测频数（非负整数，等效重复观测）
        offset=None,  # 对应 SAS offset=var：固定系数为1的偏移项（不参与拟合）
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
        # 新增实例变量（SAS 兼容）
        self.weight = weight
        self.freq = freq
        self.offset = offset

    def _more_tags(self):
        return {"binary_only": True}

    def _validate_input(self, X, y):
        # 原有输入验证逻辑
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
        
        # 分类目标验证（二进制）
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"Only binary classification supported (got {len(self.classes_)} classes)")
        y = LabelEncoder().fit_transform(y).astype(X.dtype, copy=False)
        
        self.n_samples_, self.n_features_ = X.shape
        
        # --------------------------
        # 新增：SAS 风格 weight/freq/offset 验证（严格匹配 proc logistic 行为）
        # --------------------------
        # 1. weight 和 freq 互斥（SAS 核心规则）
        if self.weight is not None and self.freq is not None:
            raise ValueError("weight and freq cannot be used simultaneously (SAS proc logistic compatible)")
        
        # 2. 处理有效权重（weight/freq 二选一，默认全1）
        self.effective_weight_ = np.ones(self.n_samples_, dtype=np.float64)
        if self.weight is not None:
            # SAS weight：非负数值，加权似然估计
            weight = np.asarray(self.weight, dtype=np.float64)
            if weight.ndim != 1 or weight.shape[0] != self.n_samples_:
                raise ValueError(f"weight must be 1D array with length {self.n_samples_} (match sample count)")
            if np.any(weight < 0):
                raise ValueError("weight cannot contain negative values (SAS proc logistic rule)")
            self.effective_weight_ = weight
        elif self.freq is not None:
            # SAS freq：非负整数，等效重复观测（非整数警告）
            freq = np.asarray(self.freq, dtype=np.float64)
            if freq.ndim != 1 or freq.shape[0] != self.n_samples_:
                raise ValueError(f"freq must be 1D array with length {self.n_samples_} (match sample count)")
            if np.any(freq < 0):
                raise ValueError("freq cannot contain negative values (SAS proc logistic rule)")
            if not np.allclose(freq, np.round(freq)):
                warnings.warn("freq should be non-negative integers (SAS proc logistic expects integers)", UserWarning, stacklevel=2)
            self.effective_weight_ = freq
        
        # 3. 处理 offset（SAS offset：固定系数为1，不参与模型估计）
        self.offset_ = np.zeros(self.n_samples_, dtype=np.float64)
        if self.offset is not None:
            offset = np.asarray(self.offset, dtype=np.float64)
            if offset.ndim != 1 or offset.shape[0] != self.n_samples_:
                raise ValueError(f"offset must be 1D array with length {self.n_samples_} (match sample count)")
            self.offset_ = offset

        return X, y

    def fit(self, X, y):
        X, y = self._validate_input(X, y)
        
        # 记录变量名称（对齐 statsmodels）
        self.endog_name_ = "y"
        self.exog_name_ = [f"x{i+1}" for i in range(self.n_features_)]
        
        # 添加截距项（Intercept 在前，对齐 statsmodels）
        if self.fit_intercept:
            X = np.hstack((np.ones((self.n_samples_, 1)), X))
            self.exog_name_.insert(0, "Intercept")
        
        # 拟合模型（传入有效权重和 offset，兼容 Firth 方法）
        coef, self.loglik, self.n_iter_ = _firth_newton_raphson(
            X, y, self.max_iter, self.max_stepsize, self.max_halfstep, self.tol,
            effective_weight=self.effective_weight_,
            offset=self.offset_
        )
        
        # 计算标准误（加权版）
        self.bse = _bse(X, coef, self.effective_weight_, self.offset_)
        self.params = coef
        
        # 计算置信区间（加权版）
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
        
        # 计算 p值和 z值（加权版）
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
            self.zvalues = self.params / (self.bse + 1e-10)  # 避免除以零
        else:
            self.pvalues = np.full(self.params.shape[0], np.nan)
            self.zvalues = np.full(self.params.shape[0], np.nan)
        
        # 分离截距和特征系数（兼容 sklearn）
        if self.fit_intercept:
            self.intercept_ = self.params[0]
            self.coef_ = self.params[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self.params
        
        # 计算 statsmodels 风格统计量（加权版）
        self.n_params_ = self.params.shape[0]
        self.AIC = 2 * self.n_params_ - 2 * self.loglik
        self.BIC = self.n_params_ * np.log(self.n_samples_) - 2 * self.loglik
        # 加权空模型（用于 McFadden R²）
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
        """完全对齐 statsmodels Logit 格式，新增 SAS 选项使用状态显示"""
        check_is_fitted(self)
        
        # 处理特征名称
        if xname is not None:
            if len(xname) != self.n_features_:
                raise ValueError(f"xname length ({len(xname)}) must match number of features ({self.n_features_})")
            exog_names = xname.copy()
            if self.fit_intercept:
                exog_names.insert(0, "Intercept")
        else:
            exog_names = self.exog_name_
        
        # 1. 模型基本信息（新增 SAS 选项状态）
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
        print(f"Offset Used:          {self.offset is not None} (SAS style)")
        print(f"Df Residuals:         {self.n_samples_ - self.n_params_}")
        print(f"Df Model:             {self.n_params_ - (1 if self.fit_intercept else 0)}")
        print(f"Pseudo R-squ. (McF):  {self.mcfadden_r2:.4f}")
        print(f"Log-Likelihood:       {self.loglik:.4f}")
        print(f"AIC:                  {self.AIC:.4f}")
        print(f"BIC:                  {self.BIC:.4f}")
        print(f"Converged:            {self.n_iter_ < self.max_iter}")
        print(f"Max Iterations:       {self.max_iter}")
        print()
        
        # 2. 系数表格（保持 statsmodels 格式）
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
            # 显著性标记
            sig_mark = ""
            if not np.isnan(pval):
                sig_mark = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "." if pval < 0.1 else ""
            ci_str = f"{ci_lower:.4f}  {ci_upper:.4f}" if not (np.isnan(ci_lower) or np.isnan(ci_upper)) else ".        ."
            print(f"{i+1:<2} {var_name:<12} {coef_str:<10} {se_str:<10} {z_str:<8} {pval_str:<8} {ci_str} {sig_mark}")
        
        # 3. 显著性说明和备注
        print(separator)
        print("Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1")
        print()
        print(f"Note: Confidence interval method = {'Wald' if self.wald else 'Profile Likelihood'}")
        print(f"Note: SAS proc logistic compatible options: weight={self.weight is not None}, freq={self.freq is not None}, offset={self.offset is not None}")
        if self.wald:
            print("Note: Wald intervals are faster but less robust for small samples/separation.")

    def decision_function(self, X):
        """对齐 SAS offset 行为：线性预测值 = X@coef + intercept + offset"""
        check_is_fitted(self)
        try:
            X = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to convert X to float64 array: {str(e)}")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got {X.ndim}D")
        # 包含 offset（与训练时一致）
        return X @ self.coef_ + self.intercept_ + self.offset_

    def predict(self, X):
        decision = self.decision_function(X)
        indices = (decision > 0).astype(int)
        return self.classes_[indices]

    def predict_proba(self, X):
        decision = self.decision_function(X)
        proba_1 = expit(decision)
        return np.column_stack([1 - proba_1, proba_1])


# ------------------------------
# 核心辅助函数（适配 weight/offset，保持 Firth 方法逻辑）
# ------------------------------
def _firth_newton_raphson(X, y, max_iter, max_stepsize, max_halfstep, tol, mask=None, effective_weight=None, offset=None):
    effective_weight = np.ones(X.shape[0]) if effective_weight is None else effective_weight
    offset = np.zeros(X.shape[0]) if offset is None else offset
    
    coef = np.zeros(X.shape[1])
    for iter in range(1, max_iter + 1):
        # 线性预测值 = X@coef + offset（SAS offset 核心逻辑）
        linear_pred = X @ coef + offset
        preds = expit(linear_pred)
        
        # 加权特征矩阵（SAS weight/freq 作用于似然）
        rootW = np.sqrt(preds * (1 - preds) * effective_weight)
        XW = rootW[:, np.newaxis] * X
        if mask is not None:
            XW[:, mask] = 0.0
        fisher_info_mtx = XW.T @ XW
        
        # 加权修正得分向量（Firth 方法 + SAS 权重）
        hat = _hat_diag(XW)
        U_star = np.matmul(X.T, effective_weight * (y - preds + np.multiply(hat, 0.5 - preds)))
        
        # 步长计算和限制
        step_size = np.linalg.lstsq(fisher_info_mtx, U_star, rcond=None)[0]
        mx = np.max(np.abs(step_size)) / max_stepsize
        if mx > 1:
            step_size /= mx
        
        # 系数更新和步长减半（确保似然递增）
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
        
        # 收敛判断
        if iter > 1 and np.linalg.norm(coef_new - coef) < tol:
            return coef_new, -loglike_new, iter
        coef = coef_new
    
    warnings.warn(f"Newton-Raphson failed to converge (max_iter={max_iter} reached)", ConvergenceWarning, stacklevel=2)
    return coef, -loglike_new, max_iter


def _loglikelihood(X, y, preds, effective_weight=None):
    """加权版 Firth 似然函数（适配 SAS weight/freq）"""
    effective_weight = np.ones(X.shape[0]) if effective_weight is None else effective_weight
    
    XW = _get_XW(X, preds, effective_weight=effective_weight)
    fisher_info_mtx = XW.T @ XW + 1e-10 * np.eye(XW.shape[1])  # 数值稳定
    penalty = 0.5 * np.log(np.linalg.det(fisher_info_mtx))
    
    # 加权标准似然（SAS weight/freq 核心作用点）
    preds_clipped = np.clip(preds, 1e-15, 1 - 1e-15)
    standard_loglik = np.sum(effective_weight * (y * np.log(preds_clipped) + (1 - y) * np.log(1 - preds_clipped)))
    
    return -1 * (standard_loglik + penalty)


def _get_XW(X, preds, mask=None, effective_weight=None):
    """加权特征矩阵（适配 SAS weight/freq）"""
    effective_weight = np.ones(X.shape[0]) if effective_weight is None else effective_weight
    rootW = np.sqrt(preds * (1 - preds) * effective_weight)
    XW = rootW[:, np.newaxis] * X
    if mask is not None:
        XW[:, mask] = 0.0
    return XW


def _get_aug_XW(X, preds, hats, effective_weight=None):
    """加权增强特征矩阵（适配 SAS weight/freq）"""
    effective_weight = np.ones(X.shape[0]) if effective_weight is None else effective_weight
    rootW = np.sqrt(preds * (1 - preds) * (1 + hats) * effective_weight)
    XW_aug = rootW[:, np.newaxis] * X
    return XW_aug


def _hat_diag(XW):
    qr, tau, _, _ = lapack.dgeqrf(XW, overwrite_a=True)
    Q, _, _ = lapack.dorgqr(qr, tau, overwrite_a=True)
    return np.einsum("ij,ij->i", Q, Q)


def _bse(X, coefs, effective_weight=None, offset=None):
    """加权版标准误（适配 SAS weight/offset）"""
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
    """加权版似然比检验（适配 SAS weight/offset）"""
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
    """带 offset 的预测概率（适配 SAS offset）"""
    offset = np.zeros(X.shape[0]) if offset is None else offset
    linear_pred = X @ coef + offset
    preds = expit(linear_pred)
    np.clip(preds, a_min=1e-15, a_max=1 - 1e-15, out=preds)
    return preds


def _profile_likelihood_ci(
    X, y, fitted_coef, full_loglik, max_iter, max_stepsize, max_halfstep, tol, alpha, test_vars,
    effective_weight=None, offset=None
):
    """加权版轮廓似然置信区间（适配 SAS weight/offset）"""
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
# 示例数据集和测试代码（验证 SAS 选项功能）
# ------------------------------
def load_sex2():
    try:
        with open_text("firthlogist.datasets", "sex2.csv") as f:
            X = np.loadtxt(f, skiprows=1, delimiter=",")
        y = X[:, 0]
        X = X[:, 1:]
        return X, y, ["age", "oc", "vic", "vicl", "vis", "dia"]
    except FileNotFoundError:
        warnings.warn("sex2 dataset not found (optional for testing)", stacklevel=2)
        return None, None, None


def load_endometrial():
    try:
        with open_text("firthlogist.datasets", "endometrial.csv") as f:
            X = np.loadtxt(f, skiprows=1, delimiter=",")
        y = X[:, -1]
        X = X[:, :-1]
        return X, y, ["NV", "PI", "EH"]
    except FileNotFoundError:
        warnings.warn("endometrial dataset not found (optional for testing)", stacklevel=2)
        return None, None, None


# In[6]:


# 模拟数据（验证 SAS 选项功能）
np.random.seed(42)
n = 50
X = np.random.normal(0, 1, (n, 2))
y = (X[:, 0] > 0).astype(int)
y[np.random.choice(n, 5)] = 1 - y[np.random.choice(n, 5)]

# 构造 SAS 风格 weight/freq/offset 数据
sas_weight = np.random.uniform(0.5, 2.0, n)  # 非负权重
sas_freq = np.random.randint(1, 5, n)        # 非负整数频数
sas_offset = np.random.normal(0, 0.5, n)     # 偏移项

# 1. 测试 SAS weight 选项
print("=== Test SAS-style weight option ===")
model_weight = FirthLogisticRegression(fit_intercept=True, weight=sas_weight)
model_weight.fit(X, y)
model_weight.summary(xname=["FeatureA", "FeatureB"])
print()

# 2. 测试 SAS freq 选项
print("=== Test SAS-style freq option ===")
model_freq = FirthLogisticRegression(fit_intercept=True, freq=sas_freq)
model_freq.fit(X, y)
model_freq.summary(xname=["FeatureA", "FeatureB"])
print()

# 3. 测试 SAS offset 选项
print("=== Test SAS-style offset option ===")
model_offset = FirthLogisticRegression(fit_intercept=True, offset=sas_offset)
model_offset.fit(X, y)
model_offset.summary(xname=["FeatureA", "FeatureB"])
print()

# 4. 对比 statsmodels Logit（验证接口一致性）
try:
    import statsmodels.api as sm
    X_sm = sm.add_constant(X)
    sm_model = sm.Logit(y, X_sm).fit(disp=0)
    print("=== statsmodels Logit Summary（接口对比）===")
    print(f"statsmodels params: {sm_model.params.round(4)}")
    print(f"本模型 params: {model_weight.params.round(4)}")
    print(f"statsmodels pvalues: {sm_model.pvalues.round(4)}")
    print(f"本模型 pvalues: {model_weight.pvalues.round(4)}")
except ImportError:
    print("提示：未安装 statsmodels，跳过接口对比")


# In[9]:


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
        print(f"Offset Used:          {self.offset is not None} (SAS style)")
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
        if self.wald:
            print("Note: Wald intervals are faster but less robust for small samples/separation.")

    def decision_function(self, X):
        """Align with SAS offset behavior: Linear predictor = X@coef + intercept + offset"""
        check_is_fitted(self)
        try:
            X = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to convert X to float64 array: {str(e)}")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got {X.ndim}D")
        # Include offset (consistent with training)
        return X @ self.coef_ + self.intercept_ + self.offset_

    def predict(self, X):
        decision = self.decision_function(X)
        indices = (decision > 0).astype(int)
        return self.classes_[indices]

    def predict_proba(self, X):
        decision = self.decision_function(X)
        proba_1 = expit(decision)
        return np.column_stack([1 - proba_1, proba_1])

# ------------------------------
# Core Helper Functions (adapted for weight/offset, retain Firth's method logic)
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
# Example Datasets and Test Code (validate SAS-style option functionality)
# ------------------------------
def load_sex2():
    try:
        with open_text("firthlogist.datasets", "sex2.csv") as f:
            X = np.loadtxt(f, skiprows=1, delimiter=",")
        y = X[:, 0]
        X = X[:, 1:]
        return X, y, ["age", "oc", "vic", "vicl", "vis", "dia"]
    except FileNotFoundError:
        warnings.warn("sex2 dataset not found (optional for testing)", stacklevel=2)
        return None, None, None

def load_endometrial():
    try:
        with open_text("firthlogist.datasets", "endometrial.csv") as f:
            X = np.loadtxt(f, skiprows=1, delimiter=",")
        y = X[:, -1]
        X = X[:, :-1]
        return X, y, ["NV", "PI", "EH"]
    except FileNotFoundError:
        warnings.warn("endometrial dataset not found (optional for testing)", stacklevel=2)
        return None, None, None


# In[10]:


# Simulate data (validate SAS-style option functionality)
np.random.seed(42)
n = 50
X = np.random.normal(0, 1, (n, 2))
y = (X[:, 0] > 0).astype(int)
y[np.random.choice(n, 5)] = 1 - y[np.random.choice(n, 5)]

# Construct SAS-style weight/freq/offset data
sas_weight = np.random.uniform(0.5, 2.0, n)  # Non-negative weights
sas_freq = np.random.randint(1, 5, n)        # Non-negative integer frequencies
sas_offset = np.random.normal(0, 0.5, n)     # Offset terms

# 1. Test SAS-style weight option
print("=== Test SAS-style weight option ===")
model_weight = FirthLogisticRegression(fit_intercept=True, weight=sas_weight)
model_weight.fit(X, y)
model_weight.summary(xname=["FeatureA", "FeatureB"])
print()

# 2. Test SAS-style freq option
print("=== Test SAS-style freq option ===")
model_freq = FirthLogisticRegression(fit_intercept=True, freq=sas_freq)
model_freq.fit(X, y)
model_freq.summary(xname=["FeatureA", "FeatureB"])
print()

# 3. Test SAS-style offset option
print("=== Test SAS-style offset option ===")
model_offset = FirthLogisticRegression(fit_intercept=True, offset=sas_offset)
model_offset.fit(X, y)
model_offset.summary(xname=["FeatureA", "FeatureB"])
print()

# 4. Compare with statsmodels Logit (validate interface consistency)
try:
    import statsmodels.api as sm
    X_sm = sm.add_constant(X)
    sm_model = sm.Logit(y, X_sm).fit(disp=0)
    print("=== statsmodels Logit Summary (for Interface Comparison) ===")
    print(f"statsmodels params: {sm_model.params.round(4)}")
    print(f"Current model params: {model_weight.params.round(4)}")
    print(f"statsmodels pvalues: {sm_model.pvalues.round(4)}")
    print(f"Current model pvalues: {model_weight.pvalues.round(4)}")
except ImportError:
    print("Note: statsmodels not installed, skipping interface comparison")


# In[ ]:





# In[ ]:





# In[ ]:




