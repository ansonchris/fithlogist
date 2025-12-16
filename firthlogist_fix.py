
def fit(self, X, y):
    """
    Fit Firth's bias-reduced logistic regression model to training data.
    Auto-saves feature names from X (pandas DataFrame/Series or numpy array).
    """
    # --------------------------
    # Auto-detect feature names (new code)
    # --------------------------
    import pandas as pd
    # Case 1: X is pandas DataFrame/Series → use column names
    if isinstance(X, (pd.DataFrame, pd.Series)):
        self.feature_names_ = X.columns.tolist()
    # Case 2: X is numpy array → default to x1, x2, ...
    elif isinstance(X, np.ndarray):
        self.feature_names_ = [f"x{i}" for i in range(1, X.shape[1] + 1)]
    # Case 3: Other input types (list, etc.) → default to x1, x2, ...
    else:
        X_np = np.asarray(X)
        self.feature_names_ = [f"x{i}" for i in range(1, X_np.shape[1] + 1)]

    # Original fit() code continues here...
    X, y = self._validate_input(X, y)
    if self.fit_const:
        X = np.hstack((X, np.ones((X.shape[0], 1))))
    # ... rest of the fit() code
    
    
def summary(self, xname=None):
    """
    Print model summary (no tabulate dependency) with CONST as FIRST row.
    Automatically uses feature names from input data (no manual xname required).

    Parameters
    ----------
    xname : list of str, optional
        Custom names for features (overrides auto-detected names). 
        Default: Auto-detected from X (pandas columns) or ["x1", "x2", ...].

    Returns
    -------
    None (prints summary to console)
    """
    check_is_fitted(self)  # Ensure model is fitted
    
    # --------------------------
    # Auto-detect feature names (no manual xname required)
    # --------------------------
    if xname is None:
        # Use pre-saved feature names from fitting
        xname = self.feature_names_
    else:
        # Validate custom xname if provided
        if len(xname) != len(self.coef_):
            raise ValueError(
                f"xname length ({len(xname)}) must match number of features ({len(self.coef_)})"
            )

    # --------------------------
    # CONST FIRST in summary
    # --------------------------
    # Step 1: Initialize lists with CONST (if fitted)
    var_names = []
    coef_vals = []
    if self.fit_const:
        var_names.append("const")  # CONST as first row
        coef_vals.append(self.const_)
    
    # Step 2: Add feature variables (after CONST)
    var_names.extend(xname)
    coef_vals.extend(self.coef_)

    # Prepare summary header (aligned columns)
    header = (
        f"{'Variable':<10} {'Coef':<10} {'Std Err':<10} "
        f"[{self.alpha/2:.3f}    {1-self.alpha/2:.3f}]    {'P-value':<10}"
    )
    separator = "-" * len(header)
    rows = []

    # Build each row of the summary table
    for i, var_name in enumerate(var_names):
        coef = coef_vals[i]
        se = self.bse_[i]
        ci_lower = self.ci_[i, 0]
        ci_upper = self.ci_[i, 1]
        pval = self.pvals_[i]

        # Format values (handle NaN for skipped CI/p-values)
        coef_str = f"{coef:.4f}" if not np.isnan(coef) else "NaN"
        se_str = f"{se:.4f}" if not np.isnan(se) else "NaN"
        ci_str = (
            f"{ci_lower:.4f}    {ci_upper:.4f}" 
            if not (np.isnan(ci_lower) or np.isnan(ci_upper)) 
            else "NaN    NaN"
        )
        pval_str = f"{pval:.4f}" if not np.isnan(pval) else "NaN"

        # Align columns (fixed width for readability)
        row = f"{var_name:<10} {coef_str:<10} {se_str:<10} {ci_str:<20} {pval_str:<10}"
        rows.append(row)

    # Print summary (CONST first + auto feature names)
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
    print(f"  Constant Term Included: {self.fit_const}")