import numpy as np

def validate_matrix(r_matrix):
    """Check for NaN/Inf and ensure numeric dtype."""
    if r_matrix.isnull().values.any():
        raise ValueError("R matrix contains NaN values.")
    if not np.issubdtype(r_matrix.values.dtype, np.number):
        raise ValueError("Non-numeric values in R matrix.")