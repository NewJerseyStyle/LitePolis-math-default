import numpy as np

class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.n_samples = 0
        self.cov_sum_sq = None # Sum of squares and cross-products for covariance calculation

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.n_samples = X.shape[0]
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        self.cov_sum_sq = np.dot(X_centered.T, X_centered)

        # Compute covariance matrix
        # Use n_samples - 1 for sample covariance, or n_samples for population covariance
        # np.cov uses n-1 by default when rowvar=False
        cov_matrix = self.cov_sum_sq / (self.n_samples - 1) if self.n_samples > 1 else np.zeros_like(self.cov_sum_sq)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # Sort eigenvectors by eigenvalues (descending)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, sorted_idx[:self.n_components]]

        # Project data
        return np.dot(X_centered, self.components)

    def update(self, X_new: np.ndarray):
        """
        Incrementally updates the PCA model with new data.
        This method updates the mean and the sum of squares/cross-products,
        then recomputes the principal components.
        """
        if self.mean is None:
            # If the model hasn't been fitted yet, just fit on the new data
            self.fit_transform(X_new)
            return

        n_new_samples = X_new.shape[0]
        total_samples = self.n_samples + n_new_samples

        # Calculate mean and covariance sum of squares for the new batch
        mean_new_batch = np.mean(X_new, axis=0)
        X_new_centered_batch = X_new - mean_new_batch
        cov_sum_sq_new_batch = np.dot(X_new_centered_batch.T, X_new_centered_batch)

        # Update sum of squares and cross-products for the combined data
        # Formula for combining sums of squares relative to the combined mean:
        # S_combined = S_old + S_new + n_old * n_new / (n_old + n_new) * (mean_old - mean_new_batch) * (mean_old - mean_new_batch).T
        self.cov_sum_sq = self.cov_sum_sq + cov_sum_sq_new_batch + \
                          (self.n_samples * n_new_samples / total_samples) * \
                          np.outer((self.mean - mean_new_batch), (self.mean - mean_new_batch))

        # Update mean
        self.mean = (self.n_samples * self.mean + n_new_samples * mean_new_batch) / total_samples
        self.n_samples = total_samples

        # Recompute covariance matrix
        cov_matrix = self.cov_sum_sq / (self.n_samples - 1) if self.n_samples > 1 else np.zeros_like(self.cov_sum_sq)

        # Recompute Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # Sort eigenvectors by eigenvalues (descending)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, sorted_idx[:self.n_components]]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Projects new data onto the existing principal components.
        """
        if self.mean is None or self.components is None:
            raise RuntimeError("PCA model has not been fitted yet.")
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)