# LitePolis Math Default Module

This module provides mathematical algorithms, specifically PCA and KMeans, for the LitePolis platform based on [LitePolis-math](https://github.com/NewJerseyStyle/LitePolis-math). It is designed to work with user vote data stored in a database (primarily StarRocks, with potential support for PostgreSQL via the [litepolis_database_default](https://github.com/NewJerseyStyle/LitePolis-database-default) module) to produce outputs for API endpoints.

## Installation

This module will be available on PyPI. You can install it using pip:

```bash
pip install litepolis-math-default
```

## Configuration

This module relies on the database connection configured for the `litepolis_database_default` module. Ensure that the `litepolis_database_default` section within your `~/litepolis/litepolis.config` file is correctly set up to connect to your database (StarRocks or PostgreSQL).

Example `~/litepolis/litepolis.config` snippet:

```ini
[litepolis_database_default]
database_url = starrocks://user:password@host:port/database
# Or for PostgreSQL:
# database_url = postgresql://user:password@host:port/database
sqlalchemy_engine_pool_size = 10
sqlalchemy_pool_max_overflow = 20
```

The module uses the `litepolis_database_default.utils.connect_db()` function to obtain the database engine based on this configuration.

## Usage

The primary use case for this module is to perform PCA on a user-vote matrix derived from your database. The results can then be used for further analysis or as input for other algorithms like KMeans clustering.

Here's a quick example demonstrating how to use the core components:

```python
import logging
from litepolis_math_default import fetch_r_matrix
from litepolis_math_default.algorithms import PCA
# from litepolis_math_default.algorithms import KMeans # Uncomment if needed

# Configure logging (optional)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # 1. Fetch and build the user-vote matrix (R matrix)
    r_matrix = fetch_r_matrix(engine)
    logging.info(f"Fetched R matrix with shape: {r_matrix.shape}")

    # Optional: Validate the matrix
    # from litepolis_math_default.validation import validate_matrix
    # validate_matrix(r_matrix)
    # logging.info("R matrix validated.")

    # 2. Apply PCA
    # The PCA algorithm expects a NumPy array
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(r_matrix.values)
    logging.info(f"PCA applied. Result shape: {pca_result.shape}")

    # 3. (Optional) Apply KMeans clustering on PCA results
    # kmeans = KMeans(n_clusters=3)
    # user_clusters = kmeans.fit_predict(pca_result)
    # logging.info("KMeans clustering applied.")
    # print("User clusters:", user_clusters)

    # 'pca_result' contains the PCA output (user coordinates in the reduced dimension space)
    # You can now use 'pca_result' in your API endpoint response or for further processing.

except Exception as e:
    logging.error(f"An error occurred: {e}")

```

## Incremental PCA Updates

The `PCA` class now supports incremental updates, allowing you to update the principal components with new data without refitting the model on the entire dataset. This can be useful when your data (like the R matrix) changes over time and you want to update the PCA model efficiently.

To use the incremental update feature:

1. Initialize and fit the `PCA` model with an initial batch of data using `fit_transform`.
2. When new data becomes available, use the `update` method with the new data batch.

```python
import numpy as np
from litepolis_math_default.algorithms import PCA

# Assume pca is already initialized and fitted with initial data
# pca = PCA(n_components=2)
# initial_data = np.random.rand(100, 10) # Example initial data
# pca.fit_transform(initial_data)

# When new data arrives
new_data_batch = np.random.rand(10, 10) # Example new data batch

# Update the PCA model with the new data
pca.update(new_data_batch)

# You can now transform new data using the updated components
transformed_new_data = pca.transform(new_data_batch)
print("Transformed new data shape:", transformed_new_data.shape)

# You can also transform the original data or any other data using the updated components
# transformed_initial_data = pca.transform(initial_data)
# print("Transformed initial data shape:", transformed_initial_data.shape)
```

This incremental update process can help maintain a relevant PCA model as your data evolves without the computational cost of refitting on the entire cumulative dataset.

## Extending the Module

The module is structured to allow for easy extension. Additional algorithms can be added to the `litepolis_math_default/algorithms/` directory.

## Development

For development and running tests, you will need `pytest`, `pandas`, `numpy`, `sqlalchemy`, and potentially `scikit-learn` (for comparison/validation of custom algorithms).

```bash
pip install pytest pandas numpy sqlalchemy scikit-learn
```
