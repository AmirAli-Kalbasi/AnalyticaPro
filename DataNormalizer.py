import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

class DataNormalizer:
    def __init__(self, batch_size=32, normalization_method='zscore'):
        """
        Initializes the DataNormalizer.

        Parameters:
        - batch_size: The size of each batch for normalization. Default is 32.
        - normalization_method: The method to use for normalization ('zscore', 'minmax', 'maxabs', 'robust'). Default is 'zscore'.
        """
        self.batch_size = batch_size
        self.normalization_method = normalization_method.lower()
        self.scaler = self._get_scaler()

    def _get_scaler(self):
        """
        Returns the appropriate scaler object based on the normalization method.
        """
        if self.normalization_method == 'zscore':
            return StandardScaler()
        elif self.normalization_method == 'minmax':
            return MinMaxScaler()
        elif self.normalization_method == 'maxabs':
            return MaxAbsScaler()
        elif self.normalization_method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unsupported normalization method: {self.normalization_method}")

    def normalize_batch(self, data_loader):
        """
        Normalizes the data from the DataLoader in batches.

        Parameters:
        - data_loader: An instance of DataLoader that yields data, hierarchy, and filename.

        Yields:
        - Normalized data (same type as input: pandas DataFrame, numpy array, or tensor).
        - A dictionary with the hierarchy information for each data file.
        - The filename of the loaded file.
        """
        batch_data = []
        hierarchy_list = []
        filenames = []

        for data, hierarchy, filename in data_loader.load_data():
            # Append data and hierarchy to batch
            batch_data.append(data)
            hierarchy_list.append(hierarchy)
            filenames.append(filename)

            # Check if batch is full
            if len(batch_data) >= self.batch_size:
                yield from self._process_batch(batch_data, hierarchy_list, filenames)
                batch_data = []  # Reset batch
                hierarchy_list = []
                filenames = []

        # Process any remaining data in the final batch
        if batch_data:
            yield from self._process_batch(batch_data, hierarchy_list, filenames)

    def _process_batch(self, batch_data, hierarchy_list, filenames):
        """
        Processes a batch of data, normalizing it.

        Parameters:
        - batch_data: A list of data to normalize.
        - hierarchy_list: A list of hierarchy dictionaries corresponding to the data.
        - filenames: A list of filenames corresponding to the data.

        Yields:
        - Normalized data (same type as input: pandas DataFrame, numpy array, or tensor).
        - A dictionary with the hierarchy information for each data file.
        - The filename of the loaded file.
        """
        for i, data in enumerate(batch_data):
            input_type = type(data)
            if isinstance(data, pd.DataFrame):
                # Normalize DataFrame using the specified scaler
                normalized_data = pd.DataFrame(
                    self.scaler.fit_transform(data),
                    columns=data.columns,
                    index=data.index
                )
            elif isinstance(data, np.ndarray):
                # Normalize image data (numpy array)
                normalized_data = self._normalize_array(data)
            elif hasattr(data, 'numpy'):  # Check if it's a tensor (TensorFlow or PyTorch)
                # Convert tensor to numpy array, normalize, and then convert back to the original tensor type
                normalized_data = self._normalize_array(data.numpy())
                normalized_data = self._convert_back_to_tensor(normalized_data, input_type)
            else:
                raise ValueError(f"Unsupported data type: {input_type}")

            yield normalized_data, hierarchy_list[i], filenames[i]

    def _normalize_array(self, array):
        """
        Normalizes a numpy array (typically image data).

        Parameters:
        - array: The numpy array to normalize.

        Returns:
        - The normalized numpy array.
        """
        # Reshape array to 2D (samples, features) if necessary
        original_shape = array.shape
        if len(original_shape) > 2:
            array = array.reshape((array.shape[0], -1))

        # Normalize using the specified scaler
        array = self.scaler.fit_transform(array)

        # Reshape back to original shape if necessary
        if len(original_shape) > 2:
            array = array.reshape(original_shape)

        return array

    def _convert_back_to_tensor(self, array, original_type):
        """
        Converts a numpy array back to the original tensor type (TensorFlow or PyTorch).

        Parameters:
        - array: The normalized numpy array.
        - original_type: The original type of the input data (TensorFlow or PyTorch tensor).

        Returns:
        - The normalized data as the original tensor type.
        """
        if 'tensorflow' in original_type.__module__:
            import tensorflow as tf
            return tf.convert_to_tensor(array)
        elif 'torch' in original_type.__module__:
            import torch
            return torch.tensor(array)
        else:
            raise ValueError(f"Unsupported tensor type: {original_type}")

    def check_output_type(self, normalized_data):
        """
        Checks and returns the data type of the normalized output.

        Parameters:
        - normalized_data: The normalized data.

        Returns:
        - The type of the normalized data.
        """
        return type(normalized_data)

# Example

normalizer = DataNormalizer(batch_size=32, normalization_method='minmax')

# Normalize data in batches
for normalized_data, hierarchy, filename in normalizer.normalize_batch(data_loader):
    print(f"Normalized {filename} with hierarchy: {hierarchy}")
    if isinstance(normalized_data, pd.DataFrame):
        print(normalized_data.head())
    else:
        print(normalized_data.shape)
