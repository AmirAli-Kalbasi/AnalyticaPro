# AnalyticaPro
## Table of Contents

- [Status](#Status)
- [DataLoader](#DataLoader)
  - [Key Features](#Key_Features)
- [DataNormalizer](#DataNormalizer)
  - [Key Features](#Key_Features)
  - [Example](#Example)

## Status: Ongoing Development 🚧

This project is currently under development. Contributions, feedback, and suggestions are welcome!

## DataLoader: A Comprehensive and Flexible Multi-Format Data Loading Utility
The DataLoader class within AnalyticaPro is a robust and highly adaptable tool designed for efficiently loading and preprocessing datasets from various file formats, including CSV, Excel, MATLAB (.mat), JSON, HDF5, Parquet, Feather, Stata, SPSS, Pickle, and images. This utility is especially suited for handling complex, hierarchical directory structures, making it a valuable asset for data scientists, engineers, and researchers working with large and diverse datasets.

### Key_Features:
- Recursive Directory Traversal: Automatically navigates and processes data from nested directories based on user-defined levels, streamlining data loading from organized and hierarchical file systems.
  - Example Structure:
    ```plaintext
    /data_path/
        ├── G1/
        │   ├── PreTest/
        │   │   ├── R1/
        │   │   │   ├── file1.csv
        │   │   │   ├── file2.csv
        │   │   └── R2/
        │   └── PostTest/
        │       ├── R1/
    │       └── R2/
        ├── G2/
        .
        .
        .
        └── G3/
        .
        .
        .
    ```
    This structure can be represented by the list levels = ['groups', 'tests', 'rats'], where:

      - 'groups' corresponds to G1, G2, G3,
      - 'tests' corresponds to PreTest, PostTest, and
      - 'rats' corresponds to the individual records or datasets such as R1, R2.

- Multi-Format Support: Capable of handling a wide range of data formats, the DataLoader is designed to seamlessly integrate with various data types, from traditional tabular data to MATLAB files and complex image datasets.

- Selective Data Extraction: Provides the ability to select specific columns and rename them during the loading process for tabular data, enabling focused analysis and customized data pipelines.

- Framework-Specific Image Loading: Supports loading image data using popular machine learning frameworks such as OpenCV, TensorFlow, and PyTorch, ensuring compatibility with a variety of image processing and deep learning workflows.

- MATLAB File Handling: Extracts specific variables from .mat files, converting them into pandas DataFrames, which are easy to manipulate and integrate with other data in Python.

- Scalable and Efficient: Designed with scalability in mind, the DataLoader can handle large datasets efficiently, making it suitable for both small-scale and enterprise-level projects.

This utility is ideal for professionals who require a dependable and versatile tool to manage and preprocess data in multiple formats as part of a larger data analysis or machine learning pipeline. Whether you are working on exploratory data analysis, data preparation for modeling, or integrating various data sources, DataLoader offers the flexibility and functionality needed to simplify your workflow.


[View the code](https://github.com/AmirAli-Kalbasi/AnalyticaPro/blob/main/data_loader.py)

## DataNormalizer
The DataNormalizer class is a flexible and efficient Python utility designed to normalize data in batches, compatible with various data types including tabular data (e.g., CSV, Excel) and image data. It integrates seamlessly with the [DataLoader class](#DataLoader). This tool is particularly useful for machine learning and data preprocessing tasks, where data needs to be standardized before feeding it into models.

### Key_Features:
- Batch Processing: Normalize data in user-defined batch sizes, which is ideal for handling large datasets that cannot be loaded entirely into memory.
- Multiple Normalization Methods: Supports a variety of normalization techniques, including:
  - Z-score Normalization: Standardizes data to have a mean of 0 and a standard deviation of 1.
  - Min-Max Scaling: Scales data to a specified range, typically [0, 1].
  - Max-Abs Scaling: Scales data by its maximum absolute value, useful for preserving sparsity.
  - Robust Scaling: Uses median and interquartile range for scaling, making it robust to outliers.
- Supports Various Data Types: Compatible with both tabular data (pandas DataFrames) and image data (numpy arrays or tensors from frameworks like TensorFlow and PyTorch).
- Automatic Column Handling: Automatically handles column extraction and renaming for DataFrame inputs, based on user specifications.

### Example
```python
base_dir = '/content/drive/My Drive/small_English_dataset'
levels = ['test_train', 'personality']  # This can be any list of levels you want to traverse

# Initialize the DataLoader for CSV files
data_loader = DataLoader(base_dir, levels, file_format = 'image', framework = 'tensorflow')

normalizer = DataNormalizer(batch_size=32, normalization_method='minmax')

# Normalize data in batches
for normalized_data, hierarchy, filename in normalizer.normalize_batch(data_loader):
    print(f"Normalized {filename} with hierarchy: {hierarchy}")
    if isinstance(normalized_data, pd.DataFrame):
        print(normalized_data.head())
    else:
        print(normalized_data.shape)
```

[View the code](https://github.com/AmirAli-Kalbasi/AnalyticaPro/blob/main/DataNormalizer.py)
