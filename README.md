# AnalyticaPro

## DataLoader: A Comprehensive and Flexible Multi-Format Data Loading Utility
The DataLoader class within AnalyticaPro is a robust and highly adaptable tool designed for efficiently loading and preprocessing datasets from various file formats, including CSV, Excel, MATLAB (.mat), JSON, HDF5, Parquet, Feather, Stata, SPSS, Pickle, and images. This utility is especially suited for handling complex, hierarchical directory structures, making it a valuable asset for data scientists, engineers, and researchers working with large and diverse datasets.

### Key Features:
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
