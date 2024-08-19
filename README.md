# AnalyticaPro
## Table of Contents

- [Status](#Status)
- [DataLoader](#DataLoader)
  - [Key Features](#DataLoader_Key_Features)
- [DataNormalizer](#DataNormalizer)
  - [Key Features](#DataNormalizer_Key_Features)
  - [Example](#ImagePreprocessor_Example) 
- [ImagePreprocessor](#ImagePreprocessor)
  - [Key Features](#ImagePreprocessor_Key_Features)
  - [Example](#ImagePreprocessor_Example)
- [DataFramePreprocessor](#DataFramePreprocessor)
  - [Key Features](#DataFramePreprocessor_Key_Features)
  - [Example](#DataFramePreprocessor_Example)

## Status: Ongoing Development 🚧

This project is currently under development. Contributions, feedback, and suggestions are welcome!

## DataLoader: A Comprehensive and Flexible Multi-Format Data Loading Utility
The DataLoader class within AnalyticaPro is a robust and highly adaptable tool designed for efficiently loading and preprocessing datasets from various file formats, including CSV, Excel, MATLAB (.mat), JSON, HDF5, Parquet, Feather, Stata, SPSS, Pickle, and images. This utility is especially suited for handling complex, hierarchical directory structures, making it a valuable asset for data scientists, engineers, and researchers working with large and diverse datasets.

### Key Features
<a name="DataLoader_Key_Features"></a>
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

- Batch Loading: This feature efficiently loads data in batches, allowing for memory-efficient processing of large datasets. The batch size is customizable, making it ideal for workflow handling data in manageable chunks. 

This utility is ideal for professionals who require a dependable and versatile tool to manage and preprocess data in multiple formats as part of a larger data analysis or machine learning pipeline. Whether you are working on exploratory data analysis, data preparation for modeling, or integrating various data sources, DataLoader offers the flexibility and functionality needed to simplify your workflow.


[View the code](https://github.com/AmirAli-Kalbasi/AnalyticaPro/blob/main/data_loader.py)

## DataNormalizer
The DataNormalizer class is a flexible and efficient Python utility designed to normalize data in batches, compatible with various data types including tabular data (e.g., CSV, Excel) and image data. It integrates seamlessly with the [DataLoader class](#DataLoader). This tool is particularly useful for machine learning and data preprocessing tasks, where data needs to be standardized before feeding it into models.

### Key Features 
<a name="DataNormalizer_Key_Features"></a>
- Batch Processing: Normalize data in user-defined batch sizes, which is ideal for handling large datasets that cannot be loaded entirely into memory.
- Multiple Normalization Methods: Supports a variety of normalization techniques, including:
  - Z-score Normalization: Standardizes data to have a mean of 0 and a standard deviation of 1.
  - Min-Max Scaling: Scales data to a specified range, typically [0, 1].
  - Max-Abs Scaling: Scales data by its maximum absolute value, useful for preserving sparsity.
  - Robust Scaling: Uses median and interquartile range for scaling, making it robust to outliers.
- Supports Various Data Types: Compatible with both tabular data (pandas DataFrames) and image data (numpy arrays or tensors from frameworks like TensorFlow and PyTorch).
- Automatic Column Handling: Automatically handles column extraction and renaming for DataFrame inputs, based on user specifications.

### Example
<a name="DataNormalizer_Example"></a>
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

## ImagePreprocessor

The ImagePreprocessor class is a comprehensive and flexible tool designed for efficient and effective image preprocessing and augmentation, tailored for machine learning and deep learning workflows. This utility is built to handle various image preprocessing tasks, offering a wide range of functionalities, including:

### Key Features 
<a name="ImagePreprocessor_Key_Features"></a>
- Resizing and Cropping: Easily resize images to a target size or crop specific regions for focused analysis.
- Padding and Color Space Conversion: Add padding to images and convert between different color spaces (RGB, Grayscale, HSV).
- Clipping and Binarization: Clip pixel values to a specified range and binarize images using a threshold.
- Noise Addition and Blurring: Add Gaussian or salt-and-pepper noise and apply Gaussian or median blurring for image smoothing.
- Normalization: Supports multiple normalization methods, including z-score, min-max, max-abs, and robust normalization to suit various machine learning models.
- Data Augmentation: Integrates seamlessly with TensorFlow's ImageDataGenerator to perform complex data augmentation, including rotations, shifts, shears, zooms, flips, and brightness adjustments.
- Parallel and Batch Processing: Efficiently preprocess large datasets by leveraging parallel processing and batch handling, ensuring scalability for large-scale projects.
- Compatibility: Works with both TensorFlow and PyTorch, supporting images as tensors and enabling smooth integration into deep learning pipelines.

This class is ideal for data scientists, machine learning engineers, and researchers who require a robust and versatile image preprocessing toolkit. The ImagePreprocessor class is designed to streamline the preparation of image datasets, making it easier to focus on model development and experimentation.

### Example
<a name="ImagePreprocessor_Example"></a>
```python
base_dir = '...'
levels = ['test_train', 'personality']  # This can be any list of levels you want to traverse

# image:
data_loader = DataLoader(base_dir, levels, file_format='image', framework='pytorch', batch_size=64)

# Define the augmentation options (if needed)
augmentations = {
    'rotation': 20,
    'width_shift': 0.2,
    'height_shift': 0.2,
    'shear': 0.2,
    'zoom': 0.2,
    'horizontal_flip': True,
    'vertical_flip': False,
    'brightness': [0.8, 1.2]
}


# Initialize the ImagePreprocessor
image_preprocessor = ImagePreprocessor(
    target_size=(224, 224),
    normalization_method='minmax',
    augment=augmentations,
    parallel=True,
    color_space='rgb',  # Convert images to RGB color space
    clip_range=(0, 255),
    noise_type='gaussian',
    blur_type='gaussian'
)

# Process the data in batches
for batch_data, batch_hierarchy, batch_filenames in data_loader.load_data():
    # Preprocess the images in the batch
    preprocessed_images = image_preprocessor.preprocess_batch(batch_data)
    
    # work with the preprocessed_images
    for i, image in enumerate(preprocessed_images):
        print(f"Processed image from file: {batch_filenames[i]}")
        print(f"Processed image shape: {image.shape}")

```

[View the code](https://github.com/AmirAli-Kalbasi/AnalyticaPro/blob/main/ImagePreprocessor.py)



### DataFramePreprocessor

The `DataFramePreprocessor` class is a versatile and efficient tool designed to streamline the preprocessing of pandas DataFrames. It offers a variety of preprocessing techniques essential for preparing data for analysis or machine learning models. The class is built to handle large datasets efficiently, with support for batch processing and optional parallel execution.

#### Key Features:
<a name="DataFramePreprocessor_Key_Features"></a>

1. **Handling Missing Values**:
   - Provides multiple strategies for dealing with missing data, including filling with mean, median, mode, or dropping missing values.

2. **Data Normalization/Scaling**:
   - Supports several normalization methods such as Z-score, Min-Max scaling, Max-Abs scaling, and Robust scaling, allowing for standardized data preparation.

3. **Outlier Detection and Handling**:
   - Offers robust methods to detect and remove outliers using Z-score or Interquartile Range (IQR), ensuring that extreme values do not skew the analysis.

4. **Feature Selection**:
   - Includes feature selection techniques like Variance Thresholding, Correlation Thresholding, Principal Component Analysis (PCA), and Linear Discriminant Analysis (LDA). Parameters for these methods can be easily customized through a dictionary interface.

5. **Batch Processing**:
   - Designed to process data in batches, making it suitable for large datasets. The batch size is configurable, ensuring that the class can handle data efficiently without overwhelming system memory.

6. **Parallel Processing**:
   - Capable of parallel execution to take full advantage of multi-core processors, significantly reducing preprocessing time for large datasets.

#### Example:
<a name="DataFramePreprocessor_Example"></a>

```python
# Initialize the DataFramePreprocessor with desired settings
df_preprocessor = DataFramePreprocessor(
    missing_values_strategy='mean',  # Fill missing values with the mean
    normalization_method='zscore',   # Normalize using Z-score normalization
    outlier_method='iqr',            # Handle outliers using the IQR method
    feature_selection_method='correlation',  # Perform feature selection based on correlation
    feature_selection_params={'correlation_threshold': 0.90},  # Set the correlation threshold
    parallel=True,  # Enable parallel processing
    batch_size=16   # Set batch size for processing
)

# Preprocess a list of DataFrames
preprocessed_dfs = df_preprocessor.preprocess_batch(dfs)
```

#### Integration Example:

This class can be seamlessly integrated with a `DataLoader` for end-to-end data processing:

```python
# Example DataLoader and DataFramePreprocessor integration

# Load and preprocess data in batches
for batch_data, batch_hierarchy, batch_filenames in data_loader.load_data():
    preprocessed_batch = df_preprocessor.preprocess_batch(batch_data)
    # Use preprocessed data for analysis or model training
```

The `DataFramePreprocessor` is designed to be flexible, allowing you to easily customize the preprocessing pipeline to suit the specific needs of your project. Whether you're handling missing data, normalizing features, detecting outliers, or selecting the most important features, this class provides a robust foundation for efficient data preprocessing.

[View the code](https://github.com/AmirAli-Kalbasi/AnalyticaPro/blob/main/DataFramePreprocessor.py)
