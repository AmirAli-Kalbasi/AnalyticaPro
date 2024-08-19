import os
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, base_dir, levels, columns=None, column_names=None, file_format='csv', mat_var_name='data', framework='opencv', batch_size=32):
        """
        Initializes the DataLoader.

        Parameters:
        - base_dir: The base directory where the data is stored.
        - levels: A list of strings representing the directory levels (e.g., ['groups', 'tests', 'rats']).
        - columns: A list of column indices to extract (e.g., [2, 3, 4]). If None, all columns are used.
        - column_names: A list of names for the extracted columns (e.g., ['NAc-shell', 'HPC-CA1', 'CPP Position']). If None, original names are retained.
        - file_format: The format of the files to be loaded ('csv', 'excel', 'mat', 'json', 'hdf5', 'parquet', 'feather', 'stata', 'spss', 'pickle', 'image'). Default is 'csv'.
        - mat_var_name: The name of the variable to extract from MATLAB files. Required if file_format is 'mat'. Default is 'data'.
        - framework: The framework to use for image loading ('opencv', 'tensorflow', 'pytorch'). Default is 'opencv'.
        - batch_size: The number of data items to yield in each batch. Default is 32.
        """
        self.base_dir = base_dir
        self.levels = levels
        self.columns = columns
        self.column_names = column_names
        self.file_format = file_format.lower()
        self.mat_var_name = mat_var_name
        self.framework = framework.lower()
        self.batch_size = batch_size

        if self.file_format == 'mat' and not self.mat_var_name:
            raise ValueError("mat_var_name must be specified when file_format is 'mat'.")

    def load_data(self):
        """
        Loads the data by recursively traversing the directory structure according to the levels specified.

        Yields:
        - A batch of data (list of pandas DataFrames or image data).
        - A list of dictionaries with the hierarchy information for each data file.
        - A list of filenames for the loaded files.
        """
        batch_data, batch_hierarchy, batch_filenames = [], [], []
        for data, hierarchy, filename in self._recursive_load(self.base_dir, {}):
            batch_data.append(data)
            batch_hierarchy.append(hierarchy)
            batch_filenames.append(filename)
            if len(batch_data) == self.batch_size:
                yield batch_data, batch_hierarchy, batch_filenames
                batch_data, batch_hierarchy, batch_filenames = [], [], []
        if batch_data:  # Yield remaining data if any
            yield batch_data, batch_hierarchy, batch_filenames

    def _recursive_load(self, current_dir, hierarchy):
        """
        Recursively traverses the directory structure.

        Parameters:
        - current_dir: The current directory being traversed.
        - hierarchy: A dictionary storing the hierarchy of directories traversed so far.

        Yields:
        - Data (pandas DataFrame or image data).
        - A dictionary with the hierarchy information for each data file.
        - The filename of the loaded file.
        """
        if len(hierarchy) == len(self.levels):
            for file in os.listdir(current_dir):
                if self._is_valid_file(file):
                    file_path = os.path.join(current_dir, file)
                    data = self._load_file(file_path)

                    # If data is a DataFrame, handle column extraction and renaming
                    if isinstance(data, pd.DataFrame):
                        if self.columns:
                            data = data.iloc[:, self.columns]
                        if self.column_names:
                            data.columns = self.column_names
                        elif self.columns:
                            data.columns = data.columns[self.columns]

                    yield data, hierarchy, file
        else:
            level = self.levels[len(hierarchy)]
            for folder in os.listdir(current_dir):
                next_dir = os.path.join(current_dir, folder)
                if os.path.isdir(next_dir):
                    new_hierarchy = hierarchy.copy()
                    new_hierarchy[level] = folder
                    yield from self._recursive_load(next_dir, new_hierarchy)

    def _is_valid_file(self, file):
        """
        Checks if the file has the correct extension based on the file_format.

        Parameters:
        - file: The file name.

        Returns:
        - bool: True if the file extension matches the expected format, False otherwise.
        """
        if self.file_format == 'csv':
            return file.endswith('.csv')
        elif self.file_format == 'excel':
            return file.endswith(('.xls', '.xlsx'))
        elif self.file_format == 'mat':
            return file.endswith('.mat')
        elif self.file_format == 'json':
            return file.endswith('.json')
        elif self.file_format == 'hdf5':
            return file.endswith(('.h5', '.hdf5'))
        elif self.file_format == 'parquet':
            return file.endswith('.parquet')
        elif self.file_format == 'feather':
            return file.endswith('.feather')
        elif self.file_format == 'stata':
            return file.endswith('.dta')
        elif self.file_format == 'spss':
            return file.endswith('.sav')
        elif self.file_format == 'pickle':
            return file.endswith('.pkl')
        elif self.file_format == 'image':
            return file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        return False

    def _load_file(self, file_path):
        """
        Loads the file based on the specified file format.

        Parameters:
        - file_path: The full path to the file.

        Returns:
        - Loaded data (pandas DataFrame or image data).
        """
        if self.file_format == 'csv':
            return pd.read_csv(file_path)
        elif self.file_format == 'excel':
            return pd.read_excel(file_path)
        elif self.file_format == 'mat':
            return self._load_mat_file(file_path)
        elif self.file_format == 'json':
            return pd.read_json(file_path)
        elif self.file_format == 'hdf5':
            return pd.read_hdf(file_path)
        elif self.file_format == 'parquet':
            return pd.read_parquet(file_path)
        elif self.file_format == 'feather':
            return pd.read_feather(file_path)
        elif self.file_format == 'stata':
            return pd.read_stata(file_path)
        elif self.file_format == 'spss':
            return pd.read_spss(file_path)
        elif self.file_format == 'pickle':
            return pd.read_pickle(file_path)
        elif self.file_format == 'image':
            return self._load_image(file_path)
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")

    def _load_mat_file(self, file_path):
        """
        Loads MATLAB file and extracts the specified variable.

        Parameters:
        - file_path: The full path to the .mat file.

        Returns:
        - pd.DataFrame: The extracted data as a pandas DataFrame.
        """
        import scipy.io as sio  # Import scipy only when needed
        mat_contents = sio.loadmat(file_path)
        if self.mat_var_name in mat_contents:
            data = mat_contents[self.mat_var_name]
            if isinstance(data, np.ndarray):
                return pd.DataFrame(data)
            else:
                raise ValueError(f"Variable '{self.mat_var_name}' is not an ndarray.")
        else:
            raise ValueError(f"Variable '{self.mat_var_name}' not found in MATLAB file.")

    def _load_image(self, file_path):
        """
        Loads an image using the specified framework.

        Parameters:
        - file_path: The full path to the image file.

        Returns:
        - Image data (as a numpy array or framework-specific tensor).
        """
        if self.framework == 'opencv':
            import cv2  # Import OpenCV only when needed
            return cv2.imread(file_path)  # OpenCV loads images as BGR
        elif self.framework == 'tensorflow':
            import tensorflow as tf  # Import TensorFlow only when needed
            image = tf.io.read_file(file_path)
            return tf.image.decode_image(image)
        elif self.framework == 'pytorch':
            from PIL import Image  # Import Pillow only when needed
            import torch  # Import PyTorch only when needed
            from torchvision import transforms  # Import transforms only when needed
            image = Image.open(file_path)
            transform = transforms.ToTensor()
            return transform(image)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

# Example usage:
base_dir = 'small_English_dataset'
levels = ['test_train', 'personality']  # This can be any list of levels you want to traverse

# image:
data_loader = DataLoader(base_dir, levels, file_format='image', framework='tensorflow', batch_size=16)

# Load and process the data in batches
for batch_data, batch_hierarchy, batch_filenames in data_loader.load_data():
    print(f"Loaded batch with files: {batch_filenames}")
    for data in batch_data:
        if isinstance(data, pd.DataFrame):
            print(data.head())  # Display the first few rows of the DataFrame
        else:
            print(type(data))  # Display the type of the loaded image data

# Example usage:
'''
base_dir = 'small_English_dataset'
levels = ['test_train', 'personality']  # This can be any list of levels you want to traverse

# image:
data_loader = DataLoader(base_dir, levels, file_format='image', framework='tensorflow', batch_size=16)

# MATLAB:
data_loader = DataLoader(base_dir, levels, file_format = 'mat', mat_var_name = 'Pulse')

# CSV:
base_dir = '/content/drive/My Drive/Neuroscience Project'
levels = ['groups', 'tests', 'rats']  # This can be any list of levels you want to traverse

# Specify columns to extract and their names
columns = [2, 3, 4]
column_names = ['NAc-shell', 'HPC-CA1', 'CPP Position']

# Initialize the DataLoader for CSV files 
data_loader = DataLoader(base_dir, levels, columns, column_names)
'''

# Load and process the data
for data, hierarchy, filename in data_loader.load_data():
    print(f"Loaded {filename} with hierarchy: {hierarchy}")
    if isinstance(data, pd.DataFrame):
        print(data.head())  # Display the first few rows of the DataFrame
    else:
        print(type(data))  # Display the type of the loaded image data
