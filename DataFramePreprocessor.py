class DataFramePreprocessor:
    def __init__(self, missing_values_strategy=None, normalization_method=None, 
                 outlier_method=None, feature_selection_method=None, 
                 feature_selection_params=None, parallel=False, batch_size=32):
        """
        Initializes the DataFramePreprocessor.

        Parameters:
        - missing_values_strategy: Strategy to handle missing values ('mean', 'median', 'mode', 'drop', None).
        - normalization_method: Method to normalize data ('zscore', 'minmax', 'maxabs', 'robust', None).
        - outlier_method: Method to handle outliers ('zscore', 'iqr', None).
        - feature_selection_method: Feature selection method ('variance', 'correlation', 'pca', 'lda', None).
        - feature_selection_params: Dictionary for feature selection parameters like 'variance_threshold' and 'correlation_threshold'.
        - parallel: Boolean indicating whether to use parallel processing.
        - batch_size: Number of data items to process in each batch.
        """
        self.missing_values_strategy = missing_values_strategy
        self.normalization_method = normalization_method.lower() if normalization_method else None
        self.outlier_method = outlier_method
        self.feature_selection_method = feature_selection_method
        self.feature_selection_params = feature_selection_params if feature_selection_params else {}
        self.parallel = parallel
        self.batch_size = batch_size

    def preprocess_batch(self, dfs):
        """
        Preprocesses a batch of DataFrames.

        Parameters:
        - dfs: A list of pandas DataFrames.

        Returns:
        - A list of preprocessed pandas DataFrames.
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

        processed_dfs = []

        for i in range(0, len(dfs), self.batch_size):
            batch = dfs[i:i + self.batch_size]
            if self.parallel:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor() as executor:
                    results = list(executor.map(self._preprocess_single_df, batch))
            else:
                results = [self._preprocess_single_df(df) for df in batch]

            processed_dfs.extend(results)

        return processed_dfs

    def _preprocess_single_df(self, df):
        """
        Preprocesses a single DataFrame.
        """
        # Handle missing values if specified
        if self.missing_values_strategy:
            df = self._handle_missing_values(df)

        # Handle outliers if specified
        if self.outlier_method:
            df = self._handle_outliers(df)

        # Normalize/scale data if specified
        if self.normalization_method:
            df = self._normalize(df)

        # Feature selection if specified
        if self.feature_selection_method:
            df = self._feature_selection(df)

        return df

    def _handle_missing_values(self, df):
        """
        Handles missing values in the DataFrame.
        """
        if self.missing_values_strategy == 'mean':
            return df.fillna(df.mean())
        elif self.missing_values_strategy == 'median':
            return df.fillna(df.median())
        elif self.missing_values_strategy == 'mode':
            return df.fillna(df.mode().iloc[0])
        elif self.missing_values_strategy == 'drop':
            return df.dropna()
        else:
            raise ValueError(f"Unsupported missing values strategy: {self.missing_values_strategy}")

    def _normalize(self, df):
        """
        Normalizes the DataFrame.
        """
        if self.normalization_method == 'zscore':
            scaler = StandardScaler()
        elif self.normalization_method == 'minmax':
            scaler = MinMaxScaler()
        elif self.normalization_method == 'maxabs':
            scaler = MaxAbsScaler()
        elif self.normalization_method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported normalization method: {self.normalization_method}")

        return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    def _handle_outliers(self, df):
        """
        Handles outliers in the DataFrame.
        """
        if self.outlier_method == 'zscore':
            from scipy import stats
            return df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
        elif self.outlier_method == 'iqr':
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
        elif self.outlier_method is None:
            return df
        else:
            raise ValueError(f"Unsupported outlier method: {self.outlier_method}")

    def _feature_selection(self, df):
        """
        Performs feature selection on the DataFrame.
        """
        if self.feature_selection_method == 'variance':
            from sklearn.feature_selection import VarianceThreshold
            threshold = self.feature_selection_params.get('variance_threshold', 0.01)
            selector = VarianceThreshold(threshold=threshold)
            return pd.DataFrame(selector.fit_transform(df), columns=df.columns[selector.get_support(indices=True)])
        elif self.feature_selection_method == 'correlation':
            threshold = self.feature_selection_params.get('correlation_threshold', 0.95)
            corr_matrix = df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            return df.drop(columns=to_drop)
        elif self.feature_selection_method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components='mle')
            return pd.DataFrame(pca.fit_transform(df))
        elif self.feature_selection_method == 'lda':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
            lda = LDA(n_components=None)
            y = df.iloc[:, -1]  # Assuming the last column is the target
            X = df.iloc[:, :-1]
            X_lda = lda.fit_transform(X, y)
            return pd.DataFrame(X_lda)
        else:
            raise ValueError(f"Unsupported feature selection method: {self.feature_selection_method}")

# Example usage:

base_dir = '...'
levels = ['status', 'neg_pos']  # This can be any list of levels you want to traverse

# Initialize the DataLoader for MATLAB files
data_loader = DataLoader(base_dir, levels, file_format='mat', mat_var_name='Pulse', batch_size=16)

# Initialize the DataFramePreprocessor with specific options
df_preprocessor = DataFramePreprocessor(
    missing_values_strategy='mean',  # Fill missing values with the mean
    normalization_method='zscore',   # Normalize using Z-score normalization
    outlier_method='iqr',            # Handle outliers using the IQR method
    feature_selection_method='correlation',  # Perform feature selection based on correlation
    feature_selection_params={'correlation_threshold': 0.90},  # Set the correlation threshold
    parallel=True,  # Enable parallel processing
    batch_size=16   # Match the DataLoader batch size for consistency
)

# Load, process, and preprocess the data in batches
for batch_data, batch_hierarchy, batch_filenames in data_loader.load_data():
    print(f"Loaded batch with files: {batch_filenames}")

    # Preprocess each DataFrame in the batch
    preprocessed_batch = df_preprocessor.preprocess_batch(batch_data)
    
    for data, preprocessed_data in zip(batch_data, preprocessed_batch):
        if isinstance(preprocessed_data, pd.DataFrame):
            print("Original DataFrame:")
            print(data.head())  # Display the first few rows of the original DataFrame
            print("Preprocessed DataFrame:")
            print(preprocessed_data.head())  # Display the first few rows of the preprocessed DataFrame
        else:
            print("Loaded data is not a DataFrame")
