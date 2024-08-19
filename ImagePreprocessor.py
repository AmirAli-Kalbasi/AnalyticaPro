class ImagePreprocessor:
    def __init__(self, target_size=(224, 224), normalization_method='zscore', augment=None, 
                 parallel=False, batch_size=32, crop=None, padding=None, color_space=None, 
                 clip_range=None, binarize_threshold=None, noise_type=None, blur_type=None):
        """
        Initializes the ImagePreprocessor.
        """
        self.target_size = target_size
        self.normalization_method = normalization_method.lower()
        self.augment = augment if augment else {}
        self.parallel = parallel
        self.batch_size = batch_size
        self.crop = crop
        self.padding = padding
        self.color_space = color_space
        self.clip_range = clip_range
        self.binarize_threshold = binarize_threshold
        self.noise_type = noise_type
        self.blur_type = blur_type
        self.datagen = self._create_augmenter() if self.augment else None

    def _create_augmenter(self):
        """
        Creates the ImageDataGenerator for data augmentation based on the provided augment dictionary.
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        return ImageDataGenerator(
            rotation_range=self.augment.get('rotation', 0),
            width_shift_range=self.augment.get('width_shift', 0),
            height_shift_range=self.augment.get('height_shift', 0),
            shear_range=self.augment.get('shear', 0),
            zoom_range=self.augment.get('zoom', 0),
            horizontal_flip=self.augment.get('horizontal_flip', False),
            vertical_flip=self.augment.get('vertical_flip', False),
            brightness_range=self.augment.get('brightness', None),
            fill_mode='nearest'
        )

    def preprocess_batch(self, images):
        """
        Preprocesses a batch of images in chunks of the specified batch size.
        """
        processed_images = []

        # Process images in chunks based on the batch size
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            if self.parallel:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor() as executor:
                    results = list(executor.map(self._preprocess_single_image, batch))
            else:
                results = [self._preprocess_single_image(image) for image in batch]

            processed_images.extend(results)

        return processed_images

    def _preprocess_single_image(self, image):
        """
        Preprocesses a single image.
        """
        input_type = type(image)

        # Convert image to numpy array if it is a tensor
        if isinstance(image, tf.Tensor):
            image = image.numpy()
        elif isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            # PyTorch tensors are typically in (C, H, W) format; convert to (H, W, C)
            image = np.transpose(image, (1, 2, 0))

        # Apply cropping if specified
        if self.crop:
            x, y, w, h = self.crop
            image = image[y:y+h, x:x+w]

        # Apply padding if specified
        if self.padding:
            top, bottom, left, right = self.padding
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Convert color space if specified
        if self.color_space:
            if self.color_space == 'gray':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif self.color_space == 'hsv':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Resize image
        image = cv2.resize(image, self.target_size)

        # Clip pixel values if specified
        if self.clip_range:
            min_val, max_val = self.clip_range
            image = np.clip(image, min_val, max_val)

        # Binarize the image if specified
        if self.binarize_threshold is not None:
            _, image = cv2.threshold(image, self.binarize_threshold, 255, cv2.THRESH_BINARY)

        # Add noise if specified
        if self.noise_type:
            image = self._add_noise(image, self.noise_type)

        # Apply blurring/filtering if specified
        if self.blur_type:
            image = self._apply_blur(image, self.blur_type)

        # Normalize image
        image = self._normalize(image)

        # Apply data augmentation if enabled
        if self.datagen:
            image = self._augment(image)

        # Convert back to the original input type
        if input_type == tf.Tensor:
            image = tf.convert_to_tensor(image)
        elif input_type == torch.Tensor:
            image = torch.tensor(image)

        return image

    def _normalize(self, image):
        """
        Normalizes the image.
        """
        if self.normalization_method == 'zscore':
            return (image - np.mean(image)) / np.std(image)
        elif self.normalization_method == 'minmax':
            return image / 255.0
        elif self.normalization_method == 'maxabs':
            return image / np.max(np.abs(image))
        elif self.normalization_method == 'robust':
            return (image - np.median(image)) / (np.percentile(image, 75) - np.percentile(image, 25))
        else:
            raise ValueError(f"Unsupported normalization method: {self.normalization_method}")

    def _augment(self, image):
        """
        Applies data augmentation to the image.
        """
        image = np.expand_dims(image, 0)  # Expand dimensions to fit the generator's input format
        it = self.datagen.flow(image, batch_size=1)
        augmented_image = next(it)[0].astype('float32')  # Use Python's built-in next()
        return augmented_image

    def _add_noise(self, image, noise_type):
        """
        Adds noise to the image.
        """
        if noise_type == 'gaussian':
            mean = 0
            var = 0.01
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, image.shape)
            noisy_image = image + gauss
            return np.clip(noisy_image, 0, 255)
        elif noise_type == 'salt_pepper':
            s_vs_p = 0.5
            amount = 0.04
            noisy_image = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            noisy_image[coords] = 1
            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            noisy_image[coords] = 0
            return noisy_image

    def _apply_blur(self, image, blur_type):
        """
        Applies blurring to the image.
        """
        if blur_type == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif blur_type == 'median':
            return cv2.medianBlur(image, 5)
        else:
            raise ValueError(f"Unsupported blur type: {blur_type}")

