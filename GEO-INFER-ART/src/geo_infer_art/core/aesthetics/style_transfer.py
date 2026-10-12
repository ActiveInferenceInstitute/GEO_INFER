"""
Style transfer module for applying artistic styles to geospatial visualizations.
"""

import logging
import os
from importlib.resources import files
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Conditional import of TensorFlow to allow use of the package without it
try:
    import tensorflow as tf
    from tensorflow.keras.applications import vgg19

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class StyleTransfer:
    """
    A class for applying artistic style transfer to geospatial visualizations.

    This class uses neural style transfer techniques to apply artistic styles to maps
    and geospatial data visualizations.

    Attributes:
        style_image: The style image to draw aesthetics from
        content_image: The content image (map) to apply the style to
        model: The style transfer model
    """

    # Predefined artistic styles
    PREDEFINED_STYLES = {
        "watercolor": "watercolor_landscape.jpg",
        "oil_painting": "oil_painting_landscape.jpg",
        "sketch": "pencil_sketch.jpg",
        "abstract": "abstract_geometric.jpg",
        "impressionist": "impressionist_landscape.jpg",
        "ukiyo_e": "ukiyo_e_wave.jpg",
        "van_gogh": "van_gogh_starry_night.jpg",
        "monet": "monet_water_lilies.jpg",
        "picasso": "picasso_cubist.jpg",
        "dali": "dali_surrealist.jpg",
        "kandinsky": "kandinsky_abstract.jpg",
        "pollock": "pollock_drip_painting.jpg",
        "hokusai": "hokusai_wave.jpg",
        "vermeer": "vermeer_interior.jpg",
        "cezanne": "cezanne_still_life.jpg",
        "matisse": "matisse_cutouts.jpg",
    }

    def __init__(
        self,
        style_image: str | np.ndarray | Image.Image | None = None,
        content_image: str | np.ndarray | Image.Image | None = None,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        """
        Initialize a StyleTransfer object.

        Args:
            style_image: Path to style image or image array/object
            content_image: Path to content image or image array/object
            seed: Integer seed or ``np.random.Generator`` for preprocessing
                noise; ``None`` draws fresh entropy.

        Raises:
            ImportError: If TensorFlow is not available for neural style transfer
        """
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for StyleTransfer. "
                "Install it with 'uv sync --package geo-infer-art --extra neural'."
            )

        self.style_image: Image.Image | None = None
        self.content_image: Image.Image | None = None
        self.model: Any | None = None
        self.rng: np.random.Generator = np.random.default_rng(seed)
        requested_device = os.environ.get("GEO_INFER_ART_TF_DEVICE", "CPU").upper()
        if requested_device not in {"CPU", "GPU"}:
            raise ValueError("GEO_INFER_ART_TF_DEVICE must be CPU or GPU")
        # CPU is the reliable default for optional TensorFlow workloads.  Some
        # hosts expose a CUDA device whose driver cannot execute the installed
        # TensorFlow kernels; callers can opt into GPU execution explicitly.
        self._tf_device = f"/{requested_device}:0"

        if style_image is not None:
            self.load_style_image(style_image)

        if content_image is not None:
            self.load_content_image(content_image)

    @classmethod
    def get_predefined_style_path(cls, style_name: str) -> str:
        """
        Get the file path for a predefined style.

        Args:
            style_name: Name of the predefined style

        Returns:
            Path to the style image file

        Raises:
            ValueError: If the style name is not recognized
        """
        if style_name not in cls.PREDEFINED_STYLES:
            raise ValueError(
                f"Unknown style: {style_name}. Available styles: "
                f"{', '.join(cls.PREDEFINED_STYLES.keys())}"
            )

        style_resource = files("geo_infer_art").joinpath(
            "data", "styles", cls.PREDEFINED_STYLES[style_name]
        )
        if style_resource.is_file():
            return str(style_resource)

        raise FileNotFoundError(
            f"Predefined style asset is not installed: {style_resource}"
        )

    def load_style_image(self, style_image: str | np.ndarray | Image.Image) -> None:
        """
        Load the style image to use for transfer.

        Args:
            style_image: Path to style image, or image array/object

        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If the image can't be processed
        """
        if isinstance(style_image, str):
            # Check if it's a predefined style name
            if style_image in self.PREDEFINED_STYLES:
                style_path = self.get_predefined_style_path(style_image)
                self.style_image = self._load_and_preprocess_image(style_path)
            elif not os.path.exists(style_image):
                raise ValueError(f"Unknown style or image path: {style_image}")
            else:
                # Assume it's a file path
                self.style_image = self._load_and_preprocess_image(style_image)
        elif isinstance(style_image, np.ndarray):
            # Preprocess the numpy array
            self.style_image = self._preprocess_image_array(style_image)
        elif isinstance(style_image, Image.Image):
            # Convert PIL Image to numpy array and preprocess
            img_array = np.array(style_image)
            self.style_image = self._preprocess_image_array(img_array)
        else:
            raise ValueError(
                f"Unsupported style_image type: {type(style_image)}. "
                "Expected string path, numpy array, or PIL Image."
            )

    def load_content_image(self, content_image: str | np.ndarray | Image.Image) -> None:
        """
        Load the content image to apply the style to.

        Args:
            content_image: Path to content image, or image array/object

        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If the image can't be processed
        """
        if isinstance(content_image, str):
            self.content_image = self._load_and_preprocess_image(content_image)
        elif isinstance(content_image, np.ndarray):
            self.content_image = self._preprocess_image_array(content_image)
        elif isinstance(content_image, Image.Image):
            img_array = np.array(content_image)
            self.content_image = self._preprocess_image_array(img_array)
        else:
            raise ValueError(
                f"Unsupported content_image type: {type(content_image)}. "
                "Expected string path, numpy array, or PIL Image."
            )

    def _load_and_preprocess_image(self, image_path: str) -> Any:
        """
        Load an image from a file and preprocess it for the VGG model.

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image tensor

        Raises:
            FileNotFoundError: If the image file doesn't exist
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load and convert to RGB (in case of grayscale or RGBA)
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))  # Resize for VGG
        img_array = np.array(img)

        return self._preprocess_image_array(img_array)

    def _preprocess_image_array(self, img_array: np.ndarray) -> Any:
        """
        Preprocess a numpy image array for the VGG model.

        Args:
            img_array: Image as a numpy array

        Returns:
            Preprocessed image tensor
        """
        # Ensure the array has 3 channels (RGB)
        if img_array.ndim == 2:
            # Grayscale to RGB
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 4:
            # RGBA to RGB (discard alpha channel)
            img_array = img_array[..., :3]

        if img_array.shape[0] != 224 or img_array.shape[1] != 224:
            img_array = np.array(
                Image.fromarray(img_array.astype(np.uint8)).resize((224, 224))
            )

        # Convert to float and preprocess for VGG19
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        img_tensor = tf.expand_dims(img_tensor, axis=0)

        # Preprocess using VGG19 preprocessing
        return vgg19.preprocess_input(img_tensor)

    def _build_model(self) -> None:
        """
        Build the style transfer model using VGG19.

        This method creates a model that extracts features at specified
        content and style layers from the VGG19 network.
        """
        # Content layer - where we'll get content features
        content_layers = ["block5_conv2"]

        # Style layers - where we'll get style features
        style_layers = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]

        with tf.device(self._tf_device):
            # Load the VGG19 model without the top classification layer
            vgg = vgg19.VGG19(include_top=False, weights="imagenet")
            vgg.trainable = False

            # Get the outputs of the content and style layers
            style_outputs = [vgg.get_layer(name).output for name in style_layers]
            content_outputs = [vgg.get_layer(name).output for name in content_layers]

            # Combine them into a single model
            model_outputs = style_outputs + content_outputs
            self.model = tf.keras.Model(vgg.input, model_outputs)

    @staticmethod
    def apply(
        geo_data: gpd.GeoDataFrame | np.ndarray,
        style: str = "watercolor",
        content_image: str | np.ndarray | Image.Image | None = None,
        style_image: str | np.ndarray | Image.Image | None = None,
        style_weight: float = 1e-2,
        content_weight: float = 1e4,
        iterations: int = 100,
        color_palette: str | None = None,
    ) -> Image.Image:
        """
        Apply artistic style transfer to geospatial data.

        Args:
            geo_data: GeoDataFrame or raster data to visualize
            style: Name of the predefined style or path to style image
            content_image: Optional content image to use instead of rendering geo_data
            style_image: Optional style image to use instead of predefined style
            style_weight: Weight for style loss in the optimization
            content_weight: Weight for content loss in the optimization
            iterations: Number of optimization iterations
            color_palette: Optional color palette to apply to the result

        Returns:
            Stylized image as a PIL Image

        Raises:
            ImportError: If TensorFlow is not available
            ValueError: If both geo_data and content_image are None
        """
        # Import locally to avoid top-level TensorFlow import
        from geo_infer_art.core.visualization import GeoArt

        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for StyleTransfer. "
                "Install it with 'uv sync --package geo-infer-art --extra neural'."
            )

        # Create a StyleTransfer instance
        transferer = StyleTransfer()

        if geo_data is not None and not hasattr(geo_data, "geometry"):
            raise ValueError(
                "geo_data must be a GeoDataFrame-like object with a geometry column"
            )

        # Load the style image
        if style_image is not None:
            transferer.load_style_image(style_image)
        else:
            transferer.load_style_image(style)

        # Prepare content image
        if content_image is not None:
            transferer.load_content_image(content_image)
        elif geo_data is not None:
            # Render the geo_data to an image
            import io

            geo_art = GeoArt(data=geo_data)
            if color_palette:
                geo_art.apply_style(color_palette=color_palette)
            else:
                geo_art.apply_style()

            # Save figure to a buffer and load as image
            buf = io.BytesIO()
            assert geo_art._figure is not None
            geo_art._figure.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)
            content_img = Image.open(buf)
            transferer.load_content_image(content_img)
            plt.close(geo_art._figure)
        else:
            raise ValueError("Either geo_data or content_image must be provided")

        # Build the model
        transferer._build_model()

        # Apply neural style transfer
        stylized_image = transferer._apply_neural_style_transfer(
            style_weight=style_weight,
            content_weight=content_weight,
            iterations=iterations,
        )

        return stylized_image

    def _apply_neural_style_transfer(
        self,
        style_weight: float = 1e-2,
        content_weight: float = 1e4,
        iterations: int = 100,
    ) -> Image.Image:
        """
        Apply neural style transfer using TensorFlow.

        Args:
            style_weight: Weight for style loss in the optimization
            content_weight: Weight for content loss in the optimization
            iterations: Number of optimization iterations

        Returns:
            Stylized image as a PIL Image

        Raises:
            ValueError: If style or content image is not loaded
        """
        if self.style_image is None:
            raise ValueError("Style image not loaded. Call load_style_image first.")

        if self.content_image is None:
            raise ValueError("Content image not loaded. Call load_content_image first.")

        if self.model is None:
            self._build_model()
        assert self.model is not None

        with tf.device(self._tf_device):
            # Get the feature representations of the content and style images
            style_features = self.model(self.style_image)
            content_features = self.model(self.content_image)

            # Split into style and content features
            num_style_layers = 5  # Defined in _build_model
            style_features = style_features[:num_style_layers]
            content_features = content_features[num_style_layers:]

            # Create a variable to optimize (initialize with content image)
            input_image = tf.Variable(self.content_image)

            # Optimizer for the image
            optimizer = tf.optimizers.Adam(
                learning_rate=0.02, beta_1=0.99, epsilon=1e-1
            )

            # Define the loss function
            def style_content_loss() -> Any:
                assert self.model is not None
                # Process the input image through the model
                outputs = self.model(input_image)
                style_outputs = outputs[:num_style_layers]
                content_outputs = outputs[num_style_layers:]

                # Calculate style loss
                style_loss = tf.add_n(
                    [
                        tf.reduce_mean((style_outputs[i] - style_features[i]) ** 2)
                        for i in range(num_style_layers)
                    ]
                )
                style_loss *= style_weight / num_style_layers

                # Calculate content loss
                content_loss = tf.add_n(
                    [
                        tf.reduce_mean((content_outputs[i] - content_features[i]) ** 2)
                        for i in range(len(content_outputs))
                    ]
                )
                content_loss *= content_weight / len(content_outputs)

                # Total loss
                return style_loss + content_loss

            # Optimization loop
            for i in range(iterations):
                with tf.GradientTape() as tape:
                    loss = style_content_loss()

                # Get the gradients and apply them to the input image
                gradients = tape.gradient(loss, input_image)
                optimizer.apply_gradients([(gradients, input_image)])

                # Clip the pixel values to be between 0 and 255
                input_image.assign(tf.clip_by_value(input_image, 0.0, 255.0))

                if i % 10 == 0:
                    logger.info("Style-transfer iteration %s: loss=%s", i, loss.numpy())

            # Convert the result back to a PIL image
            result = input_image.numpy()
        result = result.reshape(result.shape[1:])
        result = np.clip(result, 0, 255).astype("uint8")

        return Image.fromarray(result)

    def save(self, image: Image.Image, output_path: str) -> str:
        """
        Save a stylized image to a file.

        Args:
            image: PIL Image to save
            output_path: Path where the file should be saved

        Returns:
            The path to the saved file
        """
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        image.save(output_path)
        return output_path

    @classmethod
    def blend_styles(
        cls,
        geo_data: gpd.GeoDataFrame | np.ndarray,
        styles: list[str],
        weights: list[float] | None = None,
        iterations: int = 100,
        **kwargs: Any,
    ) -> Image.Image:
        """
        Apply multiple styles to geospatial data with different weights.

        Args:
            geo_data: Geospatial data to visualize
            styles: List of style names to blend
            weights: Weights for each style (must sum to 1.0 if provided)
            iterations: Number of optimization iterations
            **kwargs: Additional parameters for style transfer

        Returns:
            Stylized image with blended styles

        Raises:
            ValueError: If styles and weights have different lengths or weights don't sum to 1.0
        """
        if len(styles) < 2:
            raise ValueError("At least 2 styles required for blending")

        if weights is not None and len(weights) != len(styles):
            raise ValueError("Number of weights must match number of styles")

        if weights is not None and not np.isclose(sum(weights), 1.0):
            raise ValueError("Style weights must sum to 1.0")

        # Apply each style separately
        styled_images = []
        for i, style in enumerate(styles):
            style_weight = weights[i] if weights else 1.0 / len(styles)

            styled_img = cls.apply(
                geo_data=geo_data,
                style=style,
                iterations=iterations,
                style_weight=kwargs.get("style_weight", 1e-2) * style_weight,
                **{k: v for k, v in kwargs.items() if k != "style_weight"},
            )
            styled_images.append(styled_img)

        # Blend the styled images
        if weights is None:
            weights = [1.0 / len(styles)] * len(styles)

        # Start with the first image
        blended = np.array(styled_images[0])

        # Blend with remaining images
        for img, weight in zip(styled_images[1:], weights[1:]):
            img_array = np.array(img)
            blended = blended * (1 - weight) + img_array * weight

        return Image.fromarray(blended.astype(np.uint8))

    @classmethod
    def create_style_variation(
        cls,
        geo_data: gpd.GeoDataFrame | np.ndarray,
        base_style: str,
        variation_params: dict,
        **kwargs: Any,
    ) -> Image.Image:
        """
        Create a variation of a base style with custom parameters.

        Args:
            geo_data: Geospatial data to visualize
            base_style: Base style name to start from
            variation_params: Parameters to vary (style_weight, content_weight, iterations)
            **kwargs: Additional parameters

        Returns:
            Stylized image with style variation
        """
        # Update kwargs with variation parameters
        updated_kwargs = kwargs.copy()
        updated_kwargs.update(variation_params)

        return cls.apply(geo_data=geo_data, style=base_style, **updated_kwargs)

    def preprocess_for_style(self, style_category: str = "landscape") -> None:
        """
        Preprocess images based on style category for better results.

        Args:
            style_category: Category of style ("landscape", "portrait", "abstract", "urban")
        """
        if self.content_image is None or self.style_image is None:
            return

        # Category-specific preprocessing
        if style_category == "landscape":
            # Ensure landscape aspect ratio and scale
            self._ensure_aspect_ratio(16 / 9)
        elif style_category == "portrait":
            self._ensure_aspect_ratio(9 / 16)
        elif style_category == "abstract":
            # Add noise for abstract styles
            self._add_preprocessing_noise()
        elif style_category == "urban":
            # Enhance edges for urban styles
            self._enhance_edges()

    def _ensure_aspect_ratio(self, target_ratio: float) -> None:
        """Ensure images have the target aspect ratio.

        Crops or pads the content and style images so their width-to-height
        ratio matches *target_ratio*.  The longer dimension is cropped
        symmetrically from both sides so the center of the image is preserved.

        Args:
            target_ratio: Desired width / height ratio (e.g. 16/9).
        """
        for attr in ("content_image", "style_image"):
            img = getattr(self, attr, None)
            if img is None:
                continue
            w, h = img.size
            current_ratio = w / h if h > 0 else 1.0
            if abs(current_ratio - target_ratio) < 0.01:
                continue
            if current_ratio > target_ratio:
                new_w = int(h * target_ratio)
                offset = (w - new_w) // 2
                img = img.crop((offset, 0, offset + new_w, h))
            else:
                new_h = int(w / target_ratio)
                offset = (h - new_h) // 2
                img = img.crop((0, offset, w, offset + new_h))
            setattr(self, attr, img)

    def _add_preprocessing_noise(self) -> None:
        """Add controlled Gaussian noise to the content image.

        Small amounts of noise can help abstract style transfer converge to
        more interesting solutions by breaking up uniform regions in the
        content image.
        """
        if self.content_image is None:
            return
        arr = np.array(self.content_image, dtype=np.float32)
        noise = self.rng.normal(loc=0.0, scale=8.0, size=arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        self.content_image = Image.fromarray(arr)

    def _enhance_edges(self) -> None:
        """Enhance edges in the content image for urban style transfer.

        Uses a Laplacian-like kernel approximation to sharpen edges, which
        helps preserve architectural lines during style transfer.
        """
        if self.content_image is None:
            return
        arr = np.array(self.content_image, dtype=np.float32)
        # Simple sharpening: original + scaled high-pass component
        # Approximate high-pass by subtracting a blurred version
        from PIL import ImageFilter

        blurred = self.content_image.filter(ImageFilter.GaussianBlur(radius=2))
        blurred_arr = np.array(blurred, dtype=np.float32)
        sharpened = np.clip(arr + 0.5 * (arr - blurred_arr), 0, 255).astype(np.uint8)
        self.content_image = Image.fromarray(sharpened)

    def get_style_info(self, style_name: str) -> dict:
        """
        Get detailed information about a style.

        Args:
            style_name: Name of the style

        Returns:
            Dictionary with style information
        """
        if style_name not in self.PREDEFINED_STYLES:
            return {"error": f"Style '{style_name}' not found"}

        style_info = {
            "name": style_name,
            "filename": self.PREDEFINED_STYLES[style_name],
            "category": self._categorize_style(style_name),
            "recommended_params": self._get_recommended_params(style_name),
        }

        return style_info

    def _categorize_style(self, style_name: str) -> str:
        """Categorize a style for better preprocessing."""
        style_categories = {
            "watercolor": "landscape",
            "oil_painting": "portrait",
            "sketch": "abstract",
            "abstract": "abstract",
            "impressionist": "landscape",
            "van_gogh": "landscape",
            "monet": "landscape",
            "picasso": "abstract",
            "dali": "abstract",
            "kandinsky": "abstract",
            "pollock": "abstract",
            "hokusai": "landscape",
            "vermeer": "portrait",
            "cezanne": "landscape",
            "matisse": "abstract",
        }

        return style_categories.get(style_name, "landscape")

    def _get_recommended_params(self, style_name: str) -> dict:
        """Get recommended parameters for a style."""
        recommendations = {
            "watercolor": {
                "style_weight": 1e-1,
                "content_weight": 1e3,
                "iterations": 200,
            },
            "oil_painting": {
                "style_weight": 1e-2,
                "content_weight": 1e4,
                "iterations": 300,
            },
            "sketch": {"style_weight": 1e0, "content_weight": 1e2, "iterations": 100},
            "abstract": {"style_weight": 1e1, "content_weight": 1e1, "iterations": 500},
            "impressionist": {
                "style_weight": 1e-1,
                "content_weight": 1e3,
                "iterations": 250,
            },
        }

        return recommendations.get(
            style_name, {"style_weight": 1e-2, "content_weight": 1e4, "iterations": 100}
        )
