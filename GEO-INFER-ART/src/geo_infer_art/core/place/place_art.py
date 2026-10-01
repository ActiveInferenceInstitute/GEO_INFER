"""
PlaceArt module for creating art based on specific locations and places.
"""

import logging
import os
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geo_infer_art.core.generation import ProceduralArt

logger = logging.getLogger(__name__)


class PlaceArt:
    """
    A class for creating art based on the unique characteristics of geographic locations.

    The PlaceArt class provides methods for generating artistic representations
    inspired by the features, culture, and atmosphere of specific places.

    Attributes:
        location: Information about the geographic location
        data: Geospatial data for the location
        image: The generated artistic representation
    """

    VALID_STYLES = {"abstract", "topographic", "cultural", "mixed_media"}

    def __init__(
        self,
        location: dict[str, Any] | None = None,
        data: gpd.GeoDataFrame | None = None,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        """
        Initialize a PlaceArt object.

        Args:
            location: Dictionary with location information (name, coordinates, etc.)
            data: GeoDataFrame with geospatial data for the location
            seed: Integer seed or ``np.random.Generator`` for texture overlays;
                ``None`` draws fresh entropy.
        """
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.location = location or {}
        self.data = data
        self.image: Image.Image | None = None
        self._figure = None
        self.metadata: dict[str, Any] = {}

    @staticmethod
    def _as_pil_image(image: Image.Image | np.ndarray) -> Image.Image:
        return image if isinstance(image, Image.Image) else Image.fromarray(image)

    @staticmethod
    def _fetch_location_data(lat: float, lon: float, radius_km: float = 1.0) -> dict:
        """Return deterministic location metadata for coordinate-based artwork."""
        return {
            "name": f"Location {lat:.4f}, {lon:.4f}",
            "coordinates": (lat, lon),
            "radius_km": radius_km,
        }

    @staticmethod
    def _fetch_place_data(place_name: str) -> dict:
        """Return metadata for a named place.

        Raises:
            ValueError: If the place name is not in the known-places table
                (matching the from_place_name contract and CulturalMap.from_region).
        """
        known_places = {
            "new york": (40.7128, -74.0060, "United States"),
            "paris": (48.8566, 2.3522, "France"),
            "tokyo": (35.6762, 139.6503, "Japan"),
            "cairo": (30.0444, 31.2357, "Egypt"),
            "sydney": (-33.8688, 151.2093, "Australia"),
            "rio de janeiro": (-22.9068, -43.1729, "Brazil"),
            "cape town": (-33.9249, 18.4241, "South Africa"),
            "moscow": (55.7558, 37.6173, "Russia"),
            "mumbai": (19.0760, 72.8777, "India"),
            "beijing": (39.9042, 116.4074, "China"),
        }

        place_key = place_name.lower()
        if place_key not in known_places:
            raise ValueError(
                f"Unknown place: {place_name}. Known places: "
                f"{', '.join(sorted(known_places.keys()))}"
            )

        lat, lon, country = known_places[place_key]
        return {"name": place_name, "coordinates": (lat, lon), "country": country}

    @classmethod
    def from_coordinates(
        cls,
        lat: float,
        lon: float,
        name: str | None = None,
        radius_km: float = 1.0,
        style: str = "abstract",
    ) -> "PlaceArt":
        """
        Create place-based art from geographic coordinates.

        Args:
            lat: Latitude (-90 to 90)
            lon: Longitude (-180 to 180)
            name: Optional name of the location
            radius_km: Radius in kilometers to consider around the point
            style: Artistic style to apply ("abstract", "topographic", "cultural", etc.)

        Returns:
            A new PlaceArt object with generated art

        Raises:
            ValueError: If coordinates are invalid
        """
        # Validate coordinates
        if not -90 <= lat <= 90:
            raise ValueError("Latitude must be between -90 and 90 degrees.")

        if not -180 <= lon <= 180:
            raise ValueError("Longitude must be between -180 and 180 degrees.")
        if style not in cls.VALID_STYLES:
            raise ValueError(
                f"Unsupported style: {style}. Supported styles: {', '.join(sorted(cls.VALID_STYLES))}"
            )

        location = cls._fetch_location_data(lat, lon, radius_km)
        location["name"] = name or location.get(
            "name", f"Location {lat:.4f}, {lon:.4f}"
        )
        location["coordinates"] = (lat, lon)
        location["radius_km"] = radius_km
        location["style"] = style

        # Create PlaceArt object
        place_art = cls(location=location)

        # Generate art based on style
        place_art._generate_art()

        return place_art

    @classmethod
    def from_place_name(
        cls,
        place_name: str,
        style: str = "abstract",
        include_data: bool = False,
    ) -> "PlaceArt":
        """
        Create place-based art from a named location.

        Args:
            place_name: Name of the place (city, country, landmark, etc.)
            style: Artistic style to apply
            include_data: Whether to fetch and include geospatial data

        Returns:
            A new PlaceArt object with generated art

        Raises:
            ValueError: If the place cannot be found or geocoded
        """
        if style not in cls.VALID_STYLES:
            raise ValueError(
                f"Unsupported style: {style}. Supported styles: {', '.join(sorted(cls.VALID_STYLES))}"
            )

        location = cls._fetch_place_data(place_name)
        location["style"] = style
        location.setdefault("radius_km", 5.0)

        # Create PlaceArt object
        place_art = cls(location=location)

        # Generate art
        place_art._generate_art()

        return place_art

    def _generate_art(self) -> None:
        """
        Generate art based on the location and specified style.
        """
        if not self.location:
            raise ValueError("Location information is required for art generation.")

        style = self.location.get("style", "abstract")
        coordinates = self.location.get("coordinates")

        if not coordinates:
            raise ValueError("Location coordinates are required for art generation.")

        lat, lon = coordinates
        if style not in self.VALID_STYLES:
            raise ValueError(
                f"Unsupported style: {style}. Supported styles: {', '.join(sorted(self.VALID_STYLES))}"
            )

        if style == "abstract":
            self._generate_abstract_art(lat, lon)
        elif style == "topographic":
            self._generate_topographic_art(lat, lon)
        elif style == "cultural":
            self._generate_cultural_art(lat, lon)
        elif style == "mixed_media":
            self._generate_mixed_media_art(lat, lon)
        else:
            raise ValueError(
                f"Unsupported style: {style}. Supported styles: {', '.join(sorted(self.VALID_STYLES))}"
            )

    def _generate_abstract_art(self, lat: float, lon: float) -> None:
        """
        Generate abstract art based on location coordinates.

        Args:
            lat: Latitude
            lon: Longitude
        """
        # Use ProceduralArt with noise field algorithm
        art = ProceduralArt.from_geo_coordinates(
            lat=lat,
            lon=lon,
            algorithm="noise_field",
            additional_params={
                "octaves": 8,
                "persistence": 0.6,
                "lacunarity": 2.2,
                "scale": 120.0,
                # Use color palette based on latitude (warmer for lower latitudes)
                "color_palette": (
                    "sunset"
                    if abs(lat) < 30
                    else "ocean"
                    if abs(lat) >= 60
                    else "forest"
                ),
            },
        )

        # Store the image
        self.image = art.image

    def _generate_topographic_art(self, lat: float, lon: float) -> None:
        """
        Generate topographic-inspired art based on location coordinates.

        Args:
            lat: Latitude
            lon: Longitude
        """
        from geo_infer_art.core.generation import GenerativeMap

        # Create a bounding box around the coordinates
        # Approximate 1 degree as 111 km at the equator
        radius_deg = self.location.get("radius_km", 1.0) / 111.0
        # Adjust for longitude compression at higher latitudes
        lon_radius = radius_deg / max(0.1, np.cos(np.radians(abs(lat))))

        bbox = (lon - lon_radius, lat - radius_deg, lon + lon_radius, lat + radius_deg)

        # Use GenerativeMap to create topographic art
        gen_map = GenerativeMap.from_elevation(
            region=bbox,
            resolution=800,
            abstraction_level=0.7,
            style="contour_flow",
        )

        # Store the image
        self.image = gen_map.image

    def _generate_cultural_art(self, lat: float, lon: float) -> None:
        """
        Generate culture-inspired art based on location coordinates.

        Args:
            lat: Latitude
            lon: Longitude
        """
        # For demonstration, we'll use L-systems with parameters that vary by
        # geographic region (this is a simplification of cultural patterns)

        # Determine art parameters based on geographic region
        # This is a very simplified model that associates different
        # pattern types with different regions of the world

        # Normalize coordinates to 0-1 range
        norm_lat = (lat + 90) / 180
        norm_lon = (lon + 180) / 360

        # Create region-specific parameters
        if lon > -30 and lon < 60 and lat > 0 and lat < 40:
            # Europe/Middle East/North Africa - geometric patterns
            params = {
                "algorithm": "l_system",
                "axiom": "F+F+F+F",
                "rules": {"F": "FF+F+F+FF+F+FF-F"},
                "iterations": 3,
                "angle": 90,
                "color_palette": "sunset",
            }
        elif lon > 60 and lon < 150 and lat > 10 and lat < 50:
            # Asia - flowing organic patterns
            params = {
                "algorithm": "l_system",
                "axiom": "F",
                "rules": {"F": "FF+[+F-F-F]-[-F+F+F]"},
                "iterations": 4,
                "angle": 25,
                "color_palette": "ocean",
            }
        elif lon > -120 and lon < -30 and lat > 15 and lat < 50:
            # North America - angular and geometric
            params = {
                "algorithm": "voronoi",
                "num_points": 40,
                "point_clustering": 0.3,
                "edge_width": 1.2,
                "color_palette": "earth",
            }
        elif lat < 0:
            # Southern Hemisphere - more organic forms
            params = {
                "algorithm": "reaction_diffusion",
                "iterations": 60,
                "feed_rate": 0.037,
                "kill_rate": 0.06,
                "color_palette": "forest",
            }
        else:
            # Default - abstract noise
            params = {
                "algorithm": "noise_field",
                "octaves": 6,
                "persistence": 0.5,
                "color_palette": "pastel",
            }

        # Add geographic seed
        params["seed"] = int((norm_lat * 1000) + (norm_lon * 10000))

        algorithm_str = str(params.pop("algorithm"))
        art = ProceduralArt(
            algorithm=algorithm_str,
            params=params,
            resolution=(800, 800),
        )

        # Generate the art
        art.generate()

        # Store the image
        self.image = art.image

    def _generate_mixed_media_art(self, lat: float, lon: float) -> None:
        """
        Generate mixed media art combining multiple techniques.

        Args:
            lat: Latitude
            lon: Longitude
        """
        # First generate a base layer with noise field
        base_art = ProceduralArt.from_geo_coordinates(
            lat=lat,
            lon=lon,
            algorithm="noise_field",
            additional_params={
                "octaves": 6,
                "persistence": 0.5,
                "color_palette": "autumn",
            },
        )

        # Then add a cultural pattern layer
        # Simplified cultural pattern selection based on region
        if lon > 0:  # Eastern Hemisphere
            pattern_art = ProceduralArt(
                algorithm="l_system",
                params={
                    "axiom": "F",
                    "rules": {"F": "F+F-F-F+F"},
                    "iterations": 3,
                    "angle": 72,
                    "line_width": 0.8,
                    "seed": int(lat * 100 + lon),
                    "color_palette": "ocean",
                },
                resolution=(800, 800),
            )
        else:  # Western Hemisphere
            pattern_art = ProceduralArt(
                algorithm="voronoi",
                params={
                    "num_points": 30,
                    "point_clustering": 0.4,
                    "edge_width": 1.0,
                    "seed": int(lat * 100 + lon),
                    "color_palette": "forest",
                },
                resolution=(800, 800),
            )

        pattern_art.generate()

        # Combine the images with alpha blending
        if base_art.image is not None and pattern_art.image is not None:
            base_img = self._as_pil_image(base_art.image)
            pattern_img = self._as_pil_image(pattern_art.image)

            # Resize if needed
            base_width, base_height = base_img.size
            pattern_img = pattern_img.resize((base_width, base_height))

            # Convert to RGBA if not already
            if base_img.mode != "RGBA":
                base_img = base_img.convert("RGBA")
            if pattern_img.mode != "RGBA":
                pattern_img = pattern_img.convert("RGBA")

            # Blend images
            blended = Image.blend(base_img, pattern_img, alpha=0.7)

            # Store the result
            self.image = blended
        else:
            # Fallback to base image if blending fails
            self.image = base_art.image

    def add_metadata_overlay(
        self, position: str = "bottom", opacity: float = 0.7
    ) -> "PlaceArt":
        """
        Add location metadata as an overlay on the artwork.

        Args:
            position: Position of the overlay ("top", "bottom", "left", "right")
            opacity: Opacity of the overlay (0.0 to 1.0)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If no image has been generated
        """
        if self.image is None:
            raise ValueError("No image has been generated.")

        # Get location name
        location_name = self.location.get("name", "Unknown Location")
        lat, lon = self.location.get("coordinates", (0, 0))

        # Create a PIL image from the numpy array
        img = self._as_pil_image(self.image)
        width, height = img.size

        # Create a new image with text overlay
        from PIL import ImageDraw, ImageFont

        # Create a drawing context
        draw = ImageDraw.Draw(img)

        # Try to get a font
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont
        try:
            # Try to load a nice font
            font = ImageFont.truetype("Arial", 24)
        except OSError:
            # Fallback to default font
            font = ImageFont.load_default()

        # Create metadata text
        metadata_text = f"{location_name}\nLat: {lat:.4f}, Lon: {lon:.4f}"

        # Calculate text size
        text_bbox = draw.textbbox((0, 0), metadata_text, font=font)
        text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])

        # Determine position
        if position == "bottom":
            text_position = ((width - text_size[0]) // 2, height - text_size[1] - 20)
        elif position == "top":
            text_position = ((width - text_size[0]) // 2, 20)
        elif position == "left":
            text_position = (20, (height - text_size[1]) // 2)
        elif position == "right":
            text_position = (width - text_size[0] - 20, (height - text_size[1]) // 2)
        else:
            text_position = ((width - text_size[0]) // 2, height - text_size[1] - 20)

        # Create semi-transparent background for text
        bg_padding = 10
        bg_bounds = (
            text_position[0] - bg_padding,
            text_position[1] - bg_padding,
            text_position[0] + text_size[0] + bg_padding,
            text_position[1] + text_size[1] + bg_padding,
        )

        # Draw background
        draw.rectangle(bg_bounds, fill=(0, 0, 0, int(255 * opacity)))

        # Draw text
        draw.text(text_position, metadata_text, fill=(255, 255, 255, 255), font=font)

        # Update the image
        self.image = img

        return self

    def save(self, output_path: str) -> str:
        """
        Save the generated art to a file.

        Args:
            output_path: Path where the file should be saved

        Returns:
            The path to the saved file

        Raises:
            ValueError: If no image has been generated
        """
        if self.image is None:
            raise ValueError("No image generated. Generate art first.")

        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        img = self._as_pil_image(self.image)
        img.save(output_path)

        return output_path

    def show(self) -> None:
        """
        Display the generated art.

        Raises:
            ValueError: If no image has been generated
        """
        if self.image is None:
            raise ValueError("No image generated. Generate art first.")

        # Create a new figure
        figure = plt.figure(figsize=(10, 10))
        plt.imshow(self.image)
        plt.axis("off")
        plt.tight_layout()
        if "agg" in plt.get_backend().lower() or not plt.isinteractive():
            figure.canvas.draw()
            plt.close(figure)
            return
        plt.show()

    def create_series(self, styles: list[str], output_dir: str = "output") -> list[str]:
        """
        Create a series of artworks for the same location with different styles.

        Args:
            styles: List of artistic styles to apply
            output_dir: Directory to save the artwork series

        Returns:
            List of file paths for the created artworks

        Raises:
            ValueError: If no location is set
        """
        if not self.location:
            raise ValueError("No location set for art generation.")

        output_paths = []

        for style in styles:
            # Create a new PlaceArt with the same location but different style
            self.location["style"] = style

            # Regenerate the art
            self._generate_art()

            # Add metadata overlay
            self.add_metadata_overlay()

            # Save the artwork
            filename = f"place_art_{self.location['name'].lower().replace(' ', '_')}_{style}.png"
            output_path = os.path.join(output_dir, filename)
            self.save(output_path)
            output_paths.append(output_path)

        return output_paths

    def blend_with_style(self, style: str, blend_ratio: float = 0.5) -> "PlaceArt":
        """
        Blend the current artwork with another style.

        Args:
            style: Style to blend with
            blend_ratio: Ratio of blending (0.0 = all original, 1.0 = all new style)

        Returns:
            A new PlaceArt object with blended styles

        Raises:
            ValueError: If no image has been generated
        """
        if self.image is None:
            raise ValueError("No image generated. Generate art first.")

        # Create a new PlaceArt with the blended style
        original_style = self.location.get("style", "abstract")
        blended_location = self.location.copy()
        blended_location["style"] = f"{original_style}_blend_{style}"

        # Generate art with the new style
        blended_art = PlaceArt(location=blended_location)
        blended_art._generate_art()

        if blended_art.image is not None and self.image is not None:
            # Blend the images
            img1 = self._as_pil_image(blended_art.image)
            img2 = self._as_pil_image(self.image)
            blended_image = Image.blend(img2, img1, alpha=blend_ratio)

            # Create new PlaceArt with blended result
            result_art = PlaceArt(location=blended_location)
            result_art.image = blended_image

            return result_art

        return self

    def add_artistic_elements(self, elements: list[str], **kwargs: Any) -> "PlaceArt":
        """
        Add artistic elements to the place art.

        Args:
            elements: List of artistic elements ("frame", "signature", "texture", "overlay")
            **kwargs: Parameters for the artistic elements

        Returns:
            Self for method chaining

        Raises:
            ValueError: If no image has been generated or element is unsupported
        """
        if self.image is None:
            raise ValueError("No image generated. Generate art first.")

        img = self._as_pil_image(self.image)

        for element in elements:
            if element == "frame":
                self._add_frame(img, **kwargs)
            elif element == "signature":
                self._add_signature(img, **kwargs)
            elif element == "texture":
                self._add_texture_overlay(img, **kwargs)
            elif element == "overlay":
                self._add_overlay_pattern(img, **kwargs)
            else:
                raise ValueError(f"Unsupported artistic element: {element}")

        self.image = img

        return self

    def _add_frame(self, img: Image.Image, **kwargs: Any) -> None:
        """Add a decorative frame to the image."""
        frame_style = kwargs.get("frame_style", "simple")
        frame_color = kwargs.get("frame_color", "#8B4513")
        frame_width = kwargs.get("frame_width", 20)

        draw = ImageDraw.Draw(img)
        width, height = img.size

        if frame_style == "simple":
            # Simple border
            draw.rectangle(
                [0, 0, width - 1, height - 1], outline=frame_color, width=frame_width
            )
        elif frame_style == "ornate":
            # More decorative frame
            # Draw multiple borders
            for i in range(3):
                border_width = frame_width - i * 5
                if border_width <= 0:
                    break

                offset = i * 3
                draw.rectangle(
                    [offset, offset, width - 1 - offset, height - 1 - offset],
                    outline=frame_color,
                    width=border_width - 6,
                )

    def _add_signature(self, img: Image.Image, **kwargs: Any) -> None:
        """Add an artistic signature to the image."""
        signature_text = kwargs.get("signature_text", "Place Art")
        signature_color = kwargs.get("signature_color", "#666666")
        signature_position = kwargs.get("signature_position", "bottom_right")

        font: ImageFont.FreeTypeFont | ImageFont.ImageFont
        try:
            font = ImageFont.truetype("Arial", 20)
        except OSError:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(img)
        width, height = img.size

        # Position the signature
        text_bbox = draw.textbbox((0, 0), signature_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        if signature_position == "bottom_right":
            x = width - text_width - 20
            y = height - text_height - 20
        elif signature_position == "bottom_left":
            x = 20
            y = height - text_height - 20
        elif signature_position == "top_right":
            x = width - text_width - 20
            y = 20
        elif signature_position == "top_left":
            x = 20
            y = 20
        else:
            x = width - text_width - 20
            y = height - text_height - 20

        # Add signature with background
        draw.rectangle(
            [x - 5, y - 5, x + text_width + 5, y + text_height + 5],
            fill=(255, 255, 255, 128),
        )
        draw.text((x, y), signature_text, fill=signature_color, font=font)

    def _add_texture_overlay(self, img: Image.Image, **kwargs: Any) -> None:
        """Add a texture overlay to the image."""
        texture_type = kwargs.get("texture_type", "paper")
        texture_opacity = kwargs.get("texture_opacity", 0.1)

        width, height = img.size

        if texture_type == "paper":
            # Create paper-like texture
            texture = self.rng.integers(240, 255, (height, width, 3), dtype=np.uint8)
            texture_img = Image.fromarray(texture)

            # Blend with original
            img = Image.blend(img, texture_img, texture_opacity)

        elif texture_type == "canvas":
            # Create canvas-like texture
            texture = self.rng.integers(245, 255, (height, width, 3), dtype=np.uint8)
            # Add some subtle variations
            texture[::10, ::10] = [240, 240, 240]
            texture_img = Image.fromarray(texture)

            img = Image.blend(img, texture_img, texture_opacity)

    def _add_overlay_pattern(self, img: Image.Image, **kwargs: Any) -> None:
        """Add a decorative pattern overlay."""
        pattern_type = kwargs.get("pattern_type", "diagonal")
        pattern_color = kwargs.get("pattern_color", "#000000")
        kwargs.get("pattern_opacity", 0.05)

        draw = ImageDraw.Draw(img)
        width, height = img.size

        if pattern_type == "diagonal":
            # Diagonal lines pattern
            for i in range(-height, width, 20):
                draw.line([(i, 0), (i + height, height)], fill=pattern_color, width=1)

        elif pattern_type == "dots":
            # Dot pattern
            for x in range(0, width, 30):
                for y in range(0, height, 30):
                    draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=pattern_color)

        elif pattern_type == "grid":
            # Grid pattern
            for x in range(0, width, 50):
                draw.line([(x, 0), (x, height)], fill=pattern_color, width=1)
            for y in range(0, height, 50):
                draw.line([(0, y), (width, y)], fill=pattern_color, width=1)

    def get_location_info(self) -> dict:
        """
        Get detailed information about the location.

        Returns:
            Dictionary with location information including coordinates,
            style, and any additional metadata
        """
        return self.location.copy()

    def export_metadata(self, output_path: str) -> str:
        """
        Export location and generation metadata to a JSON file.

        Args:
            output_path: Path for the metadata file

        Returns:
            Path to the exported metadata file
        """
        import json

        metadata = {
            "location": self.location,
            "generation_info": {
                "module": "PlaceArt",
                "timestamp": self.metadata.get("timestamp", ""),
                "parameters": self.metadata.get("parameters", {}),
            },
        }

        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return output_path

    def __repr__(self) -> str:
        """Return a string representation of the PlaceArt object."""
        location_name = self.location.get("name", "Unknown Location")
        style = self.location.get("style", "unknown")
        coordinates = self.location.get("coordinates", "Unknown")

        return f"PlaceArt(location='{location_name}', style='{style}', coordinates={coordinates})"
