"""
Image Data Loading Module

Handles the loading of various microscopy image formats, with a focus on using
Bio-Formats for broad compatibility and metadata extraction.
"""

import numpy as np
import io
import warnings
import tempfile
import os

# Attempt to import primary and fallback loaders
try:
    import javabridge
    import bioformats
    BIOFORMATS_AVAILABLE = True
except ImportError:
    BIOFORMATS_AVAILABLE = False
    warnings.warn("python-bioformats not found. Falling back to basic image loaders. For full format support, please install python-bioformats and jpype1.")

try:
    from tifffile import imread as tifffile_imread
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class DataLoader:
    """Manages the loading of image data from various file formats."""

    def __init__(self):
        self.jvm_started = False
        self._start_jvm()

    def __del__(self):
        self._stop_jvm()

    def _start_jvm(self):
        """Starts the Java Virtual Machine (JVM) for Bio-Formats, if available."""
        if BIOFORMATS_AVAILABLE and not self.jvm_started:
            try:
                javabridge.start_vm(class_path=bioformats.JARS, max_heap_size='8G')
                self.jvm_started = True
            except Exception as e:
                warnings.warn(f"Could not start JVM for Bio-Formats: {e}")

    def _stop_jvm(self):
        """Stops the JVM if it was started."""
        if self.jvm_started:
            try:
                javabridge.kill_vm()
                self.jvm_started = False
            except Exception as e:
                warnings.warn(f"Could not stop JVM for Bio-Formats: {e}")

    def get_channel_count(self, filepath_or_buffer) -> int:
        """Get the number of channels in an image file."""
        if BIOFORMATS_AVAILABLE and self.jvm_started:
            try:
                fd, path = tempfile.mkstemp()
                os.close(fd)
                filepath_or_buffer.seek(0)
                with open(path, 'wb') as tmp:
                    tmp.write(filepath_or_buffer.read())
                
                metadata = bioformats.get_omexml_metadata(path)
                parser = bioformats.OMEXML(metadata)
                return parser.image(0).Pixels.SizeC
            except Exception as e:
                warnings.warn(f"Could not get channel count with Bio-Formats: {e}")
            finally:
                os.remove(path)
        return 1 # Fallback

    def load_image(self, filepath_or_buffer, channel: int = 0) -> dict:
        """
        Load a specific channel from an image using the best available method.
        """
        if BIOFORMATS_AVAILABLE and self.jvm_started:
            try:
                return self._load_with_bioformats(filepath_or_buffer, channel)
            except Exception as e:
                warnings.warn(f"Bio-Formats failed: {e}. Falling back.")

        # Fallback loaders (tifffile, Pillow) typically load only the first channel
        if channel > 0:
            warnings.warn("Multi-channel loading may not be supported by fallback loaders.")

        if TIFFFILE_AVAILABLE:
            try:
                return self._load_with_tifffile(filepath_or_buffer)
            except Exception as e:
                warnings.warn(f"Tifffile failed: {e}. Falling back.")
        
        if PIL_AVAILABLE:
            try:
                file_data = filepath_or_buffer.read()
                return self._load_as_image_fallback(file_data)
            except Exception as e:
                raise RuntimeError(f"Failed to load image with all available loaders: {e}")
        
        raise RuntimeError("No suitable image loading library is available.")

    def _load_with_bioformats(self, filepath_or_buffer, channel: int) -> dict:
        """Load a specific channel and metadata using Bio-Formats."""
        fd, path = tempfile.mkstemp()
        os.close(fd)
        
        try:
            filepath_or_buffer.seek(0)
            with open(path, 'wb') as tmp:
                tmp.write(filepath_or_buffer.read())

            metadata = bioformats.get_omexml_metadata(path)
            parser = bioformats.OMEXML(metadata)

            image_data = bioformats.load_image(path, c=channel, z=0, t=0, series=0)

            size_x = parser.image(0).Pixels.PhysicalSizeX
            size_y = parser.image(0).Pixels.PhysicalSizeY
            size_z = parser.image(0).Pixels.PhysicalSizeZ
            voxel_size = (
                float(size_z) if size_z is not None else 1.0,
                float(size_y) if size_y is not None else 0.5,
                float(size_x) if size_x is not None else 0.5,
            )

            return {
                "image_data": image_data,
                "voxel_size": voxel_size,
                "metadata": metadata,
                "status": "success",
                "loader": "Bio-Formats"
            }
        finally:
            os.remove(path)

    def _load_with_tifffile(self, filepath_or_buffer) -> dict:
        """
        Load a TIFF file. Note: Simple implementation, may not handle multi-channel correctly.
        """
        filepath_or_buffer.seek(0)
        image_data = tifffile_imread(filepath_or_buffer)
        voxel_size = (1.0, 0.5, 0.5)
        return {
            "image_data": image_data,
            "voxel_size": voxel_size,
            "metadata": "Loaded with Tifffile",
            "status": "success",
            "loader": "Tifffile"
        }

    def _load_as_image_fallback(self, file_data: bytes) -> dict:
        """Fallback for basic formats."""
        with io.BytesIO(file_data) as f:
            img = Image.open(f)
            image_data = np.array(img)
        return {
            "image_data": image_data,
            "voxel_size": (1.0,) * image_data.ndim,
            "metadata": "Loaded with Pillow",
            "status": "success",
            "loader": "Pillow"
        }
