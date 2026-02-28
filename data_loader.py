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
import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional

# Attempt to import primary and fallback loaders
try:
    import javabridge
    import bioformats
    BIOFORMATS_AVAILABLE = True
except ImportError:
    BIOFORMATS_AVAILABLE = False
    warnings.warn("python-bioformats not found. Falling back to basic image loaders. For full format support, please install python-bioformats and jpype1.")

try:
    import tifffile
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
        if TIFFFILE_AVAILABLE:
            try:
                filepath_or_buffer.seek(0)
                with tifffile.TiffFile(filepath_or_buffer) as tif:
                    series = tif.series[0]
                    axes = getattr(series, 'axes', '')
                    shape = tuple(getattr(series, 'shape', ()))
                    if 'C' in axes:
                        return int(shape[axes.index('C')])
            except Exception as e:
                warnings.warn(f"Could not get channel count with tifffile: {e}")
            finally:
                filepath_or_buffer.seek(0)
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

        if TIFFFILE_AVAILABLE:
            try:
                return self._load_with_tifffile(filepath_or_buffer, channel=channel)
            except Exception as e:
                warnings.warn(f"Tifffile failed: {e}. Falling back.")
        
        if PIL_AVAILABLE:
            try:
                file_data = filepath_or_buffer.read()
                return self._load_as_image_fallback(file_data)
            except Exception as e:
                raise RuntimeError(f"Failed to load image with all available loaders: {e}")

        raise RuntimeError("No suitable image loading library is available.")

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Convert metadata values to float when possible."""
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _resolution_to_um_per_pixel(resolution_value: Any, resolution_unit: Optional[int]) -> Optional[float]:
        """
        Convert TIFF resolution tags to um/pixel.

        TIFF XResolution/YResolution are typically pixels per unit.
        ResolutionUnit: 2=inches, 3=centimeters.
        """
        if resolution_value is None:
            return None
        try:
            if isinstance(resolution_value, tuple) and len(resolution_value) == 2:
                numerator, denominator = float(resolution_value[0]), float(resolution_value[1])
                if denominator == 0:
                    return None
                px_per_unit = numerator / denominator
            else:
                px_per_unit = float(resolution_value)
            if px_per_unit <= 0:
                return None
            if resolution_unit == 2:  # inch
                return 25400.0 / px_per_unit
            if resolution_unit == 3:  # centimeter
                return 10000.0 / px_per_unit
        except Exception:
            return None
        return None

    def _extract_tiff_metadata(self, tif: "tifffile.TiffFile", series: Any) -> Dict[str, Any]:
        """
        Extract robust physical metadata from TIFF/OME-TIFF structures.
        """
        axes = str(getattr(series, 'axes', ''))
        shape = tuple(getattr(series, 'shape', ()))
        channel_count = 1
        if 'C' in axes:
            try:
                channel_count = int(shape[axes.index('C')])
            except Exception:
                channel_count = 1

        size_x = None
        size_y = None
        size_z = None
        time_increment_s = None

        # Parse OME metadata when present.
        ome_xml = getattr(tif, 'ome_metadata', None)
        if ome_xml:
            try:
                root = ET.fromstring(ome_xml)
                pixels = root.find('.//{*}Pixels')
                if pixels is not None:
                    size_x = self._safe_float(pixels.attrib.get('PhysicalSizeX'))
                    size_y = self._safe_float(pixels.attrib.get('PhysicalSizeY'))
                    size_z = self._safe_float(pixels.attrib.get('PhysicalSizeZ'))
                    time_increment_s = self._safe_float(pixels.attrib.get('TimeIncrement'))
            except Exception as e:
                warnings.warn(f"Could not parse OME metadata: {e}")

        # Parse ImageJ metadata as fallback.
        imagej_meta = getattr(tif, 'imagej_metadata', None) or {}
        if isinstance(imagej_meta, dict):
            if size_z is None:
                size_z = self._safe_float(imagej_meta.get('spacing'))
            if time_increment_s is None:
                time_increment_s = self._safe_float(imagej_meta.get('finterval'))
                if time_increment_s is None:
                    fps = self._safe_float(imagej_meta.get('fps'))
                    if fps and fps > 0:
                        time_increment_s = 1.0 / fps

        # Parse TIFF resolution tags as fallback for X/Y.
        try:
            first_page = series.pages[0]
            tags = first_page.tags
            unit = tags.get('ResolutionUnit')
            unit_val = int(unit.value) if unit is not None else None
            if size_x is None:
                x_res_tag = tags.get('XResolution')
                x_res = x_res_tag.value if x_res_tag is not None else None
                size_x = self._resolution_to_um_per_pixel(x_res, unit_val)
            if size_y is None:
                y_res_tag = tags.get('YResolution')
                y_res = y_res_tag.value if y_res_tag is not None else None
                size_y = self._resolution_to_um_per_pixel(y_res, unit_val)
        except Exception:
            pass

        voxel_size = (
            float(size_z) if size_z and size_z > 0 else 1.0,
            float(size_y) if size_y and size_y > 0 else 0.5,
            float(size_x) if size_x and size_x > 0 else 0.5,
        )

        return {
            "axes": axes,
            "shape": shape,
            "channel_count": int(max(channel_count, 1)),
            "voxel_size": voxel_size,
            "frame_interval_s": float(time_increment_s) if time_increment_s and time_increment_s > 0 else None,
            "imagej_metadata": imagej_meta if isinstance(imagej_meta, dict) else None,
            "ome_metadata_available": bool(ome_xml),
        }

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
            time_increment = parser.image(0).Pixels.TimeIncrement
            voxel_size = (
                float(size_z) if size_z is not None else 1.0,
                float(size_y) if size_y is not None else 0.5,
                float(size_x) if size_x is not None else 0.5,
            )

            return {
                "image_data": image_data,
                "voxel_size": voxel_size,
                "frame_interval_s": float(time_increment) if time_increment is not None else None,
                "channel_count": int(parser.image(0).Pixels.SizeC),
                "metadata": metadata,
                "status": "success",
                "loader": "Bio-Formats"
            }
        finally:
            os.remove(path)

    def _load_with_tifffile(self, filepath_or_buffer, channel: int = 0) -> dict:
        """
        Load TIFF/OME-TIFF and extract available physical metadata.
        """
        filepath_or_buffer.seek(0)
        with tifffile.TiffFile(filepath_or_buffer) as tif:
            series = tif.series[0]
            full_data = series.asarray()
            metadata = self._extract_tiff_metadata(tif, series)

            image_data = np.asarray(full_data)
            axes = metadata.get("axes", "")
            if 'C' in axes:
                c_axis = axes.index('C')
                c_count = image_data.shape[c_axis]
                if c_count <= 0:
                    raise ValueError("Invalid channel axis in TIFF data.")

                # Respect selected channel by slicing on C axis.
                channel = max(0, min(channel, c_count - 1))
                image_data = np.take(image_data, indices=channel, axis=c_axis)

        filepath_or_buffer.seek(0)
        return {
            "image_data": image_data,
            "voxel_size": metadata["voxel_size"],
            "frame_interval_s": metadata.get("frame_interval_s"),
            "channel_count": metadata.get("channel_count", 1),
            "metadata": metadata,
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
