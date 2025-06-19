"""
Data Loader Module
Handles loading of multi-format microscopy data with standardized output
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import io
import warnings
from typing import Dict, Any, Optional, Union, List

# Import microscopy file readers
try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False
    warnings.warn("tifffile not available - TIFF support limited")

try:
    import pims
    PIMS_AVAILABLE = True
except ImportError:
    PIMS_AVAILABLE = False
    warnings.warn("PIMS not available - image sequence support limited")

try:
    import readlif
    READLIF_AVAILABLE = True
except ImportError:
    READLIF_AVAILABLE = False
    warnings.warn("readlif not available - Leica LIF support disabled")

try:
    import pylibczirw as czi
    PYLIBCZIRW_AVAILABLE = True
except ImportError:
    PYLIBCZIRW_AVAILABLE = False
    warnings.warn("pylibCZIrw not available - Zeiss CZI support disabled")

try:
    import fcsfiles
    FCSFILES_AVAILABLE = True
except ImportError:
    FCSFILES_AVAILABLE = False
    warnings.warn("fcsfiles not available - Zeiss Confocor FCS support disabled")

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    warnings.warn("h5py not available - HDF5 support disabled")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    warnings.warn("PIL not available - basic image support disabled")

# Supported file formats and their extensions
SUPPORTED_FORMATS = {
    'tif': 'TIFF (MetaMorph, Leica, Olympus)',
    'tiff': 'TIFF (MetaMorph, Leica, Olympus)', 
    'tiff_sequence': 'TIFF Sequence (Multiple Files)',
    'stk': 'MetaMorph Stack',
    'lsm': 'Zeiss LSM',
    'czi': 'Zeiss CZI (LSM 700, Elyra 7)',
    'lif': 'Leica LIF',
    'oif': 'Olympus OIF',
    'oib': 'Olympus OIB',
    'nd2': 'Nikon ND2',
    'lms': 'Leica LMS'
}

class DataLoader:
    """Main data loader class for handling multiple microscopy formats"""
    
    def __init__(self):
        self.supported_formats = SUPPORTED_FORMATS
        self.current_data = None
        
    def load_file(self, uploaded_file) -> Dict[str, Any]:
        """
        Load microscopy data from uploaded file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Dict containing standardized data information
        """
        
        file_extension = Path(uploaded_file.name).suffix.lower().lstrip('.')
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Read file data
        file_data = uploaded_file.getvalue()
        
        # Determine loader based on extension
        if file_extension in ['tif', 'tiff']:
            return self._load_tiff(file_data, uploaded_file.name)
        elif file_extension == 'stk':
            return self._load_metamorph_stack(file_data, uploaded_file.name)
        elif file_extension == 'lsm':
            return self._load_zeiss_lsm(file_data, uploaded_file.name)
        elif file_extension == 'czi':
            return self._load_zeiss_czi(file_data, uploaded_file.name)
        elif file_extension == 'lif':
            return self._load_leica_lif(file_data, uploaded_file.name)
        elif file_extension in ['oif', 'oib']:
            return self._load_olympus(file_data, uploaded_file.name)
        else:
            raise ValueError(f"Loader not implemented for {file_extension}")
    
    def load_tiff_sequence(self, uploaded_files: List) -> Dict[str, Any]:
        """
        Load a sequence of TIFF files as a time series
        
        Args:
            uploaded_files: List of Streamlit uploaded file objects
            
        Returns:
            Dict containing standardized data information for the sequence
        """
        
        if not TIFFFILE_AVAILABLE:
            raise ImportError("tifffile required for TIFF sequence support")
        
        if not uploaded_files:
            raise ValueError("No files provided for TIFF sequence")
        
        # Sort files by name to ensure proper time ordering
        sorted_files = sorted(uploaded_files, key=lambda f: f.name)
        
        image_stack = []
        metadata_list = []
        
        # Load each TIFF file
        for i, uploaded_file in enumerate(sorted_files):
            file_extension = Path(uploaded_file.name).suffix.lower().lstrip('.')
            
            if file_extension not in ['tif', 'tiff']:
                raise ValueError(f"All files must be TIFF format. Found: {file_extension} in {uploaded_file.name}")
            
            try:
                file_data = uploaded_file.getvalue()
                
                with io.BytesIO(file_data) as f:
                    with tifffile.TiffFile(f) as tif:
                        # Read image data
                        image_data = tif.asarray()
                        
                        # Ensure 2D images for sequence
                        if image_data.ndim > 2:
                            # If it's already a stack, take the first frame or average
                            if image_data.ndim == 3:
                                image_data = image_data[0] if image_data.shape[0] == 1 else np.mean(image_data, axis=0)
                            elif image_data.ndim == 4:
                                image_data = image_data[0, 0] if image_data.shape[0] == 1 and image_data.shape[1] == 1 else np.mean(image_data, axis=(0, 1))
                        
                        image_stack.append(image_data)
                        
                        # Extract metadata from first file
                        if i == 0:
                            base_metadata = {}
                            if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata:
                                if isinstance(tif.imagej_metadata, dict):
                                    base_metadata.update(tif.imagej_metadata)
                            
                            pixel_size = self._extract_pixel_size(tif)
                            
                        # Store individual file metadata
                        file_metadata = {
                            'filename': uploaded_file.name,
                            'file_index': i,
                            'file_size': len(file_data)
                        }
                        metadata_list.append(file_metadata)
                        
            except Exception as e:
                raise RuntimeError(f"Failed to load TIFF file {uploaded_file.name}: {str(e)}")
        
        # Stack images into 3D array (time, height, width)
        try:
            image_stack = np.stack(image_stack, axis=0)
        except ValueError as e:
            raise ValueError(f"Images have inconsistent dimensions. All images must be the same size: {str(e)}")
        
        # Create sequence metadata
        sequence_metadata = {
            'sequence_info': {
                'num_files': len(sorted_files),
                'file_names': [f.name for f in sorted_files],
                'sequence_type': 'TIFF_SEQUENCE'
            },
            'file_metadata': metadata_list
        }
        
        if 'base_metadata' in locals():
            sequence_metadata.update(base_metadata)
        
        # Estimate time interval (assume equal spacing)
        time_interval = 1.0  # Default 1 second between frames
        
        return self._standardize_data_format(
            image_data=image_stack,
            filename=f"TIFF_Sequence_{len(sorted_files)}_files",
            format_type='TIFF Sequence',
            metadata=sequence_metadata,
            pixel_size=pixel_size if 'pixel_size' in locals() else None,
            time_interval=time_interval
        )
    
    def _load_tiff(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Load TIFF files (MetaMorph, Leica, Olympus)"""
        
        if not TIFFFILE_AVAILABLE:
            raise ImportError("tifffile required for TIFF support")
        
        try:
            with io.BytesIO(file_data) as f:
                with tifffile.TiffFile(f) as tif:
                    # Read image data
                    image_data = tif.asarray()
                    
                    # Extract metadata
                    metadata = {}
                    if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata:
                        if isinstance(tif.imagej_metadata, dict):
                            metadata.update(tif.imagej_metadata)
                    
                    # Try to extract pixel size and timing information
                    pixel_size = self._extract_pixel_size(tif)
                    time_interval = self._extract_time_interval(tif)
                    
                    return self._standardize_data_format(
                        image_data=image_data,
                        filename=filename,
                        format_type='TIFF',
                        metadata=metadata,
                        pixel_size=pixel_size,
                        time_interval=time_interval
                    )
                    
        except Exception as e:
            raise RuntimeError(f"Failed to load TIFF file: {str(e)}")
    
    def _load_metamorph_stack(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Load MetaMorph stack files"""
        
        try:
            # MetaMorph stacks are essentially TIFF files with specific metadata
            return self._load_tiff(file_data, filename)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load MetaMorph stack: {str(e)}")
    
    def _load_zeiss_lsm(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Load Zeiss LSM files"""
        
        if not TIFFFILE_AVAILABLE:
            raise ImportError("tifffile required for LSM support")
        
        try:
            with io.BytesIO(file_data) as f:
                with tifffile.TiffFile(f) as tif:
                    # LSM files have specific metadata structure
                    image_data = tif.asarray()
                    
                    # Extract LSM-specific metadata
                    metadata = {}
                    if hasattr(tif, 'lsm_metadata') and tif.lsm_metadata:
                        try:
                            if isinstance(tif.lsm_metadata, dict):
                                metadata.update(tif.lsm_metadata)
                        except (TypeError, AttributeError):
                            pass
                    
                    pixel_size = self._extract_lsm_pixel_size(tif)
                    time_interval = self._extract_lsm_time_interval(tif)
                    
                    return self._standardize_data_format(
                        image_data=image_data,
                        filename=filename,
                        format_type='Zeiss LSM',
                        metadata=metadata,
                        pixel_size=pixel_size,
                        time_interval=time_interval
                    )
                    
        except Exception as e:
            raise RuntimeError(f"Failed to load Zeiss LSM file: {str(e)}")
    
    def _load_zeiss_czi(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Load Zeiss CZI files (LSM 700, Elyra 7) using pylibczirw library"""
        
        if not PYLIBCZIRW_AVAILABLE:
            warnings.warn("pylibczirw not available - using fallback image loading for CZI")
            try:
                image_data = self._load_as_image_fallback(file_data)
                return self._standardize_data_format(
                    image_data=image_data,
                    filename=filename,
                    format_type='Zeiss CZI (Fallback)',
                    metadata={'note': 'CZI metadata extraction not available - loaded via fallback'},
                    pixel_size=0.1,
                    time_interval=0.1
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load CZI file via fallback: {str(e)}")

        try:
            import pylibczirw as czi
            
            with io.BytesIO(file_data) as f:
                czi_file = czi.CziFile(f)

                # Use the 'asarray()' method to load the image data
                image_data = czi_file.asarray()

                # Extract basic metadata
                metadata = {
                    'czi_filename': filename,
                    'dims_shape': str(czi_file.dims_shape),
                    'size_c': czi_file.size_c,
                    'size_t': czi_file.size_t,
                    'size_z': czi_file.size_z,
                    'is_tiled': czi_file.is_tiled,
                    'metadata_xml_snippet': czi_file.metadata[:500]
                }

                # Extract pixel size and time interval from CZI metadata
                pixel_size = self._extract_czi_pixel_size(czi_file)
                time_interval = self._extract_czi_time_interval(czi_file)

                return self._standardize_data_format(
                    image_data=image_data,
                    filename=filename,
                    format_type='Zeiss CZI',
                    metadata=metadata,
                    pixel_size=pixel_size,
                    time_interval=time_interval
                )

        except Exception as e:
            warnings.warn(f"pylibczirw failed to load CZI file: {str(e)}. Using fallback.")
            try:
                image_data = self._load_as_image_fallback(file_data)
                return self._standardize_data_format(
                    image_data=image_data,
                    filename=filename,
                    format_type='Zeiss CZI (Fallback)',
                    metadata={'note': f'CZI library failed: {str(e)}, loaded via fallback'},
                    pixel_size=0.1,
                    time_interval=0.1
                )
            except Exception as e_fallback:
                raise RuntimeError(f"Failed to load CZI file even with fallback: {str(e_fallback)}")
    
    def _load_leica_lif(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Load Leica LIF files using readlif library"""
        
        if not READLIF_AVAILABLE:
            raise ImportError("readlif library required for Leica LIF support")
        
        try:
            # Save bytes to temporary file for readlif
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.lif', delete=False) as tmp_file:
                tmp_file.write(file_data)
                tmp_file_path = tmp_file.name
            
            try:
                # Load LIF file using readlif
                lif_file = readlif.LifFile(tmp_file_path)
                
                # Get the first image series
                if len(lif_file.image_list) == 0:
                    raise ValueError("No images found in LIF file")
                
                # Use first image for now - could be extended to handle multiple series
                image_series = lif_file.get_image(0)
                
                # Extract image data
                image_data = np.array(list(image_series.get_frame(0)))
                
                # Get metadata
                metadata = {
                    'name': image_series.name,
                    'dims': image_series.dims,
                    'channels': image_series.channels,
                    'scale': image_series.scale,
                    'timestamps': getattr(image_series, 'timestamps', None)
                }
                
                # Extract pixel size and time interval
                pixel_size = image_series.scale[0] if image_series.scale else 0.1  # First scale value (usually X)
                time_interval = 0.1  # Default, would need to calculate from timestamps
                
                if hasattr(image_series, 'timestamps') and image_series.timestamps:
                    if len(image_series.timestamps) > 1:
                        time_interval = image_series.timestamps[1] - image_series.timestamps[0]
                
                return self._standardize_data_format(
                    image_data=image_data,
                    filename=filename,
                    format_type='Leica LIF',
                    metadata=metadata,
                    pixel_size=pixel_size,
                    time_interval=time_interval
                )
                
            finally:
                # Clean up temporary file
                import os
                try:
                    os.unlink(tmp_file_path)
                except OSError:
                    pass
                    
        except Exception as e:
            raise RuntimeError(f"Failed to load Leica LIF file: {str(e)}")
    
    def _load_olympus(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Load Olympus OIF/OIB files"""
        
        try:
            # Olympus files - placeholder implementation
            image_data = self._load_as_image_fallback(file_data)
            
            return self._standardize_data_format(
                image_data=image_data,
                filename=filename,
                format_type='Olympus',
                metadata={'note': 'Olympus metadata extraction not fully implemented'},
                pixel_size=0.1,
                time_interval=0.1
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Olympus file: {str(e)}")
    
    def _load_as_image_fallback(self, file_data: bytes) -> np.ndarray:
        """Fallback image loading using PIL"""
        
        if not PIL_AVAILABLE:
            raise ImportError("PIL required for basic image support")
        
        try:
            with io.BytesIO(file_data) as f:
                img = Image.open(f)
                return np.array(img)
        except Exception as e:
            raise RuntimeError(f"Failed to load as image: {str(e)}")
    
    def _extract_pixel_size(self, tif) -> float:
        """Extract pixel size from TIFF metadata"""
        try:
            # Try various metadata sources
            if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata:
                if 'spacing' in tif.imagej_metadata:
                    return float(tif.imagej_metadata['spacing'])
            
            # Try resolution tags
            if hasattr(tif, 'pages') and tif.pages:
                page = tif.pages[0]
                if hasattr(page, 'tags'):
                    x_res = page.tags.get('XResolution')
                    if x_res and x_res.value:
                        return 1.0 / float(x_res.value[0] / x_res.value[1])
            
            return 0.1  # Default pixel size in micrometers
            
        except Exception:
            return 0.1
    
    def _extract_time_interval(self, tif) -> float:
        """Extract time interval from TIFF metadata"""
        try:
            if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata:
                if 'finterval' in tif.imagej_metadata:
                    return float(tif.imagej_metadata['finterval'])
                if 'fps' in tif.imagej_metadata:
                    return 1.0 / float(tif.imagej_metadata['fps'])
            
            return 0.1  # Default time interval in seconds
            
        except Exception:
            return 0.1
    
    def _extract_lsm_pixel_size(self, tif) -> float:
        """Extract pixel size from LSM metadata"""
        try:
            if hasattr(tif, 'lsm_metadata') and tif.lsm_metadata:
                voxel_sizes = tif.lsm_metadata.get('VoxelSizes')
                if voxel_sizes:
                    return float(voxel_sizes[0]) * 1e6  # Convert to micrometers
            
            return 0.1
            
        except Exception:
            return 0.1
    
    def _extract_lsm_time_interval(self, tif) -> float:
        """Extract time interval from LSM metadata"""
        try:
            if hasattr(tif, 'lsm_metadata') and tif.lsm_metadata:
                time_interval = tif.lsm_metadata.get('TimeInterval')
                if time_interval:
                    return float(time_interval)
            
            return 0.1
            
        except Exception:
            return 0.1

    def _extract_czi_pixel_size(self, czi_file) -> float:
        """Extract pixel size from CZI metadata (simplified)"""
        try:
            # Attempt to find scaling information in the XML metadata
            scaling_elements = czi_file.metadata_tree.findall(".//Scaling/Items/Distance")
            for item in scaling_elements:
                if item.get("Id") == "X" or item.get("Id") == "Y":
                    value = float(item.find("Value").text)
                    unit = item.find("Unit").text
                    if unit == "m":
                        return value * 1e6  # Convert meters to micrometers
                    elif unit == "µm":
                        return value
            return 0.1  # Default if not found
        except Exception:
            return 0.1  # Fallback default

    def _extract_czi_time_interval(self, czi_file) -> float:
        """Extract time interval from CZI metadata (simplified)"""
        try:
            # Attempt to find time interval in XML metadata
            return 0.1  # Default if not found
        except Exception:
            return 0.1  # Fallback default
    
    def _standardize_data_format(self, 
                               image_data: np.ndarray, 
                               filename: str,
                               format_type: str,
                               metadata: Dict,
                               pixel_size: float,
                               time_interval: float) -> Dict[str, Any]:
        """
        Standardize data format for cross-compatibility
        
        Returns:
            Standardized data dictionary
        """
        
        # Ensure image data is in standard format (T, Y, X, C) or (Y, X, C) or (Y, X)
        standardized_data = self._standardize_image_dimensions(image_data)
        
        # Extract basic information
        shape = standardized_data.shape
        dtype = str(standardized_data.dtype)
        
        # Determine number of time points and channels with better logic for multichannel data
        if len(shape) == 4:  # T, Y, X, C
            time_points, height, width, channels = shape
        elif len(shape) == 3:
            # More sophisticated logic for 3D data
            # Check if last dimension could be channels (1-10 channels common)
            if shape[2] <= 10 and shape[0] > 10:  # Likely T, Y, X with channels as last dim would be small
                time_points, height, width = shape
                channels = 1
            elif shape[2] <= 10:  # Likely Y, X, C
                height, width, channels = shape
                time_points = 1
            else:  # Likely T, Y, X (time series)
                time_points, height, width = shape
                channels = 1
        else:  # 2D
            height, width = shape
            time_points = 1
            channels = 1
        
        # Extract channel information from metadata if available
        channel_names = self._extract_channel_names(metadata, channels)
        channel_colors = self._extract_channel_colors(metadata, channels)
        
        # Create standardized data structure
        data_info = {
            'filename': filename,
            'format': format_type,
            'image_data': standardized_data,
            'shape': shape,
            'dtype': dtype,
            'pixel_size': pixel_size,
            'time_interval': time_interval,
            'time_points': time_points,
            'channels': channels,
            'height': height,
            'width': width,
            'channel_names': channel_names,
            'channel_colors': channel_colors,
            'metadata': metadata,
            'acquisition_params': self._extract_acquisition_parameters(metadata)
        }
        
        return data_info
    
    def _extract_channel_names(self, metadata: Dict, num_channels: int) -> List[str]:
        """Extract channel names from metadata"""
        
        channel_names = []
        
        # Try to extract from common metadata fields
        if 'channels' in metadata and isinstance(metadata['channels'], list):
            channel_names = metadata['channels'][:num_channels]
        elif 'channel_names' in metadata and isinstance(metadata['channel_names'], list):
            channel_names = metadata['channel_names'][:num_channels]
        elif 'ChannelNames' in metadata and isinstance(metadata['ChannelNames'], list):
            channel_names = metadata['ChannelNames'][:num_channels]
        
        # Fill in default names if not enough channels found
        while len(channel_names) < num_channels:
            channel_names.append(f"Channel {len(channel_names) + 1}")
        
        return channel_names[:num_channels]
    
    def _extract_channel_colors(self, metadata: Dict, num_channels: int) -> List[str]:
        """Extract channel colors from metadata or assign defaults"""
        
        # Default color scheme for up to 10 channels
        default_colors = [
            '#FF0000',  # Red
            '#00FF00',  # Green  
            '#0000FF',  # Blue
            '#FFFF00',  # Yellow
            '#FF00FF',  # Magenta
            '#00FFFF',  # Cyan
            '#FFA500',  # Orange
            '#800080',  # Purple
            '#FFC0CB',  # Pink
            '#A52A2A'   # Brown
        ]
        
        channel_colors = []
        
        # Try to extract from metadata
        if 'channel_colors' in metadata and isinstance(metadata['channel_colors'], list):
            channel_colors = metadata['channel_colors'][:num_channels]
        elif 'ChannelColors' in metadata and isinstance(metadata['ChannelColors'], list):
            channel_colors = metadata['ChannelColors'][:num_channels]
        
        # Fill with defaults if needed
        while len(channel_colors) < num_channels:
            color_idx = len(channel_colors) % len(default_colors)
            channel_colors.append(default_colors[color_idx])
        
        return channel_colors[:num_channels]
    
    def _standardize_image_dimensions(self, image_data: np.ndarray) -> np.ndarray:
        """Standardize image dimensions to consistent format"""
        
        # Ensure data is at least 2D
        if image_data.ndim < 2:
            raise ValueError("Image data must be at least 2D")
        
        # Handle different dimension arrangements
        if image_data.ndim == 2:
            # 2D image - keep as is
            return image_data
        elif image_data.ndim == 3:
            # Could be T,Y,X or Y,X,C - try to determine
            if image_data.shape[2] <= 4:
                # Likely Y,X,C (channels)
                return image_data
            else:
                # Likely T,Y,X (time series)
                return image_data
        elif image_data.ndim == 4:
            # Likely T,Y,X,C or T,C,Y,X - standardize to T,Y,X,C
            return image_data
        else:
            # Higher dimensions - take first 4
            return image_data[:, :, :, :] if image_data.ndim > 4 else image_data
    
    def _extract_acquisition_parameters(self, metadata: Dict) -> Dict[str, Any]:
        """Extract relevant acquisition parameters from metadata"""
        
        acq_params = {}
        
        # Common parameter mappings
        param_mappings = {
            'exposure_time': ['ExposureTime', 'exposure', 'ExpTime'],
            'gain': ['Gain', 'gain'],
            'laser_power': ['LaserPower', 'laser_power', 'Power'],
            'objective': ['Objective', 'objective', 'ObjectiveName'],
            'na': ['NA', 'NumericalAperture', 'numerical_aperture'],
            'excitation': ['Excitation', 'excitation', 'ExcitationWavelength'],
            'emission': ['Emission', 'emission', 'EmissionWavelength']
        }
        
        for param_name, possible_keys in param_mappings.items():
            for key in possible_keys:
                if key in metadata:
                    acq_params[param_name] = metadata[key]
                    break
        
        return acq_params
    
    def get_supported_formats(self) -> Dict[str, str]:
        """Return dictionary of supported formats"""
        return self.supported_formats.copy()
    
    def validate_file_format(self, filename: str) -> bool:
        """Check if file format is supported"""
        extension = Path(filename).suffix.lower().lstrip('.')
        return extension in self.supported_formats
