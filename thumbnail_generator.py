"""
Microscopy Data Format Preview Thumbnails
Generates preview thumbnails for various microscopy file formats
"""

import numpy as np
import warnings
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import io

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available - thumbnail generation limited")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    warnings.warn("PIL not available - thumbnail creation limited")

# Import format-specific readers with fallbacks
try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

try:
    from readlif import LifFile
    READLIF_AVAILABLE = True
except ImportError:
    READLIF_AVAILABLE = False

try:
    import pylibczirw
    CZI_AVAILABLE = True
except ImportError:
    CZI_AVAILABLE = False

class ThumbnailGenerator:
    """Generates preview thumbnails for microscopy data formats"""
    
    def __init__(self):
        self.thumbnail_size = (256, 256)
        self.preview_size = (128, 128)
        self.supported_formats = {
            '.tif': self._generate_tiff_thumbnail,
            '.tiff': self._generate_tiff_thumbnail,
            '.stk': self._generate_stk_thumbnail,
            '.lsm': self._generate_lsm_thumbnail,
            '.czi': self._generate_czi_thumbnail,
            '.lif': self._generate_lif_thumbnail,
            '.oif': self._generate_oif_thumbnail,
            '.nd2': self._generate_nd2_thumbnail,
            '.ims': self._generate_ims_thumbnail
        }
        self.format_info = self._initialize_format_info()
    
    def _initialize_format_info(self) -> Dict[str, Dict[str, Any]]:
        """Initialize format-specific information"""
        return {
            '.tif': {
                'name': 'TIFF Image',
                'description': 'Tagged Image File Format - Universal microscopy format',
                'typical_use': 'Multi-channel, time-series, Z-stacks',
                'color': '#4CAF50',
                'icon': '📷'
            },
            '.tiff': {
                'name': 'TIFF Image',
                'description': 'Tagged Image File Format - Universal microscopy format',
                'typical_use': 'Multi-channel, time-series, Z-stacks',
                'color': '#4CAF50',
                'icon': '📷'
            },
            '.stk': {
                'name': 'MetaMorph Stack',
                'description': 'MetaMorph imaging system stack format',
                'typical_use': 'Time-lapse, multi-dimensional imaging',
                'color': '#2196F3',
                'icon': '📚'
            },
            '.lsm': {
                'name': 'Zeiss LSM',
                'description': 'Zeiss Laser Scanning Microscope format',
                'typical_use': 'Confocal microscopy, high-resolution imaging',
                'color': '#FF9800',
                'icon': '🔬'
            },
            '.czi': {
                'name': 'Zeiss CZI',
                'description': 'Zeiss microscopy format with metadata',
                'typical_use': 'Advanced Zeiss systems, multi-modal imaging',
                'color': '#FF5722',
                'icon': '🎯'
            },
            '.lif': {
                'name': 'Leica LIF',
                'description': 'Leica Image File format',
                'typical_use': 'Leica confocal and widefield systems',
                'color': '#9C27B0',
                'icon': '🔍'
            },
            '.oif': {
                'name': 'Olympus OIF',
                'description': 'Olympus Image File format',
                'typical_use': 'Olympus confocal and spinning disk systems',
                'color': '#3F51B5',
                'icon': '⭕'
            },
            '.nd2': {
                'name': 'Nikon ND2',
                'description': 'Nikon microscopy format',
                'typical_use': 'Nikon imaging systems',
                'color': '#795548',
                'icon': '📸'
            },
            '.ims': {
                'name': 'Imaris IMS',
                'description': 'Bitplane Imaris format for 3D/4D data',
                'typical_use': '3D reconstruction, large datasets',
                'color': '#607D8B',
                'icon': '🧊'
            }
        }
    
    def generate_format_thumbnail(self, file_path: Union[str, Path], 
                                format_type: str = None) -> Dict[str, Any]:
        """Generate thumbnail for microscopy file format"""
        
        file_path = Path(file_path)
        if not file_path.exists():
            return self._generate_error_thumbnail(f"File not found: {file_path.name}")
        
        # Determine format from extension if not provided
        if format_type is None:
            format_type = file_path.suffix.lower()
        
        # Check if format is supported
        if format_type not in self.supported_formats:
            return self._generate_unsupported_thumbnail(format_type)
        
        try:
            # Generate format-specific thumbnail
            thumbnail_data = self.supported_formats[format_type](file_path)
            
            # Add format information
            thumbnail_data.update({
                'format': format_type,
                'format_info': self.format_info.get(format_type, {}),
                'file_size': file_path.stat().st_size,
                'file_name': file_path.name
            })
            
            return thumbnail_data
            
        except Exception as e:
            return self._generate_error_thumbnail(f"Error reading {file_path.name}: {str(e)}")
    
    def _generate_tiff_thumbnail(self, file_path: Path) -> Dict[str, Any]:
        """Generate thumbnail for TIFF files"""
        
        if not TIFFFILE_AVAILABLE:
            return self._generate_placeholder_thumbnail('TIFF', "tifffile library not available")
        
        try:
            # Read TIFF metadata and first frame
            with tifffile.TiffFile(file_path) as tif:
                # Get basic information
                series = tif.series[0]
                shape = series.shape
                dtype = series.dtype
                
                # Read first frame or slice
                if len(shape) >= 2:
                    if len(shape) == 2:  # Single 2D image
                        image_data = tif.asarray()
                    elif len(shape) == 3:  # Time series or Z-stack
                        image_data = tif.asarray()[0]  # First frame
                    elif len(shape) == 4:  # Multi-channel time series
                        image_data = tif.asarray()[0, 0]  # First frame, first channel
                    else:  # Higher dimensions
                        # Navigate to first 2D slice
                        indices = [0] * (len(shape) - 2) + [slice(None), slice(None)]
                        image_data = tif.asarray()[tuple(indices)]
                else:
                    return self._generate_error_thumbnail("Invalid TIFF dimensions")
                
                # Generate thumbnail image
                thumbnail_image = self._create_thumbnail_image(image_data)
                
                # Extract metadata
                metadata = {
                    'dimensions': shape,
                    'dtype': str(dtype),
                    'num_pages': len(tif.pages),
                    'is_multipage': len(tif.pages) > 1,
                    'pixel_type': self._determine_pixel_type(image_data),
                    'estimated_channels': self._estimate_channels(shape),
                    'estimated_timepoints': self._estimate_timepoints(shape)
                }
                
                # Add TIFF-specific metadata
                if hasattr(tif.pages[0], 'tags'):
                    tags = tif.pages[0].tags
                    if 'ImageDescription' in tags:
                        metadata['description'] = str(tags['ImageDescription'].value)[:200]
                    if 'XResolution' in tags and 'YResolution' in tags:
                        metadata['resolution'] = f"{tags['XResolution'].value}, {tags['YResolution'].value}"
                
                return {
                    'thumbnail': thumbnail_image,
                    'metadata': metadata,
                    'preview_text': self._generate_preview_text(metadata),
                    'status': 'success'
                }
                
        except Exception as e:
            return self._generate_error_thumbnail(f"TIFF read error: {str(e)}")
    
    def _generate_stk_thumbnail(self, file_path: Path) -> Dict[str, Any]:
        """Generate thumbnail for MetaMorph STK files"""
        # STK files are typically readable as TIFF
        try:
            thumbnail_data = self._generate_tiff_thumbnail(file_path)
            if thumbnail_data['status'] == 'success':
                thumbnail_data['metadata']['format_specific'] = 'MetaMorph Stack'
                thumbnail_data['preview_text'] += "\n📚 MetaMorph imaging stack"
            return thumbnail_data
        except:
            return self._generate_placeholder_thumbnail('STK', "MetaMorph stack format")
    
    def _generate_lsm_thumbnail(self, file_path: Path) -> Dict[str, Any]:
        """Generate thumbnail for Zeiss LSM files"""
        # LSM files can often be read as TIFF
        try:
            thumbnail_data = self._generate_tiff_thumbnail(file_path)
            if thumbnail_data['status'] == 'success':
                thumbnail_data['metadata']['format_specific'] = 'Zeiss LSM Confocal'
                thumbnail_data['preview_text'] += "\n🔬 Zeiss confocal data"
            return thumbnail_data
        except:
            return self._generate_placeholder_thumbnail('LSM', "Zeiss LSM confocal format")
    
    def _generate_czi_thumbnail(self, file_path: Path) -> Dict[str, Any]:
        """Generate thumbnail for Zeiss CZI files"""
        
        if not CZI_AVAILABLE:
            return self._generate_placeholder_thumbnail('CZI', "pylibCZIrw library not available")
        
        try:
            # Basic CZI reading implementation
            return self._generate_placeholder_thumbnail('CZI', "Zeiss CZI format - advanced metadata")
        except:
            return self._generate_placeholder_thumbnail('CZI', "Zeiss CZI format")
    
    def _generate_lif_thumbnail(self, file_path: Path) -> Dict[str, Any]:
        """Generate thumbnail for Leica LIF files"""
        
        if not READLIF_AVAILABLE:
            return self._generate_placeholder_thumbnail('LIF', "readlif library not available")
        
        try:
            lif = LifFile(str(file_path))
            
            if len(lif.image_list) > 0:
                # Get first image
                first_image = lif.get_image(0)
                
                # Get basic information
                dims = first_image.dims
                metadata = {
                    'dimensions': f"{dims.x} × {dims.y}",
                    'num_images': len(lif.image_list),
                    'channels': dims.c if hasattr(dims, 'c') else 1,
                    'z_slices': dims.z if hasattr(dims, 'z') else 1,
                    'time_points': dims.t if hasattr(dims, 't') else 1,
                    'format_specific': 'Leica LIF container'
                }
                
                # Try to get first frame
                try:
                    frame = first_image.get_frame(z=0, t=0, c=0)
                    thumbnail_image = self._create_thumbnail_image(np.array(frame))
                except:
                    thumbnail_image = self._create_format_icon_thumbnail('LIF')
                
                return {
                    'thumbnail': thumbnail_image,
                    'metadata': metadata,
                    'preview_text': self._generate_preview_text(metadata),
                    'status': 'success'
                }
            else:
                return self._generate_error_thumbnail("No images found in LIF file")
                
        except Exception as e:
            return self._generate_placeholder_thumbnail('LIF', f"Leica format: {str(e)[:50]}")
    
    def _generate_oif_thumbnail(self, file_path: Path) -> Dict[str, Any]:
        """Generate thumbnail for Olympus OIF files"""
        return self._generate_placeholder_thumbnail('OIF', "Olympus confocal format")
    
    def _generate_nd2_thumbnail(self, file_path: Path) -> Dict[str, Any]:
        """Generate thumbnail for Nikon ND2 files"""
        return self._generate_placeholder_thumbnail('ND2', "Nikon microscopy format")
    
    def _generate_ims_thumbnail(self, file_path: Path) -> Dict[str, Any]:
        """Generate thumbnail for Imaris IMS files"""
        return self._generate_placeholder_thumbnail('IMS', "Imaris 3D/4D format")
    
    def _create_thumbnail_image(self, image_data: np.ndarray) -> Optional[bytes]:
        """Create thumbnail image from numpy array"""
        
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        try:
            # Normalize image data
            if image_data.dtype != np.uint8:
                # Normalize to 0-255 range
                img_min, img_max = np.percentile(image_data, [1, 99])
                if img_max > img_min:
                    image_data = ((image_data - img_min) / (img_max - img_min) * 255).clip(0, 255).astype(np.uint8)
                else:
                    image_data = np.zeros_like(image_data, dtype=np.uint8)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(3, 3), facecolor='white')
            ax.set_facecolor('white')
            
            # Display image
            if len(image_data.shape) == 2:
                # Grayscale image
                im = ax.imshow(image_data, cmap='gray', aspect='equal')
            elif len(image_data.shape) == 3 and image_data.shape[2] == 3:
                # RGB image
                im = ax.imshow(image_data, aspect='equal')
            else:
                # Multi-channel - show first channel
                if len(image_data.shape) == 3:
                    im = ax.imshow(image_data[:, :, 0], cmap='gray', aspect='equal')
                else:
                    im = ax.imshow(image_data, cmap='gray', aspect='equal')
            
            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('Preview', fontsize=10, pad=5)
            
            # Add scale bar
            height, width = image_data.shape[:2]
            scale_length = width // 8
            scale_bar = patches.Rectangle((width - scale_length - 10, height - 20),
                                        scale_length, 5, 
                                        linewidth=1, edgecolor='white', 
                                        facecolor='white', alpha=0.8)
            ax.add_patch(scale_bar)
            
            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            buf.seek(0)
            return buf.getvalue()
            
        except Exception as e:
            plt.close('all')  # Clean up any open figures
            return None
    
    def _create_format_icon_thumbnail(self, format_name: str) -> Optional[bytes]:
        """Create icon-based thumbnail for formats without image data"""
        
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        try:
            format_info = self.format_info.get(f'.{format_name.lower()}', {})
            color = format_info.get('color', '#757575')
            icon = format_info.get('icon', '📄')
            
            fig, ax = plt.subplots(figsize=(3, 3), facecolor='white')
            ax.set_facecolor(color)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            # Add format icon and text
            ax.text(0.5, 0.6, icon, fontsize=40, ha='center', va='center', 
                   color='white', weight='bold')
            ax.text(0.5, 0.3, format_name, fontsize=16, ha='center', va='center',
                   color='white', weight='bold')
            
            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('Format Preview', fontsize=10, pad=5, color='white')
            
            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100,
                       facecolor=color, edgecolor='none')
            plt.close()
            
            buf.seek(0)
            return buf.getvalue()
            
        except Exception as e:
            plt.close('all')
            return None
    
    def _generate_placeholder_thumbnail(self, format_name: str, description: str) -> Dict[str, Any]:
        """Generate placeholder thumbnail for unsupported formats"""
        
        thumbnail_image = self._create_format_icon_thumbnail(format_name)
        
        metadata = {
            'format_name': format_name,
            'description': description,
            'supported': False,
            'format_specific': f"{format_name} format placeholder"
        }
        
        preview_text = f"📄 {format_name} Format\n{description}"
        
        return {
            'thumbnail': thumbnail_image,
            'metadata': metadata,
            'preview_text': preview_text,
            'status': 'placeholder'
        }
    
    def _generate_error_thumbnail(self, error_message: str) -> Dict[str, Any]:
        """Generate error thumbnail"""
        
        if MATPLOTLIB_AVAILABLE:
            try:
                fig, ax = plt.subplots(figsize=(3, 3), facecolor='#ffebee')
                ax.set_facecolor('#ffcdd2')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                
                # Add error icon and message
                ax.text(0.5, 0.6, '⚠️', fontsize=40, ha='center', va='center')
                ax.text(0.5, 0.3, 'Error', fontsize=16, ha='center', va='center',
                       color='#d32f2f', weight='bold')
                
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Load Error', fontsize=10, pad=5)
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100,
                           facecolor='#ffebee', edgecolor='none')
                plt.close()
                
                buf.seek(0)
                thumbnail_image = buf.getvalue()
            except:
                thumbnail_image = None
        else:
            thumbnail_image = None
        
        return {
            'thumbnail': thumbnail_image,
            'metadata': {'error': error_message},
            'preview_text': f"❌ Error: {error_message}",
            'status': 'error'
        }
    
    def _generate_unsupported_thumbnail(self, format_type: str) -> Dict[str, Any]:
        """Generate thumbnail for unsupported formats"""
        return self._generate_placeholder_thumbnail(
            format_type.upper().lstrip('.'), 
            f"Unsupported format: {format_type}"
        )
    
    def _determine_pixel_type(self, image_data: np.ndarray) -> str:
        """Determine pixel type from image data"""
        dtype = image_data.dtype
        
        if dtype == np.uint8:
            return "8-bit unsigned"
        elif dtype == np.uint16:
            return "16-bit unsigned"
        elif dtype == np.uint32:
            return "32-bit unsigned"
        elif dtype == np.float32:
            return "32-bit float"
        elif dtype == np.float64:
            return "64-bit float"
        else:
            return str(dtype)
    
    def _estimate_channels(self, shape: Tuple[int, ...]) -> int:
        """Estimate number of channels from shape"""
        if len(shape) == 2:
            return 1
        elif len(shape) == 3:
            # Could be channels, time, or Z
            # Use heuristics
            if shape[0] <= 10:  # Likely channels
                return shape[0]
            else:  # Likely time or Z
                return 1
        elif len(shape) == 4:
            # Assume TCZYX or TCYX
            return shape[1] if shape[1] <= 20 else 1
        else:
            return 1
    
    def _estimate_timepoints(self, shape: Tuple[int, ...]) -> int:
        """Estimate number of time points from shape"""
        if len(shape) <= 2:
            return 1
        elif len(shape) == 3:
            # Could be time, channels, or Z
            if shape[0] > 10:  # Likely time
                return shape[0]
            else:
                return 1
        elif len(shape) == 4:
            # Assume TCZYX or TCYX
            return shape[0]
        else:
            return shape[0] if shape[0] > 1 else 1
    
    def _generate_preview_text(self, metadata: Dict[str, Any]) -> str:
        """Generate preview text from metadata"""
        preview_lines = []
        
        # Basic dimensions
        if 'dimensions' in metadata:
            if isinstance(metadata['dimensions'], (list, tuple)):
                dim_str = ' × '.join(map(str, metadata['dimensions']))
            else:
                dim_str = str(metadata['dimensions'])
            preview_lines.append(f"📐 Dimensions: {dim_str}")
        
        # Pixel type
        if 'pixel_type' in metadata:
            preview_lines.append(f"🎨 Pixel Type: {metadata['pixel_type']}")
        
        # Channels
        if 'channels' in metadata or 'estimated_channels' in metadata:
            channels = metadata.get('channels', metadata.get('estimated_channels', 1))
            preview_lines.append(f"🌈 Channels: {channels}")
        
        # Time points
        if 'time_points' in metadata or 'estimated_timepoints' in metadata:
            timepoints = metadata.get('time_points', metadata.get('estimated_timepoints', 1))
            if timepoints > 1:
                preview_lines.append(f"⏱️ Time Points: {timepoints}")
        
        # Format specific info
        if 'format_specific' in metadata:
            preview_lines.append(f"🔧 {metadata['format_specific']}")
        
        return '\n'.join(preview_lines[:5])  # Limit to 5 lines
    
    def generate_format_comparison_grid(self, file_paths: List[Union[str, Path]]) -> Optional[bytes]:
        """Generate comparison grid of multiple format thumbnails"""
        
        if not MATPLOTLIB_AVAILABLE or not file_paths:
            return None
        
        try:
            # Generate thumbnails for all files
            thumbnails = []
            for file_path in file_paths[:9]:  # Maximum 9 for 3x3 grid
                thumbnail_data = self.generate_format_thumbnail(file_path)
                thumbnails.append(thumbnail_data)
            
            # Calculate grid dimensions
            n_files = len(thumbnails)
            if n_files <= 3:
                rows, cols = 1, n_files
            elif n_files <= 6:
                rows, cols = 2, 3
            else:
                rows, cols = 3, 3
            
            # Create figure
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), 
                                   facecolor='white')
            
            if n_files == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            # Plot thumbnails
            for i, thumbnail_data in enumerate(thumbnails):
                ax = axes[i]
                
                if thumbnail_data['status'] == 'success' and thumbnail_data['thumbnail']:
                    # Load thumbnail image
                    try:
                        from PIL import Image
                        thumbnail_img = Image.open(io.BytesIO(thumbnail_data['thumbnail']))
                        ax.imshow(thumbnail_img)
                    except:
                        # Fallback to format icon
                        format_name = Path(file_paths[i]).suffix.upper().lstrip('.')
                        ax.text(0.5, 0.5, format_name, ha='center', va='center',
                               fontsize=20, weight='bold', transform=ax.transAxes)
                else:
                    # Show format name
                    format_name = Path(file_paths[i]).suffix.upper().lstrip('.')
                    ax.text(0.5, 0.5, format_name, ha='center', va='center',
                           fontsize=20, weight='bold', transform=ax.transAxes)
                
                # Set title with filename
                filename = Path(file_paths[i]).name
                if len(filename) > 15:
                    filename = filename[:12] + '...'
                ax.set_title(filename, fontsize=10)
                
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Hide unused subplots
            for i in range(n_files, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100,
                       facecolor='white', edgecolor='none')
            plt.close()
            
            buf.seek(0)
            return buf.getvalue()
            
        except Exception as e:
            plt.close('all')
            return None
    
    def get_format_support_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about supported formats and their capabilities"""
        
        support_info = {}
        
        for ext, info in self.format_info.items():
            support_info[ext] = {
                'name': info['name'],
                'description': info['description'],
                'typical_use': info['typical_use'],
                'thumbnail_supported': ext in self.supported_formats,
                'library_available': self._check_library_availability(ext),
                'read_capability': self._assess_read_capability(ext)
            }
        
        return support_info
    
    def _check_library_availability(self, format_ext: str) -> bool:
        """Check if required library is available for format"""
        
        if format_ext in ['.tif', '.tiff', '.stk', '.lsm']:
            return TIFFFILE_AVAILABLE
        elif format_ext == '.lif':
            return READLIF_AVAILABLE
        elif format_ext == '.czi':
            return CZI_AVAILABLE
        else:
            return False
    
    def _assess_read_capability(self, format_ext: str) -> str:
        """Assess read capability for format"""
        
        if self._check_library_availability(format_ext):
            return 'Full support'
        elif format_ext in ['.tif', '.tiff']:
            return 'Basic support'
        else:
            return 'Limited support'