"""
Two-Domain Nuclear Elastography Module
Based on Ghosh et al. (2021) - Image-based elastography of heterochromatin 
and euchromatin domains in the deforming cell nucleus

Implements:
- Hill function-based chromatin segmentation
- Inverse finite element optimization
- Relative stiffness ratio (E_h/E_e) determination

Reference:
Ghosh, S., et al. (2021). Image-based elastography of heterochromatin and
euchromatin domains in the deforming cell nucleus.
Soft Matter, 17(22), 5543-5555.
"""

import numpy as np
from scipy import optimize, ndimage, interpolate
from scipy.special import expit  # sigmoid function
from typing import Dict, Any, Tuple, Optional, List
import warnings

try:
    from skimage.filters import gaussian, threshold_otsu
    from skimage.morphology import binary_opening, binary_closing, disk
    from skimage.measure import label, regionprops
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not available - some elastography features limited")


class TwoDomainNuclearElastography:
    """
    Two-Domain Nuclear Elastography Analysis
    
    Solves the inverse problem to quantify the relative elasticity (stiffness ratio)
    between heterochromatin and euchromatin domains in the cell nucleus.
    
    Key outputs:
    - Binary chromatin domain segmentation (H/E)
    - Optimized stiffness ratio (E_h/E_e)
    - Displacement field reconstruction quality metrics
    """
    
    def __init__(self):
        self.name = "Two-Domain Nuclear Elastography"
        self.available = True
    
    def analyze(self, fluorescence_image: np.ndarray,
                displacement_x: np.ndarray,
                displacement_y: np.ndarray,
                nuclear_mask: Optional[np.ndarray] = None,
                parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Complete Two-Domain Nuclear Elastography analysis pipeline
        
        Args:
            fluorescence_image: Raw fluorescence intensity image of nucleus
            displacement_x: X-component of experimental displacement field
            displacement_y: Y-component of experimental displacement field
            nuclear_mask: Optional binary mask defining nuclear boundary
            parameters: Analysis parameters
                - hill_coefficient: Hill equation coefficient n (default: 2)
                - mesh_spacing: FE mesh element size (default: 4)
                - material_model: 'hookean' or 'neo_hookean' (default: 'hookean')
                - poisson_ratio: Poisson's ratio (default: 0.45)
                - initial_stiffness_ratio: Initial guess for E_h/E_e (default: 1.5)
                - optimization_method: 'Nelder-Mead', 'Powell', 'BFGS' (default: 'Nelder-Mead')
                - max_iterations: Maximum optimization iterations (default: 200)
        
        Returns:
            Dictionary containing elastography analysis results
        """
        if parameters is None:
            parameters = {}
        
        # Default parameters
        hill_coefficient = parameters.get('hill_coefficient', 2)
        mesh_spacing = parameters.get('mesh_spacing', 4)
        material_model = parameters.get('material_model', 'hookean')
        poisson_ratio = parameters.get('poisson_ratio', 0.45)
        initial_ratio = parameters.get('initial_stiffness_ratio', 1.5)
        opt_method = parameters.get('optimization_method', 'Nelder-Mead')
        max_iterations = parameters.get('max_iterations', 200)
        
        try:
            # Validate inputs
            if fluorescence_image.ndim != 2:
                return {'status': 'error', 'message': 'Elastography requires 2D fluorescence image'}
            
            if fluorescence_image.shape != displacement_x.shape:
                return {'status': 'error', 'message': 'Image and displacement field dimensions must match'}
            
            # Create nuclear mask if not provided
            if nuclear_mask is None:
                nuclear_mask = self._create_nuclear_mask(fluorescence_image)
            
            # Step 3.1: Chromatin Segmentation using Hill Function
            segmentation_result = self._segment_chromatin_domains(
                fluorescence_image, nuclear_mask, hill_coefficient
            )
            
            # Step 3.2: Create finite element mesh
            mesh = self._create_3d_mesh(
                nuclear_mask, segmentation_result['domain_mask'],
                mesh_spacing
            )
            
            # Step 3.3: Inverse Finite Element Optimization
            optimization_result = self._inverse_fe_optimization(
                mesh, displacement_x, displacement_y,
                nuclear_mask, material_model, poisson_ratio,
                initial_ratio, opt_method, max_iterations
            )
            
            # Calculate validation metrics
            validation = self._calculate_validation_metrics(
                optimization_result, displacement_x, displacement_y, nuclear_mask
            )
            
            return {
                'status': 'success',
                'method': 'Two-Domain Nuclear Elastography',
                'segmentation': segmentation_result,
                'mesh': mesh,
                'optimization': optimization_result,
                'validation': validation,
                'stiffness_ratio': optimization_result['optimal_ratio'],
                'parameters_used': parameters
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Elastography analysis failed: {str(e)}'}
    
    def _create_nuclear_mask(self, image: np.ndarray) -> np.ndarray:
        """Create nuclear mask using Otsu thresholding"""
        if SKIMAGE_AVAILABLE:
            threshold = threshold_otsu(image)
            mask = image > threshold
            mask = binary_closing(mask, disk(5))
            mask = binary_opening(mask, disk(3))
        else:
            # Simple threshold
            threshold = np.mean(image) + 0.5 * np.std(image)
            mask = image > threshold
            mask = ndimage.binary_fill_holes(mask)
        
        return mask
    
    def _segment_chromatin_domains(self, image: np.ndarray,
                                   nuclear_mask: np.ndarray,
                                   hill_coefficient: float) -> Dict[str, Any]:
        """
        Step 3.1: Chromatin Segmentation using Hill Function
        
        1. Sort pixel intensities
        2. Fit sorted curve to Hill equation (sigmoidal)
        3. Find inflection point as cutoff intensity
        4. Classify: pixels > cutoff = Heterochromatin, < cutoff = Euchromatin
        """
        # Get intensities within nucleus
        nuclear_intensities = image[nuclear_mask]
        
        # Sort intensities
        sorted_intensities = np.sort(nuclear_intensities)
        n_pixels = len(sorted_intensities)
        
        # Normalized x-axis (fraction of pixels)
        x_norm = np.arange(n_pixels) / (n_pixels - 1)
        
        # Normalize intensities to [0, 1]
        I_min = sorted_intensities.min()
        I_max = sorted_intensities.max()
        y_norm = (sorted_intensities - I_min) / (I_max - I_min + 1e-10)
        
        # Fit Hill equation: y = x^n / (K^n + x^n)
        # Reparametrized for fitting: y = (x/K)^n / (1 + (x/K)^n)
        hill_fit = self._fit_hill_equation(x_norm, y_norm, hill_coefficient)
        
        # Find inflection point of fitted curve
        cutoff_fraction = hill_fit['inflection_point']
        cutoff_idx = int(cutoff_fraction * n_pixels)
        cutoff_intensity = sorted_intensities[min(cutoff_idx, n_pixels - 1)]
        
        # Create domain mask
        # Heterochromatin (H): high intensity (> cutoff)
        # Euchromatin (E): low intensity (< cutoff)
        domain_mask = np.zeros_like(image, dtype=np.int8)
        domain_mask[nuclear_mask & (image >= cutoff_intensity)] = 1  # Heterochromatin
        domain_mask[nuclear_mask & (image < cutoff_intensity)] = -1  # Euchromatin
        
        # Calculate domain statistics
        hetero_mask = domain_mask == 1
        eu_mask = domain_mask == -1
        
        hetero_area = np.sum(hetero_mask)
        eu_area = np.sum(eu_mask)
        total_area = hetero_area + eu_area
        
        return {
            'domain_mask': domain_mask,
            'heterochromatin_mask': hetero_mask,
            'euchromatin_mask': eu_mask,
            'cutoff_intensity': cutoff_intensity,
            'hill_fit': hill_fit,
            'heterochromatin_fraction': hetero_area / total_area if total_area > 0 else 0,
            'euchromatin_fraction': eu_area / total_area if total_area > 0 else 0,
            'sorted_intensities': sorted_intensities,
            'fitted_curve': hill_fit['fitted_curve']
        }
    
    def _fit_hill_equation(self, x: np.ndarray, y: np.ndarray,
                          n_init: float) -> Dict[str, Any]:
        """
        Fit Hill equation: y = x^n / (K^n + x^n)
        
        The inflection point of this sigmoidal curve is at x = K * ((n-1)/(n+1))^(1/n)
        """
        def hill_func(x, K, n):
            x_safe = np.maximum(x, 1e-10)
            K_safe = max(K, 1e-10)
            return np.power(x_safe, n) / (np.power(K_safe, n) + np.power(x_safe, n))
        
        try:
            # Initial guess
            p0 = [0.5, n_init]
            
            # Bounds: 0 < K < 1, 1 < n < 10
            bounds = ([0.01, 1.0], [0.99, 10.0])
            
            popt, pcov = optimize.curve_fit(
                hill_func, x, y,
                p0=p0, bounds=bounds, maxfev=5000
            )
            
            K_opt, n_opt = popt
            perr = np.sqrt(np.diag(pcov))
            
            # Calculate inflection point
            # For Hill equation, inflection at x = K * ((n-1)/(n+1))^(1/n) when n > 1
            if n_opt > 1:
                inflection_x = K_opt * np.power((n_opt - 1) / (n_opt + 1), 1 / n_opt)
            else:
                inflection_x = K_opt  # Approximate
            
            # Generate fitted curve
            x_fit = np.linspace(0, 1, 1000)
            y_fit = hill_func(x_fit, K_opt, n_opt)
            
            # Calculate R-squared
            residuals = y - hill_func(x, K_opt, n_opt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'K': K_opt,
                'n': n_opt,
                'K_error': perr[0],
                'n_error': perr[1],
                'inflection_point': inflection_x,
                'r_squared': r_squared,
                'fitted_curve': (x_fit, y_fit)
            }
            
        except Exception as e:
            warnings.warn(f"Hill equation fitting failed: {str(e)}")
            # Fallback: use median as cutoff
            return {
                'K': 0.5,
                'n': n_init,
                'inflection_point': 0.5,
                'r_squared': 0,
                'fitted_curve': (np.array([0, 1]), np.array([0, 1]))
            }
    
    def _create_3d_mesh(self, nuclear_mask: np.ndarray,
                       domain_mask: np.ndarray,
                       spacing: int) -> Dict[str, Any]:
        """
        Create simplified 2D/3D finite element mesh
        
        For 2D analysis, creates triangular mesh elements assigned to 
        heterochromatin (H) or euchromatin (E) domains
        """
        H, W = nuclear_mask.shape
        
        # Node positions (simplified regular grid within nucleus)
        y_nodes = np.arange(spacing // 2, H, spacing)
        x_nodes = np.arange(spacing // 2, W, spacing)
        
        # Create nodes
        nodes = []
        node_ids = {}
        idx = 0
        
        for iy, y in enumerate(y_nodes):
            for ix, x in enumerate(x_nodes):
                # Only include nodes within or near nucleus
                y_int, x_int = int(y), int(x)
                if 0 <= y_int < H and 0 <= x_int < W:
                    if nuclear_mask[y_int, x_int]:
                        nodes.append([x, y])
                        node_ids[(iy, ix)] = idx
                        idx += 1
        
        nodes = np.array(nodes) if nodes else np.array([[W//2, H//2]])
        
        # Create elements (triangles)
        elements = []
        element_domains = []  # 1 for heterochromatin, -1 for euchromatin
        
        ny, nx = len(y_nodes), len(x_nodes)
        
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                # Check if all four corners are valid nodes
                corners = [(iy, ix), (iy, ix+1), (iy+1, ix), (iy+1, ix+1)]
                valid_corners = [c for c in corners if c in node_ids]
                
                if len(valid_corners) >= 3:
                    # Determine element domain from center
                    center_y = (y_nodes[iy] + y_nodes[iy + 1]) // 2
                    center_x = (x_nodes[ix] + x_nodes[ix + 1]) // 2
                    
                    if 0 <= center_y < H and 0 <= center_x < W:
                        element_domain = domain_mask[int(center_y), int(center_x)]
                    else:
                        element_domain = 0
                    
                    # Create triangular elements if we have valid nodes
                    if (iy, ix) in node_ids and (iy, ix+1) in node_ids and (iy+1, ix) in node_ids:
                        elements.append([
                            node_ids[(iy, ix)],
                            node_ids[(iy, ix+1)],
                            node_ids[(iy+1, ix)]
                        ])
                        element_domains.append(element_domain)
                    
                    if (iy, ix+1) in node_ids and (iy+1, ix+1) in node_ids and (iy+1, ix) in node_ids:
                        elements.append([
                            node_ids[(iy, ix+1)],
                            node_ids[(iy+1, ix+1)],
                            node_ids[(iy+1, ix)]
                        ])
                        element_domains.append(element_domain)
        
        elements = np.array(elements) if elements else np.array([[0, 0, 0]])
        element_domains = np.array(element_domains) if element_domains else np.array([0])
        
        return {
            'nodes': nodes,
            'elements': elements,
            'element_domains': element_domains,
            'n_heterochromatin': np.sum(element_domains == 1),
            'n_euchromatin': np.sum(element_domains == -1),
            'spacing': spacing
        }
    
    def _inverse_fe_optimization(self, mesh: Dict[str, Any],
                                 exp_disp_x: np.ndarray,
                                 exp_disp_y: np.ndarray,
                                 nuclear_mask: np.ndarray,
                                 material_model: str,
                                 poisson_ratio: float,
                                 initial_ratio: float,
                                 opt_method: str,
                                 max_iterations: int) -> Dict[str, Any]:
        """
        Step 3.2: Inverse Finite Element Optimization
        
        Find optimal E_h/E_e ratio that minimizes difference between
        simulated and experimental displacement fields
        """
        # Sample experimental displacement at node locations
        nodes = mesh['nodes']
        
        exp_disp_at_nodes = self._sample_displacement_at_nodes(
            exp_disp_x, exp_disp_y, nodes
        )
        
        # Objective function: sum of squared errors
        def objective(log_ratio):
            ratio = np.exp(log_ratio[0])  # Use log to ensure positive
            
            # Run forward simulation
            sim_disp = self._forward_fe_simulation(
                mesh, exp_disp_at_nodes, ratio,
                material_model, poisson_ratio
            )
            
            # Calculate error
            error = self._calculate_displacement_error(
                sim_disp, exp_disp_at_nodes
            )
            
            return error
        
        # Optimization
        result = optimize.minimize(
            objective,
            x0=[np.log(initial_ratio)],
            method=opt_method,
            options={'maxiter': max_iterations, 'disp': False}
        )
        
        optimal_ratio = np.exp(result.x[0])
        
        # Run final simulation with optimal ratio
        final_sim_disp = self._forward_fe_simulation(
            mesh, exp_disp_at_nodes, optimal_ratio,
            material_model, poisson_ratio
        )
        
        # Calculate final error
        final_error = self._calculate_displacement_error(
            final_sim_disp, exp_disp_at_nodes
        )
        
        # Estimate E_h and E_e (normalized, relative values)
        # Convention: E_e = 1 (reference), E_h = ratio * E_e
        E_euchromatin = 1.0
        E_heterochromatin = optimal_ratio * E_euchromatin
        
        return {
            'optimal_ratio': optimal_ratio,
            'E_heterochromatin': E_heterochromatin,
            'E_euchromatin': E_euchromatin,
            'final_error': final_error,
            'simulated_displacement': final_sim_disp,
            'experimental_displacement': exp_disp_at_nodes,
            'optimization_success': result.success,
            'optimization_iterations': result.nit if hasattr(result, 'nit') else max_iterations
        }
    
    def _sample_displacement_at_nodes(self, disp_x: np.ndarray,
                                      disp_y: np.ndarray,
                                      nodes: np.ndarray) -> np.ndarray:
        """Sample experimental displacement field at mesh node locations"""
        H, W = disp_x.shape
        
        # Create interpolators
        y_coords = np.arange(H)
        x_coords = np.arange(W)
        
        interp_x = interpolate.RegularGridInterpolator(
            (y_coords, x_coords), disp_x,
            method='linear', bounds_error=False, fill_value=0
        )
        interp_y = interpolate.RegularGridInterpolator(
            (y_coords, x_coords), disp_y,
            method='linear', bounds_error=False, fill_value=0
        )
        
        # Sample at node locations
        node_coords = np.stack([nodes[:, 1], nodes[:, 0]], axis=1)  # [y, x]
        
        disp_at_nodes = np.zeros((len(nodes), 2))
        disp_at_nodes[:, 0] = interp_x(node_coords)
        disp_at_nodes[:, 1] = interp_y(node_coords)
        
        return disp_at_nodes
    
    def _forward_fe_simulation(self, mesh: Dict[str, Any],
                               boundary_displacement: np.ndarray,
                               stiffness_ratio: float,
                               material_model: str,
                               poisson_ratio: float) -> np.ndarray:
        """
        Forward FE simulation with given stiffness ratio
        
        Solves equilibrium equations with material properties assigned
        based on domain segmentation
        """
        nodes = mesh['nodes']
        elements = mesh['elements']
        element_domains = mesh['element_domains']
        
        num_nodes = len(nodes)
        num_dof = num_nodes * 2
        
        # Assign elastic moduli based on domain
        # E_euchromatin = 1 (reference), E_heterochromatin = ratio
        E_eu = 1.0
        E_hetero = stiffness_ratio * E_eu
        
        # Assemble global stiffness matrix
        K_global = np.zeros((num_dof, num_dof))
        
        for elem_idx, element in enumerate(elements):
            domain = element_domains[elem_idx]
            
            # Assign material property
            if domain == 1:  # Heterochromatin
                E = E_hetero
            elif domain == -1:  # Euchromatin
                E = E_eu
            else:  # Mixed or boundary
                E = (E_hetero + E_eu) / 2
            
            # Calculate element stiffness matrix
            K_elem = self._element_stiffness_matrix(
                nodes[element], E, poisson_ratio, material_model
            )
            
            # Assemble into global matrix
            for i, ni in enumerate(element):
                for j, nj in enumerate(element):
                    for di in range(2):
                        for dj in range(2):
                            K_global[2*ni + di, 2*nj + dj] += K_elem[2*i + di, 2*j + dj]
        
        # Apply boundary conditions (prescribed displacement at boundary nodes)
        # Identify boundary nodes (nodes with non-zero experimental displacement)
        boundary_mask = np.any(np.abs(boundary_displacement) > 1e-10, axis=1)
        
        if np.sum(boundary_mask) == 0:
            # If no boundary conditions, return experimental displacement
            return boundary_displacement.copy()
        
        # Solve reduced system for internal nodes
        internal_mask = ~boundary_mask
        
        if np.sum(internal_mask) == 0:
            # All nodes are boundary nodes
            return boundary_displacement.copy()
        
        # Create force vector from boundary conditions
        F = np.zeros(num_dof)
        
        # Extract prescribed DOFs
        prescribed_dof = []
        prescribed_values = []
        for i in range(num_nodes):
            if boundary_mask[i]:
                prescribed_dof.extend([2*i, 2*i + 1])
                prescribed_values.extend([boundary_displacement[i, 0], 
                                         boundary_displacement[i, 1]])
        
        prescribed_dof = np.array(prescribed_dof)
        prescribed_values = np.array(prescribed_values)
        
        # Calculate force contribution from prescribed displacements
        for i, dof in enumerate(prescribed_dof):
            F -= K_global[:, dof] * prescribed_values[i]
        
        # Identify free DOFs
        all_dof = np.arange(num_dof)
        free_dof = np.setdiff1d(all_dof, prescribed_dof)
        
        if len(free_dof) == 0:
            return boundary_displacement.copy()
        
        # Reduced stiffness matrix
        K_reduced = K_global[np.ix_(free_dof, free_dof)]
        F_reduced = F[free_dof]
        
        # Add small regularization for numerical stability
        K_reduced += 1e-10 * np.eye(len(free_dof))
        
        # Solve for free DOF displacements
        try:
            u_free = np.linalg.solve(K_reduced, F_reduced)
        except np.linalg.LinAlgError:
            # If singular, use least squares
            u_free, _, _, _ = np.linalg.lstsq(K_reduced, F_reduced, rcond=None)
        
        # Assemble full displacement vector
        u_full = np.zeros(num_dof)
        u_full[prescribed_dof] = prescribed_values
        u_full[free_dof] = u_free
        
        # Reshape to nodal displacement array
        simulated_displacement = np.zeros((num_nodes, 2))
        simulated_displacement[:, 0] = u_full[0::2]
        simulated_displacement[:, 1] = u_full[1::2]
        
        return simulated_displacement
    
    def _element_stiffness_matrix(self, elem_nodes: np.ndarray,
                                  E: float, nu: float,
                                  material_model: str) -> np.ndarray:
        """
        Calculate element stiffness matrix for triangular element
        
        For plane stress (2D):
        K_e = A * B^T * D * B
        
        Where:
        - A: element area
        - B: strain-displacement matrix
        - D: material constitutive matrix
        """
        # Node coordinates
        x1, y1 = elem_nodes[0]
        x2, y2 = elem_nodes[1]
        x3, y3 = elem_nodes[2]
        
        # Element area
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        area = max(area, 1e-10)
        
        # Shape function derivatives (constant for linear triangle)
        b1 = (y2 - y3) / (2 * area)
        b2 = (y3 - y1) / (2 * area)
        b3 = (y1 - y2) / (2 * area)
        c1 = (x3 - x2) / (2 * area)
        c2 = (x1 - x3) / (2 * area)
        c3 = (x2 - x1) / (2 * area)
        
        # B matrix (strain-displacement)
        B = np.array([
            [b1, 0, b2, 0, b3, 0],
            [0, c1, 0, c2, 0, c3],
            [c1, b1, c2, b2, c3, b3]
        ])
        
        # D matrix (constitutive) - plane stress
        if material_model == 'hookean' or material_model == 'linear':
            factor = E / (1 - nu**2)
            D = factor * np.array([
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1 - nu) / 2]
            ])
        else:  # neo_hookean (linearized for small strain)
            mu = E / (2 * (1 + nu))
            lam = E * nu / ((1 + nu) * (1 - 2*nu))
            D = np.array([
                [lam + 2*mu, lam, 0],
                [lam, lam + 2*mu, 0],
                [0, 0, mu]
            ])
        
        # Element stiffness matrix
        K_elem = area * B.T @ D @ B
        
        return K_elem
    
    def _calculate_displacement_error(self, sim_disp: np.ndarray,
                                      exp_disp: np.ndarray) -> float:
        """
        Calculate sum of squared errors between simulated and experimental displacements
        SE = Σ [d'(x,y) - d(x,y)]²
        """
        diff = sim_disp - exp_disp
        error = np.sum(diff**2)
        
        return error
    
    def _calculate_validation_metrics(self, optimization_result: Dict[str, Any],
                                      exp_disp_x: np.ndarray,
                                      exp_disp_y: np.ndarray,
                                      nuclear_mask: np.ndarray) -> Dict[str, Any]:
        """Calculate validation metrics for the elastography result"""
        
        sim_disp = optimization_result['simulated_displacement']
        exp_disp = optimization_result['experimental_displacement']
        
        # Root Mean Square Error
        rmse_x = np.sqrt(np.mean((sim_disp[:, 0] - exp_disp[:, 0])**2))
        rmse_y = np.sqrt(np.mean((sim_disp[:, 1] - exp_disp[:, 1])**2))
        rmse_total = np.sqrt(np.mean(np.sum((sim_disp - exp_disp)**2, axis=1)))
        
        # Correlation coefficient
        if np.std(exp_disp[:, 0]) > 0 and np.std(sim_disp[:, 0]) > 0:
            corr_x = np.corrcoef(sim_disp[:, 0], exp_disp[:, 0])[0, 1]
        else:
            corr_x = 0
        
        if np.std(exp_disp[:, 1]) > 0 and np.std(sim_disp[:, 1]) > 0:
            corr_y = np.corrcoef(sim_disp[:, 1], exp_disp[:, 1])[0, 1]
        else:
            corr_y = 0
        
        # Normalized error
        exp_magnitude = np.sqrt(np.sum(exp_disp**2))
        normalized_error = optimization_result['final_error'] / (exp_magnitude + 1e-10)
        
        return {
            'rmse_x': rmse_x,
            'rmse_y': rmse_y,
            'rmse_total': rmse_total,
            'correlation_x': corr_x,
            'correlation_y': corr_y,
            'normalized_error': normalized_error,
            'stiffness_ratio': optimization_result['optimal_ratio'],
            'optimization_converged': optimization_result['optimization_success']
        }


def get_elastography_parameters() -> Dict[str, Any]:
    """Get default parameters for Two-Domain Nuclear Elastography"""
    return {
        'hill_coefficient': {
            'default': 2,
            'min': 1,
            'max': 10,
            'description': 'Hill equation coefficient for sigmoidal fit'
        },
        'mesh_spacing': {
            'default': 4,
            'min': 2,
            'max': 16,
            'description': 'Finite element mesh node spacing in pixels'
        },
        'material_model': {
            'default': 'hookean',
            'options': ['hookean', 'neo_hookean'],
            'description': 'Material constitutive model'
        },
        'poisson_ratio': {
            'default': 0.45,
            'min': 0.1,
            'max': 0.49,
            'description': "Poisson's ratio (nearly incompressible for biological materials)"
        },
        'initial_stiffness_ratio': {
            'default': 1.5,
            'min': 0.5,
            'max': 5.0,
            'description': 'Initial guess for E_h/E_e ratio'
        },
        'optimization_method': {
            'default': 'Nelder-Mead',
            'options': ['Nelder-Mead', 'Powell', 'BFGS'],
            'description': 'Optimization algorithm'
        },
        'max_iterations': {
            'default': 200,
            'min': 50,
            'max': 1000,
            'description': 'Maximum optimization iterations'
        }
    }
