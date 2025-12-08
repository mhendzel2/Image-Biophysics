"""
Deformation Microscopy (DM) Module
Based on Ghosh et al. (2019) - Deformation Microscopy for Dynamic Intracellular 
and Intranuclear Mapping of Mechanics with High Spatiotemporal Resolution

Implements:
- Hyperelastic warping image registration
- High-resolution strain tensor computation
- Hydrostatic, deviatoric, shear, and principal strain maps

Reference:
Ghosh, S., et al. (2019). Deformation Microscopy for Dynamic Intracellular and
Intranuclear Mapping of Mechanics with High Spatiotemporal Resolution.
Cell Reports, 27(5), 1607-1620.
"""

import numpy as np
from scipy import ndimage, optimize, interpolate
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Dict, Any, Tuple, Optional, List
import warnings

try:
    from skimage.filters import gaussian
    from skimage.transform import warp
    from skimage.registration import optical_flow_tvl1
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not available - some DM features limited")


class DeformationMicroscopy:
    """
    Deformation Microscopy (DM) Analysis
    
    Quantifies high-resolution intracellular and intranuclear strain maps
    using hyperelastic warping for image registration.
    
    Key outputs:
    - Nodal displacement maps d(x,y)
    - Strain tensors (E_xx, E_yy, E_xy)
    - Hydrostatic strain (area/volume change)
    - Deviatoric strain (shape change)
    - Principal strains (maximal tensile/compressive)
    """
    
    def __init__(self):
        self.name = "Deformation Microscopy"
        self.available = SKIMAGE_AVAILABLE
    
    def analyze(self, template_image: np.ndarray,
                target_image: np.ndarray,
                parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Complete Deformation Microscopy analysis pipeline
        
        Args:
            template_image: Undeformed/reference image
            target_image: Deformed/target image
            parameters: Analysis parameters
                - mesh_spacing: Finite element mesh node spacing (default: 8)
                - material_model: 'neo_hookean' or 'linear' (default: 'neo_hookean')
                - regularization: Smoothness regularization weight (default: 0.1)
                - iterations: Maximum optimization iterations (default: 100)
                - pixel_size: Pixel size in μm (default: 0.1)
                - convergence_threshold: Convergence criterion (default: 1e-5)
        
        Returns:
            Dictionary containing displacement and strain analysis results
        """
        if parameters is None:
            parameters = {}
        
        # Default parameters
        mesh_spacing = parameters.get('mesh_spacing', 8)
        material_model = parameters.get('material_model', 'neo_hookean')
        regularization = parameters.get('regularization', 0.1)
        iterations = parameters.get('iterations', 100)
        pixel_size = parameters.get('pixel_size', 0.1)
        convergence_threshold = parameters.get('convergence_threshold', 1e-5)
        
        try:
            # Validate inputs
            if template_image.shape != target_image.shape:
                return {'status': 'error', 'message': 'Template and target images must have same dimensions'}
            
            if template_image.ndim != 2:
                return {'status': 'error', 'message': 'DM requires 2D images'}
            
            # Step 2.1: Image Registration using Hyperelastic Warping
            displacement_result = self._hyperelastic_warping(
                template_image, target_image,
                mesh_spacing, material_model, regularization,
                iterations, convergence_threshold
            )
            
            if displacement_result['status'] != 'success':
                return displacement_result
            
            # Step 2.2: Strain Tensor Computation
            strain_result = self._compute_strain_tensors(
                displacement_result['displacement_x'],
                displacement_result['displacement_y'],
                pixel_size
            )
            
            # Generate warped image for validation
            warped_image = self._warp_image(
                template_image,
                displacement_result['displacement_x'],
                displacement_result['displacement_y']
            )
            
            # Calculate registration quality metrics
            quality_metrics = self._calculate_registration_quality(
                template_image, target_image, warped_image
            )
            
            return {
                'status': 'success',
                'method': 'Deformation Microscopy',
                'displacement': displacement_result,
                'strain': strain_result,
                'warped_image': warped_image,
                'quality_metrics': quality_metrics,
                'parameters_used': parameters
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Deformation Microscopy failed: {str(e)}'}
    
    def _hyperelastic_warping(self, template: np.ndarray,
                             target: np.ndarray,
                             mesh_spacing: int,
                             material_model: str,
                             regularization: float,
                             max_iterations: int,
                             convergence_threshold: float) -> Dict[str, Any]:
        """
        Step 2.1: Hyperelastic Warping Image Registration
        
        Minimizes combined energy functional:
        E_total = E_strain (hyperelastic) + E_intensity (image similarity)
        """
        H, W = template.shape
        
        # Create finite element mesh
        mesh = self._create_mesh(H, W, mesh_spacing)
        
        # Initialize displacement field
        num_nodes = len(mesh['nodes'])
        displacement = np.zeros(num_nodes * 2)  # [u1, v1, u2, v2, ...]
        
        # Normalize images
        template_norm = (template - np.mean(template)) / (np.std(template) + 1e-10)
        target_norm = (target - np.mean(target)) / (np.std(target) + 1e-10)
        
        # Optimization loop
        prev_energy = np.inf
        converged = False
        
        for iteration in range(max_iterations):
            # Calculate energy and gradient
            energy, gradient = self._compute_energy_gradient(
                displacement, template_norm, target_norm,
                mesh, material_model, regularization
            )
            
            # Check convergence
            energy_change = abs(prev_energy - energy) / (abs(prev_energy) + 1e-10)
            if energy_change < convergence_threshold:
                converged = True
                break
            
            # Gradient descent step with line search
            step_size = self._line_search(
                displacement, gradient, template_norm, target_norm,
                mesh, material_model, regularization, energy
            )
            
            displacement = displacement - step_size * gradient
            prev_energy = energy
        
        # Interpolate displacement to full image grid
        disp_x, disp_y = self._interpolate_displacement(
            displacement, mesh, H, W
        )
        
        return {
            'status': 'success',
            'displacement_x': disp_x,
            'displacement_y': disp_y,
            'mesh': mesh,
            'nodal_displacement': displacement,
            'converged': converged,
            'final_energy': prev_energy,
            'iterations': iteration + 1
        }
    
    def _create_mesh(self, H: int, W: int, spacing: int) -> Dict[str, Any]:
        """Create 2D finite element mesh over image domain"""
        
        # Node positions
        y_nodes = np.arange(0, H, spacing)
        x_nodes = np.arange(0, W, spacing)
        
        # Ensure coverage of full image
        if y_nodes[-1] < H - 1:
            y_nodes = np.append(y_nodes, H - 1)
        if x_nodes[-1] < W - 1:
            x_nodes = np.append(x_nodes, W - 1)
        
        # Create node coordinates
        nodes = []
        node_ids = {}
        idx = 0
        for iy, y in enumerate(y_nodes):
            for ix, x in enumerate(x_nodes):
                nodes.append([x, y])
                node_ids[(iy, ix)] = idx
                idx += 1
        
        nodes = np.array(nodes)
        
        # Create quadrilateral elements
        elements = []
        ny, nx = len(y_nodes), len(x_nodes)
        
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                # Nodes of quadrilateral (counterclockwise)
                n1 = node_ids[(iy, ix)]
                n2 = node_ids[(iy, ix + 1)]
                n3 = node_ids[(iy + 1, ix + 1)]
                n4 = node_ids[(iy + 1, ix)]
                
                # Split into two triangles
                elements.append([n1, n2, n4])  # Triangle 1
                elements.append([n2, n3, n4])  # Triangle 2
        
        elements = np.array(elements)
        
        return {
            'nodes': nodes,
            'elements': elements,
            'y_grid': y_nodes,
            'x_grid': x_nodes,
            'spacing': spacing
        }
    
    def _compute_energy_gradient(self, displacement: np.ndarray,
                                 template: np.ndarray,
                                 target: np.ndarray,
                                 mesh: Dict[str, Any],
                                 material_model: str,
                                 regularization: float) -> Tuple[float, np.ndarray]:
        """Compute total energy and gradient"""
        
        # Strain energy (regularization)
        E_strain, grad_strain = self._compute_strain_energy(
            displacement, mesh, material_model
        )
        
        # Intensity energy (image similarity)
        E_intensity, grad_intensity = self._compute_intensity_energy(
            displacement, template, target, mesh
        )
        
        # Total energy
        total_energy = E_intensity + regularization * E_strain
        total_gradient = grad_intensity + regularization * grad_strain
        
        return total_energy, total_gradient
    
    def _compute_strain_energy(self, displacement: np.ndarray,
                               mesh: Dict[str, Any],
                               material_model: str) -> Tuple[float, np.ndarray]:
        """
        Compute strain energy using Neo-Hookean or linear model
        
        Neo-Hookean: W = μ/2 * (I₁ - 3) - μ * ln(J) + λ/2 * (ln(J))²
        Linear: W = μ * (E:E) + λ/2 * (tr(E))²
        """
        nodes = mesh['nodes']
        elements = mesh['elements']
        num_nodes = len(nodes)
        
        # Material parameters (normalized)
        mu = 1.0  # Shear modulus
        lam = 1.0  # Lamé's first parameter
        
        total_energy = 0.0
        gradient = np.zeros_like(displacement)
        
        for element in elements:
            # Get nodal positions and displacements
            elem_nodes = nodes[element]
            elem_disp = np.array([
                [displacement[2*n], displacement[2*n + 1]] for n in element
            ])
            
            # Calculate deformation gradient F
            F, dF_du = self._compute_deformation_gradient(elem_nodes, elem_disp)
            
            if material_model == 'neo_hookean':
                W, dW_dF = self._neo_hookean_energy(F, mu, lam)
            else:  # linear
                W, dW_dF = self._linear_elastic_energy(F, mu, lam)
            
            # Element area
            area = self._triangle_area(elem_nodes)
            
            total_energy += W * area
            
            # Gradient contribution
            for i, n in enumerate(element):
                for j in range(2):
                    dF_du_ij = dF_du[i, j]  # Derivative of F w.r.t. u_i,j
                    grad_contrib = np.sum(dW_dF * dF_du_ij) * area
                    gradient[2*n + j] += grad_contrib
        
        return total_energy, gradient
    
    def _compute_deformation_gradient(self, elem_nodes: np.ndarray,
                                      elem_disp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute deformation gradient F = I + ∂u/∂X for triangular element"""
        
        # Reference configuration (material coordinates)
        X = elem_nodes
        
        # Deformed configuration
        x = X + elem_disp
        
        # Shape function derivatives (for linear triangle)
        # N_i = a_i + b_i*x + c_i*y
        dN_dX = self._shape_function_derivatives(X)
        
        # Displacement gradient
        du_dX = np.zeros((2, 2))
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    du_dX[j, k] += elem_disp[i, j] * dN_dX[i, k]
        
        # Deformation gradient F = I + du/dX
        F = np.eye(2) + du_dX
        
        # Derivative of F w.r.t. nodal displacements
        dF_du = np.zeros((3, 2, 2, 2))  # [node, component, F_row, F_col]
        for i in range(3):
            for j in range(2):  # displacement component
                for k in range(2):  # F column
                    dF_du[i, j, j, k] = dN_dX[i, k]
        
        return F, dF_du
    
    def _shape_function_derivatives(self, nodes: np.ndarray) -> np.ndarray:
        """Calculate shape function derivatives for linear triangle"""
        x1, y1 = nodes[0]
        x2, y2 = nodes[1]
        x3, y3 = nodes[2]
        
        # Area
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        area = max(area, 1e-10)  # Avoid division by zero
        
        # Shape function derivatives
        dN_dX = np.array([
            [(y2 - y3) / (2 * area), (x3 - x2) / (2 * area)],
            [(y3 - y1) / (2 * area), (x1 - x3) / (2 * area)],
            [(y1 - y2) / (2 * area), (x2 - x1) / (2 * area)]
        ])
        
        return dN_dX
    
    def _neo_hookean_energy(self, F: np.ndarray,
                           mu: float, lam: float) -> Tuple[float, np.ndarray]:
        """
        Neo-Hookean hyperelastic energy density
        W = μ/2 * (I₁ - 2) - μ * ln(J) + λ/2 * (ln(J))²
        """
        # Right Cauchy-Green tensor C = F^T F
        C = F.T @ F
        
        # Invariants
        I1 = np.trace(C)
        J = np.linalg.det(F)
        J = max(J, 1e-10)  # Ensure positive
        
        # Energy
        W = mu / 2 * (I1 - 2) - mu * np.log(J) + lam / 2 * np.log(J)**2
        
        # Derivative dW/dF (First Piola-Kirchhoff stress P)
        # P = μF - μF^{-T} + λ ln(J) F^{-T}
        F_inv_T = np.linalg.inv(F).T
        P = mu * F - mu * F_inv_T + lam * np.log(J) * F_inv_T
        
        return W, P
    
    def _linear_elastic_energy(self, F: np.ndarray,
                               mu: float, lam: float) -> Tuple[float, np.ndarray]:
        """
        Linear elastic energy density (small strain approximation)
        W = μ * (E:E) + λ/2 * (tr(E))²
        """
        # Small strain tensor E = 1/2 * (F + F^T) - I
        E = 0.5 * (F + F.T) - np.eye(2)
        
        # Energy
        tr_E = np.trace(E)
        W = mu * np.sum(E * E) + lam / 2 * tr_E**2
        
        # Derivative dW/dE
        dW_dE = 2 * mu * E + lam * tr_E * np.eye(2)
        
        # Convert to dW/dF
        dW_dF = 0.5 * (dW_dE + dW_dE.T)
        
        return W, dW_dF
    
    def _triangle_area(self, nodes: np.ndarray) -> float:
        """Calculate area of triangle"""
        x1, y1 = nodes[0]
        x2, y2 = nodes[1]
        x3, y3 = nodes[2]
        return 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    
    def _compute_intensity_energy(self, displacement: np.ndarray,
                                  template: np.ndarray,
                                  target: np.ndarray,
                                  mesh: Dict[str, Any]) -> Tuple[float, np.ndarray]:
        """
        Compute intensity-based image similarity energy
        E_intensity = Σ (I_template(x + u) - I_target(x))²
        """
        nodes = mesh['nodes']
        num_nodes = len(nodes)
        
        H, W = template.shape
        
        # Create interpolator for template
        y_coords = np.arange(H)
        x_coords = np.arange(W)
        template_interp = interpolate.RegularGridInterpolator(
            (y_coords, x_coords), template,
            method='linear', bounds_error=False, fill_value=0
        )
        
        # Interpolate displacement to dense grid
        disp_x, disp_y = self._interpolate_displacement(displacement, mesh, H, W)
        
        # Warped coordinates
        yy, xx = np.mgrid[:H, :W]
        warped_x = xx + disp_x
        warped_y = yy + disp_y
        
        # Evaluate template at warped coordinates
        coords = np.stack([warped_y.ravel(), warped_x.ravel()], axis=1)
        warped_template = template_interp(coords).reshape(H, W)
        
        # Intensity difference
        diff = warped_template - target
        
        # Energy (sum of squared differences)
        energy = np.sum(diff**2)
        
        # Gradient w.r.t. nodal displacements (simplified - using chain rule)
        gradient = np.zeros(num_nodes * 2)
        
        # Compute image gradients
        grad_y, grad_x = np.gradient(template)
        grad_x_interp = interpolate.RegularGridInterpolator(
            (y_coords, x_coords), grad_x,
            method='linear', bounds_error=False, fill_value=0
        )
        grad_y_interp = interpolate.RegularGridInterpolator(
            (y_coords, x_coords), grad_y,
            method='linear', bounds_error=False, fill_value=0
        )
        
        warped_grad_x = grad_x_interp(coords).reshape(H, W)
        warped_grad_y = grad_y_interp(coords).reshape(H, W)
        
        # For each node, compute gradient contribution
        for i, (nx, ny) in enumerate(nodes):
            # Find region influenced by this node
            # Using bilinear interpolation weights
            y_min = max(0, int(ny - mesh['spacing']))
            y_max = min(H, int(ny + mesh['spacing']))
            x_min = max(0, int(nx - mesh['spacing']))
            x_max = min(W, int(nx + mesh['spacing']))
            
            # Accumulate gradient
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    # Bilinear weight
                    weight = self._bilinear_weight(x, y, nx, ny, mesh['spacing'])
                    
                    grad_contrib = 2 * diff[y, x] * weight
                    gradient[2*i] += grad_contrib * warped_grad_x[y, x]
                    gradient[2*i + 1] += grad_contrib * warped_grad_y[y, x]
        
        return energy, gradient
    
    def _bilinear_weight(self, x: float, y: float, 
                        nx: float, ny: float, spacing: float) -> float:
        """Calculate bilinear interpolation weight"""
        dx = abs(x - nx) / spacing
        dy = abs(y - ny) / spacing
        
        if dx >= 1 or dy >= 1:
            return 0
        
        return (1 - dx) * (1 - dy)
    
    def _line_search(self, displacement: np.ndarray,
                    gradient: np.ndarray,
                    template: np.ndarray,
                    target: np.ndarray,
                    mesh: Dict[str, Any],
                    material_model: str,
                    regularization: float,
                    current_energy: float) -> float:
        """Backtracking line search for step size"""
        alpha = 1.0
        beta = 0.5  # Reduction factor
        c = 1e-4  # Armijo condition constant
        
        grad_norm_sq = np.sum(gradient**2)
        
        for _ in range(20):  # Max line search iterations
            new_disp = displacement - alpha * gradient
            new_energy, _ = self._compute_energy_gradient(
                new_disp, template, target, mesh, material_model, regularization
            )
            
            # Armijo condition
            if new_energy <= current_energy - c * alpha * grad_norm_sq:
                break
            
            alpha *= beta
        
        return max(alpha, 1e-6)
    
    def _interpolate_displacement(self, nodal_disp: np.ndarray,
                                  mesh: Dict[str, Any],
                                  H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate nodal displacements to full image grid"""
        nodes = mesh['nodes']
        y_grid = mesh['y_grid']
        x_grid = mesh['x_grid']
        
        ny, nx = len(y_grid), len(x_grid)
        
        # Reshape nodal displacements to grid
        disp_x_nodes = np.zeros((ny, nx))
        disp_y_nodes = np.zeros((ny, nx))
        
        idx = 0
        for iy in range(ny):
            for ix in range(nx):
                disp_x_nodes[iy, ix] = nodal_disp[2*idx]
                disp_y_nodes[iy, ix] = nodal_disp[2*idx + 1]
                idx += 1
        
        # Interpolate to full grid
        interp_x = interpolate.RegularGridInterpolator(
            (y_grid, x_grid), disp_x_nodes,
            method='linear', bounds_error=False, fill_value=0
        )
        interp_y = interpolate.RegularGridInterpolator(
            (y_grid, x_grid), disp_y_nodes,
            method='linear', bounds_error=False, fill_value=0
        )
        
        yy, xx = np.mgrid[:H, :W]
        coords = np.stack([yy.ravel(), xx.ravel()], axis=1)
        
        disp_x = interp_x(coords).reshape(H, W)
        disp_y = interp_y(coords).reshape(H, W)
        
        return disp_x, disp_y
    
    def _compute_strain_tensors(self, disp_x: np.ndarray,
                                disp_y: np.ndarray,
                                pixel_size: float) -> Dict[str, Any]:
        """
        Step 2.2: Strain Tensor Computation
        
        Computes:
        - Hydrostatic strain: E_hyd = 1/2 * (E_xx + E_yy)
        - Deviatoric strain: E_dev = sqrt((E_xx - E_hyd)² + E_xy²)
        - Shear strain: E_xy
        - Principal strains (max tensile/compressive)
        """
        # Compute displacement gradients
        du_dx = np.gradient(disp_x, pixel_size, axis=1)
        du_dy = np.gradient(disp_x, pixel_size, axis=0)
        dv_dx = np.gradient(disp_y, pixel_size, axis=1)
        dv_dy = np.gradient(disp_y, pixel_size, axis=0)
        
        # Green-Lagrange strain tensor components
        # E = 1/2 * (F^T F - I) for large deformation
        # For small deformation: E ≈ 1/2 * (∇u + ∇u^T)
        E_xx = du_dx + 0.5 * (du_dx**2 + dv_dx**2)
        E_yy = dv_dy + 0.5 * (du_dy**2 + dv_dy**2)
        E_xy = 0.5 * (du_dy + dv_dx + du_dx*du_dy + dv_dx*dv_dy)
        
        # Hydrostatic strain (area change in 2D)
        E_hydrostatic = 0.5 * (E_xx + E_yy)
        
        # Deviatoric strain (shape change)
        E_deviatoric = np.sqrt((E_xx - E_hydrostatic)**2 + E_xy**2)
        
        # Principal strains (eigenvalues of strain tensor)
        E_principal_1 = np.zeros_like(E_xx)
        E_principal_2 = np.zeros_like(E_xx)
        principal_angle = np.zeros_like(E_xx)
        
        for i in range(E_xx.shape[0]):
            for j in range(E_xx.shape[1]):
                strain_tensor = np.array([
                    [E_xx[i, j], E_xy[i, j]],
                    [E_xy[i, j], E_yy[i, j]]
                ])
                
                eigenvalues, eigenvectors = np.linalg.eigh(strain_tensor)
                
                # Sort eigenvalues (max first)
                idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                E_principal_1[i, j] = eigenvalues[0]  # Maximum (tensile)
                E_principal_2[i, j] = eigenvalues[1]  # Minimum (compressive)
                
                # Principal direction angle
                principal_angle[i, j] = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        
        # Effective strain (von Mises equivalent)
        E_effective = np.sqrt(2/3 * (E_xx**2 + E_yy**2 + 2*E_xy**2))
        
        return {
            'E_xx': E_xx,
            'E_yy': E_yy,
            'E_xy': E_xy,
            'E_hydrostatic': E_hydrostatic,
            'E_deviatoric': E_deviatoric,
            'E_principal_1': E_principal_1,
            'E_principal_2': E_principal_2,
            'principal_angle': principal_angle,
            'E_effective': E_effective,
            'max_tensile_strain': np.max(E_principal_1),
            'max_compressive_strain': np.min(E_principal_2),
            'mean_hydrostatic': np.mean(E_hydrostatic),
            'mean_deviatoric': np.mean(E_deviatoric)
        }
    
    def _warp_image(self, image: np.ndarray,
                   disp_x: np.ndarray,
                   disp_y: np.ndarray) -> np.ndarray:
        """Warp image using displacement field"""
        H, W = image.shape
        
        yy, xx = np.mgrid[:H, :W]
        warped_x = xx + disp_x
        warped_y = yy + disp_y
        
        # Interpolate
        coords = np.stack([warped_y, warped_x], axis=-1)
        warped = ndimage.map_coordinates(image, [warped_y, warped_x], order=1, mode='constant')
        
        return warped
    
    def _calculate_registration_quality(self, template: np.ndarray,
                                        target: np.ndarray,
                                        warped: np.ndarray) -> Dict[str, float]:
        """Calculate registration quality metrics"""
        
        # Normalized Cross Correlation before and after
        ncc_before = self._ncc(template, target)
        ncc_after = self._ncc(warped, target)
        
        # Mean Squared Error
        mse_before = np.mean((template - target)**2)
        mse_after = np.mean((warped - target)**2)
        
        # Structural Similarity Index (simplified)
        ssim_before = self._ssim(template, target)
        ssim_after = self._ssim(warped, target)
        
        return {
            'ncc_before': ncc_before,
            'ncc_after': ncc_after,
            'ncc_improvement': ncc_after - ncc_before,
            'mse_before': mse_before,
            'mse_after': mse_after,
            'mse_reduction': (mse_before - mse_after) / mse_before if mse_before > 0 else 0,
            'ssim_before': ssim_before,
            'ssim_after': ssim_after,
            'ssim_improvement': ssim_after - ssim_before
        }
    
    def _ncc(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Normalized Cross Correlation"""
        img1_norm = (img1 - np.mean(img1)) / (np.std(img1) + 1e-10)
        img2_norm = (img2 - np.mean(img2)) / (np.std(img2) + 1e-10)
        return np.mean(img1_norm * img2_norm)
    
    def _ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Simplified Structural Similarity Index"""
        c1 = 0.01**2
        c2 = 0.03**2
        
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return ssim


def get_dm_parameters() -> Dict[str, Any]:
    """Get default parameters for Deformation Microscopy"""
    return {
        'mesh_spacing': {
            'default': 8,
            'min': 4,
            'max': 32,
            'description': 'Finite element mesh node spacing in pixels'
        },
        'material_model': {
            'default': 'neo_hookean',
            'options': ['neo_hookean', 'linear'],
            'description': 'Hyperelastic material model'
        },
        'regularization': {
            'default': 0.1,
            'min': 0.001,
            'max': 10.0,
            'description': 'Strain energy regularization weight'
        },
        'iterations': {
            'default': 100,
            'min': 10,
            'max': 500,
            'description': 'Maximum optimization iterations'
        },
        'pixel_size': {
            'default': 0.1,
            'min': 0.01,
            'max': 10.0,
            'description': 'Pixel size in micrometers'
        },
        'convergence_threshold': {
            'default': 1e-5,
            'min': 1e-8,
            'max': 1e-2,
            'description': 'Convergence criterion for optimization'
        }
    }
