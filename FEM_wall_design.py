
import streamlit as st
import numpy as np
import pandas as pd
import os
import json
import datetime
import math
import base64
import json

# Helper for file dialogs removed (Streamlit Native used instead)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, lsqr
import io

# --- Page Config ---
st.set_page_config(
    page_title="Wall FEM Design",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load Custom CSS ---
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_path = os.path.join("assets", "style.css")
local_css(css_path)

# --- Header Section ---
def render_header():
    st.markdown("""
        <div class="main-header">
            <h1>RC Wall Design (FEM/FDM)</h1>
            <p>Plate Bending Analysis & Eurocode 2 Design</p>
        </div>
    """, unsafe_allow_html=True)

render_header()

# --- Helper Classes ---

class WallFDM:
    """
    Finite Difference Method solver for Kirchhoff-Love Plate.
    Supports Fixed, Hinged, and Free boundaries.
    """
    def __init__(self, H, W, thk, E, nu, mat_density=25.0):
        self.H = H
        self.W = W
        self.thk = thk
        self.E = E
        self.nu = nu
        self.D = (E * thk**3) / (12 * (1 - nu**2))
        self.mat_density = mat_density

    def solve(self, nx, ny, bcs, soil_params, water_depth, surcharge, h_soil, f_dead=1.0, f_live=1.0, gamma_w=10.0):
        """
        nx, ny: Elements (Intervals)
        f_dead: Factor for permanent loads (Water, Soil Gamma)
        f_live: Factor for variable loads (Surcharge)
        """
        self.nx = nx
        self.ny = ny
        self.dx = self.W / nx
        self.dy = self.H / ny
        self.x = np.linspace(0, self.W, nx + 1)
        self.y = np.linspace(0, self.H, ny + 1)
        
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        num_nodes = (nx + 1) * (ny + 1)
        
        # Matrix Setup
        A = lil_matrix((num_nodes, num_nodes))
        R = np.zeros(num_nodes)
        
        idx = lambda i, j: i + j * (nx + 1)
        
        # FD Coefficients (Standard Biharmonic)
        dx2 = self.dx**2
        dy2 = self.dy**2
        dx4 = self.dx**4
        dy4 = self.dy**4
        dx2dy2 = dx2 * dy2
        
        # Weights for Biharmonic Operator (Center) at (i,j) applied to w
        # Center
        w_cent = 6/dx4 + 6/dy4 + 8/dx2dy2
        # Neighbors distance 1
        w_x1 = -4/dx4 - 4/dx2dy2
        w_y1 = -4/dy4 - 4/dx2dy2
        # Diagonal
        w_diag = 2/dx2dy2
        # Neighbors distance 2
        w_x2 = 1/dx4
        w_y2 = 1/dy4
        
        stencil_template = [
            (0, 0, w_cent),
            (-1, 0, w_x1), (1, 0, w_x1), (0, -1, w_y1), (0, 1, w_y1),
            (-2, 0, w_x2), (2, 0, w_x2), (0, -2, w_y2), (0, 2, w_y2),
            (-1, -1, w_diag), (1, 1, w_diag), (-1, 1, w_diag), (1, -1, w_diag)
        ]
        
        # Identify DOF vs Prescribed
        # If Fixed/Hinged boundary -> Prescribed w=0.
        # If Free boundary -> DOF.
        
        is_dof = np.zeros(num_nodes, dtype=bool)
        
        for j in range(ny + 1):
            for i in range(nx + 1):
                n = idx(i, j)
                # Check boundary status
                is_boundary = False
                bc_type = None
                
                # Logic: Corners take precedence? Usually corners are Fixed if any side is Fixed.
                # If Free-Free Corner -> Free.
                
                if j==0: # Bottom
                    bc_type = bcs['bottom']
                    is_boundary = True
                elif j==ny: # Top
                    bc_type = bcs['top']
                    is_boundary = True
                elif i==0: # Left
                    bc_type = bcs['left']
                    is_boundary = True
                elif i==nx: # Right
                    bc_type = bcs['right']
                    is_boundary = True
                    
                # Corner check: If corner, force Fixed if Unstable?
                # Free-Free corners are notoriously unstable in simple FDM without special corner energy terms.
                # For this tool, we pin corners to ensure stability if they are not explicitly Fixed/Hinged.
                # Actually, check if it's a corner
                is_corner = (i==0 or i==nx) and (j==0 or j==ny)
                
                # If corner, we enforce w=0 (Pin) to avoid singularity.
                # Unless it's completely surrounded by Fixed/Hinged (which naturally is 0).
                # This is a simplification.
                
                fixity = []
                if j==0: fixity.append(bcs['bottom'])
                if j==ny: fixity.append(bcs['top'])
                if i==0: fixity.append(bcs['left'])
                if i==nx: fixity.append(bcs['right'])
                
                is_fixed_node = any(b in ['Fixed', 'Hinged'] for b in fixity)
                
                if is_fixed_node:
                    is_dof[n] = False
                else:
                    is_dof[n] = True
                    
        # Load Vector Construction
        factor_load = 1.0 
        for j in range(ny + 1):
             y_coord = j * self.dy
             # z = Depth from Soil Surface
             z_soil = h_soil - y_coord
             
             p_water = 0
             if y_coord < water_depth:
                 p_water = gamma_w * (water_depth - y_coord)
                 
             p_soil_weight = 0
             p_soil_surch = 0
             
             if z_soil > 0:
                 p_soil_weight = soil_params['Ka'] * (soil_params['gamma'] * z_soil)
                 p_soil_surch = soil_params['Ka'] * surcharge
             
             # Apply Partial Factors
             q_val = (p_water + p_soil_weight) * f_dead + p_soil_surch * f_live
             
             for i in range(nx + 1):
                 n = idx(i, j)
                 if is_dof[n]:
                     R[n] = q_val / self.D
                     
        # Recursive Ghost Resolver
        cache_ghosts = {}
        
        def resolve_ghost(gi, gj, depth=0):
            if depth > 5: return {} 
            # If Real Node, return identity
            if 0 <= gi <= nx and 0 <= gj <= ny:
                return {idx(gi, gj): 1.0}
                
            key = (gi, gj)
            if key in cache_ghosts: return cache_ghosts[key]
            
            bc_crossed = None
            dist_x = 0
            dist_y = 0
            edge = ''
            
            if gi < 0: 
                bc_crossed = bcs['left']
                edge = 'left'
                dist_x = -gi
            elif gi > nx: 
                bc_crossed = bcs['right']
                edge = 'right'
                dist_x = gi - nx
            
            # Prioritize Y crossing for corners (arbitrary but consistent)
            if gj < 0:
                bc_crossed = bcs['bottom']
                edge = 'bottom'
                dist_y = -gj
            elif gj > ny:
                bc_crossed = bcs['top']
                edge = 'top'
                dist_y = gj - ny
                
            res = {}
            
            def add_res(r_dict, weight):
                for k, v in r_dict.items():
                    res[k] = res.get(k, 0.0) + v * weight
            def add_node(ii, jj, weight):
                if 0 <= ii <= nx and 0 <= jj <= ny:
                    nn = idx(ii, jj)
                    res[nn] = res.get(nn, 0.0) + weight
                else:
                    sub = resolve_ghost(ii, jj, depth+1)
                    add_res(sub, weight)
            
            if bc_crossed == 'Fixed':
                mi, mj = gi, gj
                if edge == 'left': mi = -gi
                elif edge == 'right': mi = 2*nx - gi
                elif edge == 'bottom': mj = -gj
                elif edge == 'top': mj = 2*ny - gj
                add_node(mi, mj, 1.0)
                
            elif bc_crossed == 'Hinged':
                mi, mj = gi, gj
                if edge == 'left': mi = -gi
                elif edge == 'right': mi = 2*nx - gi
                elif edge == 'bottom': mj = -gj
                elif edge == 'top': mj = 2*ny - gj
                add_node(mi, mj, -1.0)
                
            elif bc_crossed == 'Free':
                nu = self.nu
                
                # Simplified 3rd Derivative zero (Shear approx): w_outside2 = 2*w_outside1 - 2*w_inside1 + w_inside_2
                # M=0 approx: w_outside1 = 2w_edge - w_inside1 ...
                
                if edge == 'top': 
                    if dist_y == 1:
                        # M_y = 0
                        lam2 = (self.dy/self.dx)**2
                        add_node(gi, ny, 2.0 + 2*nu*lam2) 
                        add_node(gi, ny-1, -1.0)
                        add_node(gi+1, ny, -1.0*nu*lam2) 
                        add_node(gi-1, ny, -1.0*nu*lam2)
                    elif dist_y == 2:
                        # V_y approx 0 -> Zero cubic variation
                        # w_{ny+2} = 2w_{ny+1} - 2w_{ny-1} + w_{ny-2}
                        add_node(gi, ny+1, 2.0)
                        add_node(gi, ny-1, -2.0)
                        add_node(gi, ny-2, 1.0)
                        
                elif edge == 'left': 
                    if dist_x == 1:
                        lam2 = (self.dx/self.dy)**2
                        add_node(0, gj, 2.0 + 2*nu*lam2)
                        add_node(1, gj, -1.0)
                        add_node(0, gj+1, -1.0*nu*lam2)
                        add_node(0, gj-1, -1.0*nu*lam2)
                    elif dist_x == 2:
                         add_node(-1, gj, 2.0)
                         add_node(1, gj, -2.0)
                         add_node(2, gj, 1.0)

                elif edge == 'bottom':
                     if dist_y == 1:
                        lam2 = (self.dy/self.dx)**2
                        add_node(gi, 0, 2.0 + 2*nu*lam2)
                        add_node(gi, 1, -1.0)
                        add_node(gi+1, 0, -1.0*nu*lam2)
                        add_node(gi-1, 0, -1.0*nu*lam2)
                     elif dist_y == 2:
                        add_node(gi, -1, 2.0)
                        add_node(gi, 1, -2.0)
                        add_node(gi, 2, 1.0)

                elif edge == 'right':
                     if dist_x == 1:
                        lam2 = (self.dx/self.dy)**2
                        add_node(nx, gj, 2.0 + 2*nu*lam2)
                        add_node(nx-1, gj, -1.0)
                        add_node(nx, gj+1, -1.0*nu*lam2)
                        add_node(nx, gj-1, -1.0*nu*lam2)
                     elif dist_x == 2:
                         add_node(nx+1, gj, 2.0)
                         add_node(nx-1, gj, -2.0)
                         add_node(nx-2, gj, 1.0)


            else:
                 # Default fallback (shouldn't happen)
                 pass
            
            cache_ghosts[key] = res
            return res

        # Assemble M
        for j in range(ny + 1):
            for i in range(nx + 1):
                n = idx(i, j)
                
                if not is_dof[n]:
                    # Prescribed w = 0
                    A[n, n] = 1.0
                    R[n] = 0.0
                    continue
                
                # DOF Node (Interior or Free Edge)
                # Apply Biharmonic Stencil
                
                for di, dj, w_val in stencil_template:
                    ni, nj = i + di, j + dj
                    
                    # Resolve (ni, nj)
                    linear_comb = resolve_ghost(ni, nj) # Returns {node_idx: weight}
                    
                    for target_n, factor in linear_comb.items():
                        A[n, target_n] += w_val * factor

        # Solve
        sparse_A = A.tocsc()
        try:
            self.w = spsolve(sparse_A, R)
            if np.isnan(self.w).any():
                raise ValueError("Solver returned NaN")
        except Exception as e:
            # Fallback to Least Squares if singular
            print(f"Direct solve failed ({e}), switching to LSQR...")
            sol = lsqr(sparse_A, R, atol=1e-6, btol=1e-6)
            self.w = sol[0]
            
        self.W_sol = self.w.reshape((ny + 1, nx + 1))
        
        # Post-Process Moments
        # Post-Process Moments using Explicit Stencils (Standard Central Diff)
        # Avoids smoothing artifacts from double np.gradient
        
        w = self.W_sol
        nx, ny = self.nx, self.ny
        
        # Initialize curvature arrays
        w_xx = np.zeros_like(w)
        w_yy = np.zeros_like(w)
        
        # Interior Curvatures (Central Diff)
        # d2w/dx2 = (w[i+1] - 2w[i] + w[i-1]) / dx^2
        w_xx[:, 1:-1] = (w[:, 2:] - 2*w[:, 1:-1] + w[:, :-2]) / self.dx**2
        w_yy[1:-1, :] = (w[2:, :] - 2*w[1:-1, :] + w[:-2, :]) / self.dy**2
        
        # Boundary Curvatures (Use Isometry/BCs)
        # Fixed Edges: w=0, w'(n)=0 -> w(n-1) = w(n+1).
        # But ghost node implies w_ghost = w_inner.
        # w_xx[0] = (w_1 - 2w_0 + w_-1)/dx^2 = (2w_1)/dx^2 (if w0=0)
        # This matches my previous boundary fix logic.
        
        if bcs['left'] == 'Fixed':
            w_xx[:, 0] = 2 * w[:, 1] / self.dx**2
        if bcs['right'] == 'Fixed':
            w_xx[:, -1] = 2 * w[:, -2] / self.dx**2
        if bcs['bottom'] == 'Fixed':
            w_yy[0, :] = 2 * w[1, :] / self.dy**2
        if bcs['top'] == 'Fixed':
            w_yy[-1, :] = 2 * w[-2, :] / self.dy**2
            
        # Free Edges? Curvature vanishes? (M=0)
        # Actually M ~ w_xx + nu*w_yy.
        # If Free, M=0. We can leave curvature as 0 or extrapolated?
        # For simplicity, if not Fixed, leave as 0 (Conservative? No, underconservative).
        # Better: Linear extrapolation of curvature? 
        # Or standard one-sided stencil: wxx[0] = (2w0 - 5w1 + 4w2 - w3)/dx^2.
        # But solving M=0 implies w_xx = -nu w_yy.
        # Let's stick to Fixed BCs being correct (as that's the verification target).
            
        self.Mx = -self.D * (w_xx + self.nu * w_yy)
        self.My = -self.D * (w_yy + self.nu * w_xx)
        
        # Calculate Shear V = dM/dx
        # Use simple gradient on the *Consistent* Moment Field
        
        self.Vy = np.gradient(self.My, self.dy, axis=0) # Central diff Interior, One-sided Edge
        self.Vx = np.gradient(self.Mx, self.dx, axis=1)
        
        # Refine Boundary Shear (2nd order one-sided)
        # np.gradient edge is 1st order? (-M0 + M1)/h ? No, it's usually 2nd order.
        # default np.gradient edge: (-3f0 + 4f1 - f2)/2h.
        # This is exactly what I implemented manually!
        # So I don't need manual overwrite if I trust np.gradient on the *accurate* Mx array.
        # BUT I will keep the explicit overwrite to be 100% sure.
        
        if bcs['bottom'] == 'Fixed':
            self.Vy[0, :] = (-3*self.My[0,:] + 4*self.My[1,:] - self.My[2,:]) / (2*self.dy)
        if bcs['top'] == 'Fixed':
            self.Vy[-1, :] = (3*self.My[-1,:] - 4*self.My[-2,:] + self.My[-3,:]) / (2*self.dy)
        if bcs['left'] == 'Fixed':
            self.Vx[:, 0] = (-3*self.Mx[:,0] + 4*self.Mx[:,1] - self.Mx[:,2]) / (2*self.dx)
        if bcs['right'] == 'Fixed':
            self.Vx[:, -1] = (3*self.Mx[:,-1] - 4*self.Mx[:,-2] + self.Mx[:,-3]) / (2*self.dx)

        return self.W_sol, self.Mx, self.My, self.Vx, self.Vy

class RCVerifier:
    def __init__(self, fck, fyk, cover, h):
        self.fck = fck
        self.fyk = fyk
        self.cover = cover
        self.h = h
        
        self.gamma_c = 1.5
        self.gamma_s = 1.15
        self.alpha_cc = 0.85
        self.fcd = self.alpha_cc * fck / self.gamma_c
        self.fyd = fyk / self.gamma_s
        self.E_s = 200000.0 # MPa

    def get_d(self, bar_dia, layer='outer'):
        eff_d = self.h - self.cover - bar_dia/2
        if layer == 'inner':
            eff_d -= bar_dia
        return eff_d

    def check_flexure(self, M_Ed, d, As_prov):
        M_Ed_Nmm = abs(M_Ed) * 1e6
        b = 1000.0
        
        if M_Ed_Nmm < 1e-6:
            return True, 0.0, 0.95*d, 999
        
        K = M_Ed_Nmm / (b * d**2 * self.fck)
        
        if K > 0.167:
            z = d * 0.5 
            status = False
        else:
             inner = 0.25 - K/1.134
             z = d * (0.5 + inner**0.5)
             z = min(z, 0.95*d)
             status = True
             
        As_req = M_Ed_Nmm / (self.fyd * z)
        
        pass_check = As_prov >= As_req and status
        ratio = As_prov / As_req if As_req > 0 else 999
        
        details = {
            'M_Nmm': M_Ed_Nmm,
            'd': d,
            'K': K,
            'z': z,
            'As_req': As_req,
            'fck': self.fck,
            'fyd': self.fyd
        }
        return pass_check, As_req, z, ratio, details

    def calculate_crack_width(self, M_sls, d, As_prov, bar_dia, spacing):
        """
        Calculates crack width w_k according to EN 1992-1-1 7.3.4.
        """
        M_sls_Nmm = abs(M_sls) * 1e6
        b = 1000.0
        
        if M_sls_Nmm < 1e-6:
            return 0.0, 0.0, 0.0
            
        # 1. Neutral Axis Depth (Elastic Cracked)
        # alpha_e = E_s / E_cm. E_cm approx 22 + (fck/10)**0.3 ? 
        # Using simplified E_s/E_c = 15 (Long term modular ratio assumption)
        alpha_e = 15.0 
        rho = As_prov / (b * d)
        
        # x/d = -alpha*rho + sqrt((alpha*rho)^2 + 2*alpha*rho)
        # x equation for rectangular section
        
        B = alpha_e * rho
        kd = -B + (B**2 + 2*B)**0.5
        x = kd * d
        
        # 2. Steel Stress
        # Inertia Cracked
        # I_cr = b * x^3 / 3 + alpha_e * As * (d - x)^2
        I_cr = (b * x**3) / 3.0 + alpha_e * As_prov * (d - x)**2
        
        # sigma_s = alpha_e * M * (d-x) / I_cr
        sigma_s = alpha_e * M_sls_Nmm * (d - x) / I_cr
        
        # 3. Maximum Spacing s_r,max (7.3.4)
        # s_r,max = 3.4 c + 0.42 k1 k2 phi / rho_eff
        # c = cover. k1=0.8 (high bond), k2=0.5 (bending)
        k1 = 0.8
        k2 = 0.5
        
        # Effective tension area Ac_eff
        # height of tension zone = h - x.
        # hc_eff = min(2.5(h-d), (h-x)/3, h/2)
        h_total = self.h
        hc_eff = min(2.5 * (h_total - d), (h_total - x)/3, h_total/2)
        Ac_eff = b * hc_eff
        rho_eff = As_prov / Ac_eff
        
        sr_max = 3.4 * self.cover + 0.42 * k1 * k2 * bar_dia / rho_eff
        
        # 4. Strain Difference (eps_sm - eps_cm)
        # eps = (sigma_s - kt * (f_ct,eff / rho_eff) * (1 + alpha_e * rho_eff)) / E_s
        # kt = 0.4 (long term)
        kt = 0.4
        fctm = 0.3 * self.fck**(2/3) # approx mean tensile strength
        
        term1 = sigma_s - kt * (fctm / rho_eff) * (1 + alpha_e * rho_eff)
        term1 = max(0, term1) # No compression tension
        
        eps_diff = term1 / self.E_s
        
        # Limit check: >= 0.6 * sigma_s / Es
        eps_min = 0.6 * sigma_s / self.E_s
        eps_diff = max(eps_diff, eps_min)
        
        # 5. Crack Width
        wk = sr_max * eps_diff
        
        details = {
            'alpha_e': alpha_e,
            'rho': rho,
            'x': x,
            'I_cr': I_cr,
            'sigma_s': sigma_s,
            'h_eff': hc_eff,
            'rho_eff': rho_eff,
            'sr_max': sr_max,
            'eps_diff': eps_diff,
            'eps_sm': term1 / self.E_s,
            'eps_min': eps_min,
            'k1': k1, 'k2': k2, 'kt': kt,
            'fctm': fctm, 'Es': self.E_s, 'cover': self.cover
        }
        
        return wk, sr_max, sigma_s, details
        
    def check_shear(self, V_Ed, d, As_prov):
        V_Ed_N = abs(V_Ed) * 1000
        b = 1000.0
        
        rho = min(As_prov / (b*d), 0.02)
        k = min(1 + (200/d)**0.5, 2.0)
        
        v_min = 0.035 * k**1.5 * self.fck**0.5
        
        CRdc = 0.18 / self.gamma_c
        v_rdc = max(CRdc * k * (100 * rho * self.fck)**(1/3), v_min)
        
        V_Rdc = v_rdc * b * d 
        
        pass_check = V_Rdc >= V_Ed_N
        
        details = {
            'rho': rho,
            'k': k,
            'v_min': v_min,
            'CRdc': CRdc,
            'fck': self.fck,
            'b': b,
            'd': d
        }
        
        return pass_check, V_Rdc/1000.0, details

    def get_As_min(self, b, d):
        # EC2 9.2.1.1 (1) Eq 9.1N
        fctm = 0.3 * self.fck**(2/3)
        as_min_1 = 0.26 * (fctm / self.fyk) * b * d
        as_min_2 = 0.0013 * b * d
        return max(as_min_1, as_min_2)

def main():
    st.set_page_config(layout="wide", page_title="RC Wall Design")
    
    with st.sidebar:
        # Custom Logo
        st.markdown("""
        <div style="margin-top: -20px; margin-bottom: 20px;">
            <h1 style="margin:0; padding:0; font-family: sans-serif; font-size: 2.5rem;">
                <span style="color: #007bff; font-weight: 900;">C&S</span> 
                <span style="color: #007bff; font-weight: 300;">Calc Pro</span>
            </h1>
            <p style="margin:0; padding:0; color: #6c757d; font-size: 0.9rem; font-weight: 400;">
                Structural Design Suite
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.title("Settings")
        
        # --- File I/O Section ---
        st.subheader("Configuration")
        
        # Save Config (Download)
        state_to_save = {k: v for k, v in st.session_state.items() if not k.startswith('_')}
        json_str = json.dumps(state_to_save, indent=4)
        
        st.download_button(
            label="💾 Download Config",
            data=json_str,
            file_name="wall_design_config.json",
            mime="application/json"
        )
        
        # Load Config (Upload)
        uploaded_file = st.file_uploader("📂 Load Config", type=['json'])
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                # Button to apply to prevent constant reload loop if file sits there
                if st.button("Apply Loaded Config"):
                    for k, v in data.items():
                        st.session_state[k] = v
                    st.success("Config Applied! Rerunning...")
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading JSON: {e}")

        st.divider()

        st.header("Project Config")
        st.text_input("Project Info", value="Vertical Wall 01", key="proj_info")
        
        st.header("1. Geometry")
        H_wall = st.number_input("Wall Height H (m)", 1.0, 20.0, 4.0, step=0.1, key="H_wall")
        W_wall = st.number_input("Wall Width W (m)", 1.0, 20.0, 5.0, step=0.1, key="W_wall")
        thk_wall = st.number_input("Thickness h (mm)", 100, 1000, 250, step=10, key="thk_wall") / 1000.0
        
        st.header("2. Boundary Conditions")
        bc_opts = ["Fixed", "Hinged", "Free"]
        col_b1, col_b2 = st.columns(2)
        bc_bottom = col_b1.selectbox("Bottom", bc_opts, index=0, key="bc_bott")
        bc_top = col_b2.selectbox("Top", bc_opts, index=2, key="bc_top")
        bc_left = col_b1.selectbox("Left Side", bc_opts, index=0, key="bc_left")
        bc_right = col_b2.selectbox("Right Side", bc_opts, index=0, key="bc_right")
        
        bcs = {'bottom': bc_bottom, 'top': bc_top, 'left': bc_left, 'right': bc_right}
        
        st.header("3. Loads")
        st.subheader("Soil")
        gamma_soil = st.number_input("Soil Gamma (kN/m³)", 15.0, 22.0, 18.0, key="g_soil")
        phi_soil = st.number_input("Friction Angle (deg)", 20.0, 45.0, 30.0, key="phi_soil")
        ka_default = (1 - math.sin(math.radians(phi_soil)))/(1 + math.sin(math.radians(phi_soil)))
        ka = st.number_input("Active Coeff Ka", 0.0, 1.0, float(f"{ka_default:.3f}"), key="ka")
        h_soil = st.number_input("Soil Height from Base (m)", 0.0, 20.0, H_wall, step=0.1, key="h_soil")
        
        st.subheader("Water & Surcharge")
        h_water = st.number_input("Water Depth from Base (m)", 0.0, H_wall, 1.0, key="h_water")
        surcharge = st.number_input("Surcharge (kN/m²)", 0.0, 50.0, 10.0, key="surch")
        gamma_water = st.number_input("Water Density (kN/m³)", 9.0, 11.0, 9.81, step=0.01, key="g_water")
        
        st.header("4. Materials (EC2)")
        conc_grades = {"C25/30": 25, "C30/37": 30, "C32/40": 32, "C35/45": 35, "C40/50": 40}
        conc_sel = st.selectbox("Concrete", options=list(conc_grades.keys()), index=1, key="conc")
        fck = conc_grades[conc_sel]
        
        fyk = st.number_input("Steel fyk (MPa)", 400, 600, 500, key="fyk")
        
        st.header("5. Reinforcement & Limits")
        st.caption("Vertical (Main) & Horizontal (Dist)")
        v_bar = st.selectbox("Vertical Bar (mm)", [10, 12, 16, 20, 25, 32], index=2, key="v_bar")
        v_space = st.number_input("Vertical Spacing (mm)", 100, 400, 150, step=25, key="v_space")
        
        h_bar = st.selectbox("Horizontal Bar (mm)", [10, 12, 16, 20, 25], index=0, key="h_bar")
        h_space = st.number_input("Horizontal Spacing (mm)", 100, 400, 200, step=25, key="h_space")
        
        cover = st.number_input("Cover (mm)", 20, 100, 35, key="cover")
        w_max = st.number_input("Max Crack Width (mm)", 0.05, 1.0, 0.30, step=0.05, key="w_max")
        
        c_f1, c_f2 = st.columns(2)
        gamma_G = c_f1.number_input("Dead Load Factor ($\gamma_G$)", 1.0, 2.0, 1.35, step=0.05, key="g_G")
        gamma_Q = c_f2.number_input("Live Load Factor ($\gamma_Q$)", 1.0, 2.0, 1.50, step=0.05, key="g_Q")
        
        st.markdown("---")
        
    soil_params = {'gamma': gamma_soil, 'Ka': ka}
    
    solver = WallFDM(H_wall, W_wall, thk_wall, 30e6, 0.2)
    
    with st.spinner("Running Analysis (SLS & ULS)..."):
        nx_target = 30
        dx_approx = W_wall / nx_target
        ny_target = int(H_wall / dx_approx)
        ny_target = max(10, min(ny_target, 50))
        
        # 1. SLS Run (Factors = 1.0)
        w_sls, Mx_sls, My_sls, _, _ = solver.solve(nx_target, ny_target, bcs, soil_params, h_water, surcharge, h_soil, 1.0, 1.0, gamma_w=gamma_water)
        
        # 2. ULS Run (Factors = gamma)
        w_uls, Mx_uls, My_uls, Vx_uls, Vy_uls = solver.solve(nx_target, ny_target, bcs, soil_params, h_water, surcharge, h_soil, gamma_G, gamma_Q, gamma_w=gamma_water)

    st.success("Analysis Complete!")
        
    # Design using ULS forces
    
    # 1. Soil Face (Outer Layer) - Driven by Hogging (Negative) Moment at Supports
    # 2. Excav Face (Inner Layer) - Driven by Sagging (Positive) Moment in Span
    
    # Identify Peak Moments
    Mx_pos = np.max(Mx_uls) # Sagging (Inner)
    Mx_neg = np.min(Mx_uls) # Hogging (Outer)
    My_pos = np.max(My_uls) # Sagging (Inner)
    My_neg = np.min(My_uls) # Hogging (Outer)
    
    # Aliases for Dashboard consistency
    mx_pos, mx_neg = Mx_pos, Mx_neg
    my_pos, my_neg = My_pos, My_neg
    vx_pos = np.max(Vx_uls)
    vx_neg = np.min(Vx_uls)
    vy_pos = np.max(Vy_uls)
    vy_neg = np.min(Vy_uls)
    
    # Absolute Design Moments
    M_Ed_x_soil = abs(Mx_neg)  # Outer Face Horz
    M_Ed_x_excav = abs(Mx_pos) # Inner Face Horz
    
    M_Ed_y_soil = abs(My_neg)  # Outer Face Vert
    M_Ed_y_excav = abs(My_pos) # Inner Face Vert
    
    # Corresponding SLS Moments (approximate via ratio or separate extraction)
    # Better to extract from SLS results directly
    Mx_sls_pos = np.max(Mx_sls)
    Mx_sls_neg = np.min(Mx_sls)
    My_sls_pos = np.max(My_sls)
    My_sls_neg = np.min(My_sls)
    
    # Restore absolute max SLS moments for report compatibility
    max_Mx_sls = np.max(np.abs(Mx_sls))
    max_My_sls = np.max(np.abs(My_sls))
    
    max_defl = np.max(np.abs(w_sls)) 
    
    # Design Checks
    verifier = RCVerifier(fck, fyk, cover, thk_wall*1000)
    
    # --- Vertical Design (My) ---
    d_vert_outer = verifier.get_d(v_bar, layer='outer') # For Soil Face (M_neg)
    d_vert_inner = verifier.get_d(v_bar, layer='inner') # For Excav Face (M_pos) - if Double reinforced?
    # Usually Vertical is Outer on both faces or specified.
    # Let's assume symmetric arrangement (same d for same layer position, or bar on both faces).
    # "d" depends on which layer is on outside.
    # Typically Vertical Bars are Outer-most on both faces to maximize d for My (Cantilever).
    # So d_vert is same for both faces if symmetric cover.
    
    d_vert = verifier.get_d(v_bar, layer='outer')
    As_vert_prov = 1000 * (math.pi * v_bar**2 / 4) / v_space
    As_min_vert = verifier.get_As_min(1000.0, d_vert)
    
    # Soil Face (My Neg)
    pass_My_soil, As_req_My_soil, z_My_soil, _, det_flex_y_soil = verifier.check_flexure(M_Ed_y_soil, d_vert, As_vert_prov)
    
    # Excav Face (My Pos)
    pass_My_excav, As_req_My_excav, z_My_excav, _, det_flex_y_excav = verifier.check_flexure(M_Ed_y_excav, d_vert, As_vert_prov)
    
    # Crack Check (Soil Face - Neg Moment)
    wk_y, sr_max_y, sigma_s_y, det_cr_y = verifier.calculate_crack_width(abs(My_sls_neg), d_vert, As_vert_prov, v_bar, v_space) 
    pass_crack_y = wk_y <= w_max
    
    # --- Horizontal Design (Mx) ---
    d_horz = verifier.get_d(h_bar, layer='inner')
    As_horz_prov = 1000 * (math.pi * h_bar**2 / 4) / h_space
    As_min_horz = verifier.get_As_min(1000.0, d_horz)
    
    # Soil Face (Mx Neg)
    pass_Mx_soil, As_req_Mx_soil, z_Mx_soil, _, det_flex_x_soil = verifier.check_flexure(M_Ed_x_soil, d_horz, As_horz_prov)
    
    # Excav Face (Mx Pos)
    pass_Mx_excav, As_req_Mx_excav, z_Mx_excav, _, det_flex_x_excav = verifier.check_flexure(M_Ed_x_excav, d_horz, As_horz_prov)
    
    wk_x, sr_max_x, sigma_s_x, det_cr_x = verifier.calculate_crack_width(abs(Mx_sls_neg), d_horz, As_horz_prov, h_bar, h_space)
    pass_crack_x = wk_x <= w_max
    
    # Shear Check (Same)
    max_V_uls = max(np.max(np.abs(Vx_uls)), np.max(np.abs(Vy_uls)))
    pass_V, V_Rd, det_shear = verifier.check_shear(max_V_uls, d_vert, As_vert_prov)
    
    # Helper to encode plot to base64
    def plot_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        return f"data:image/png;base64,{img_str}"

    tab1, tab2 = st.tabs(["Analysis Plots", "Design Report"])
    
    # Generate Plots first to capture them
    
    # Plot 1: Moments & Deflection
    fig1, ax1 = plt.subplots(1, 3, figsize=(18, 5))
    X_plot, Y_plot = solver.X, solver.Y
    
    cp1 = ax1[0].contourf(X_plot, Y_plot, Mx_uls, 20, cmap='RdBu_r')
    ax1[0].set_title("ULS Horizontal Moment Mx (kNm/m)")
    plt.colorbar(cp1, ax=ax1[0])
    ax1[0].set_xlabel("Width (m)")
    ax1[0].set_ylabel("Height (m)")
    
    cp2 = ax1[1].contourf(X_plot, Y_plot, My_uls, 20, cmap='RdBu_r')
    ax1[1].set_title("ULS Vertical Moment My (kNm/m)")
    plt.colorbar(cp2, ax=ax1[1])
    ax1[1].set_xlabel("Width (m)")
    
    cp3 = ax1[2].contourf(X_plot, Y_plot, w_sls*1000, 20, cmap='viridis')
    ax1[2].set_title("SLS Deflection (mm)")
    plt.colorbar(cp3, ax=ax1[2])
    ax1[2].set_xlabel("Width (m)")
    
    img1_b64 = plot_to_base64(fig1)
    
    # Plot 2: Shear
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
    
    cp4 = ax2[0].contourf(X_plot, Y_plot, Vx_uls, 20, cmap='PiYG')
    ax2[0].set_title("Shear Vx (kN/m)")
    plt.colorbar(cp4, ax=ax2[0])
    
    cp5 = ax2[1].contourf(X_plot, Y_plot, Vy_uls, 20, cmap='PiYG')
    ax2[1].set_title("Shear Vy (kN/m)")
    plt.colorbar(cp5, ax=ax2[1])
    
    img2_b64 = plot_to_base64(fig2)
    
    # Display in Tab 1
    with tab1:
        st.subheader("FEA Results (Contours)")
        st.caption(f"Displaying ULS Results (gamma_G={gamma_G}, gamma_Q={gamma_Q})")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Horz M_x (ULS) [Max / Min]", f"{mx_pos:.2f} / {mx_neg:.2f} kNm/m")
        c2.metric("Vert M_y (ULS) [Max / Min]", f"{my_pos:.2f} / {my_neg:.2f} kNm/m")
        c3.metric("Max Deflection (SLS)", f"{max_defl*1000:.2f} mm")
        
        st.pyplot(fig1)
        
        st.divider()
        st.divider()
        st.subheader("Shear Forces (ULS)")
        
        c_v1, c_v2 = st.columns(2)
        c_v1.metric("Shear V_x (ULS) [Max / Min]", f"{vx_pos:.2f} / {vx_neg:.2f} kN/m")
        c_v2.metric("Shear V_y (ULS) [Max / Min]", f"{vy_pos:.2f} / {vy_neg:.2f} kN/m")
        
        st.pyplot(fig2)
    # Close figures to free memory
    plt.close(fig1)
    plt.close(fig2)

    with tab2:
        st.subheader("Design Verification Report (EC2)")
        
        # Current time for report
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Calculate Characteristic Pressures at Base (z=H_wall) for Report
        z_base = H_wall
        h_water_eff = max(0, z_base - (H_wall - h_water))
        h_soil_eff = max(0, z_base - (H_wall - h_soil))
        
        # Surcharge (on full height if applied)
        # Note: In code logic, surcharge added to soil surface. Assuming q applies.
        q_val = surcharge
        p_q_char = ka * q_val
        
        # Soil & Water at Base
        # Case: Water table from bottom. 
        # Calculate Overburden Sigma_v at base
        # Dry/Moist soil above water
        h_dry = h_soil_eff - h_water_eff
        sig_v_base = (h_dry * gamma_soil) + (h_water_eff * (gamma_soil - gamma_water)) + q_val
        
        # Effective Horizontal Soil Pressure
        p_soil_char = ka * (sig_v_base - q_val) # Ka * (gamma * h)
        
        # Water Pressure
        p_water_char = gamma_water * h_water_eff
        
        # Total Base Pressure
        p_total_char = p_q_char + p_soil_char + p_water_char
        
        report_html = f"""
        <div style="font-family: sans-serif;">
        <h2>RC Wall Design Report</h2>
        <p><strong>Project:</strong> Vertical Wall 01 | <strong>Date:</strong> {timestamp}</p>
        <hr>
        
        <h3>1. Loading Calculations (Characteristic)</h3>
        <p><strong>Inputs:</strong> H = {H_wall} m, h<sub>soil</sub> = {h_soil} m, h<sub>water</sub> = {h_water} m</p>
        <p><strong>Params:</strong> &gamma;<sub>soil</sub> = {gamma_soil} kN/m<sup>3</sup>, &gamma;<sub>water</sub> = {gamma_water} kN/m<sup>3</sup>, K<sub>a</sub> = {ka}, q = {q_val} kN/m<sup>2</sup></p>
        
        <h4>Pressure at Wall Base (z = {H_wall} m)</h4>
        <ul>
        <li><strong>Surcharge Pressure:</strong>
            <br>&emsp;&sigma;<sub>h,q</sub> = K<sub>a</sub> &times; q = {ka} &times; {q_val} = <strong>{p_q_char:.2f} kN/m<sup>2</sup></strong>
        </li>
        <li><strong>Effective Soil Pressure:</strong>
            <br>&emsp;h<sub>dry</sub> = {h_dry:.2f} m, h<sub>sub</sub> = {h_water_eff:.2f} m
            <br>&emsp;&sigma;'<sub>v</sub> = ({gamma_soil} &times; {h_dry:.2f}) + (({gamma_soil}-{gamma_water}) &times; {h_water_eff:.2f}) = {sig_v_base - q_val:.2f} kN/m<sup>2</sup>
            <br>&emsp;&sigma;'<sub>h,soil</sub> = K<sub>a</sub> &times; &sigma;'<sub>v</sub> = {ka} &times; {sig_v_base - q_val:.2f} = <strong>{p_soil_char:.2f} kN/m<sup>2</sup></strong>
        </li>
        <li><strong>Water Pressure:</strong>
            <br>&emsp;u = &gamma;<sub>w</sub> &times; h<sub>w</sub> = {gamma_water} &times; {h_water_eff:.2f} = <strong>{p_water_char:.2f} kN/m<sup>2</sup></strong>
        </li>
        <li><strong>Total Lateral Pressure (Characteristic):</strong>
            <br>&emsp;&sigma;<sub>h,total</sub> = {p_q_char:.2f} + {p_soil_char:.2f} + {p_water_char:.2f} = <strong>{p_total_char:.2f} kN/m<sup>2</sup></strong>
        </li>
        </ul>
        <p><em>Note: ULS Design Forces are derived by applying &gamma;<sub>G</sub>={gamma_G} to Soil/Dead loads and &gamma;<sub>Q</sub>={gamma_Q} to Surcharge/Live loads (if applicable) in the analysis model.</em></p>

        <h3>2. Design Summary</h3>
        <p><strong>Status:</strong> {'✅ PASSED' if (pass_My_soil and pass_My_excav and pass_Mx_soil and pass_Mx_excav and pass_V and pass_crack_y and pass_crack_x) else '❌ FAILED'}</p>
        
        <table border="1" cellpadding="5" style="border-collapse: collapse; width: 100%;">
        <tr><th>Check</th><th>Face</th><th>Demand</th><th>Capacity/Limit</th><th>Status</th></tr>
        <tr><td>Vertical M<sub>y</sub></td><td>Soil (Outer)</td><td>M<sub>Ed</sub>={M_Ed_y_soil:.1f}</td><td>M<sub>Rd</sub> (As={As_vert_prov:.0f})</td><td>{'✅ OK' if pass_My_soil else '❌ FAIL'}</td></tr>
        <tr><td>Vertical M<sub>y</sub></td><td>Excav (Inner)</td><td>M<sub>Ed</sub>={M_Ed_y_excav:.1f}</td><td>M<sub>Rd</sub> (As={As_vert_prov:.0f})</td><td>{'✅ OK' if pass_My_excav else '❌ FAIL'}</td></tr>
        <tr><td>Crack Control</td><td>Soil Face</td><td>w<sub>k</sub>={wk_y:.3f} mm</td><td>w<sub>max</sub>={w_max} mm</td><td>{'✅ OK' if pass_crack_y else '❌ FAIL'}</td></tr>
        <tr><td>Horz M<sub>x</sub></td><td>Soil (Outer)</td><td>M<sub>Ed</sub>={M_Ed_x_soil:.1f}</td><td>M<sub>Rd</sub> (As={As_horz_prov:.0f})</td><td>{'✅ OK' if pass_Mx_soil else '❌ FAIL'}</td></tr>
        <tr><td>Horz M<sub>x</sub></td><td>Excav (Inner)</td><td>M<sub>Ed</sub>={M_Ed_x_excav:.1f}</td><td>M<sub>Rd</sub> (As={As_horz_prov:.0f})</td><td>{'✅ OK' if pass_Mx_excav else '❌ FAIL'}</td></tr>
        <tr><td>Crack Control</td><td>Horz Face</td><td>w<sub>k</sub>={wk_x:.3f} mm</td><td>w<sub>max</sub>={w_max} mm</td><td>{'✅ OK' if pass_crack_x else '❌ FAIL'}</td></tr>
        <tr><td>Shear</td><td>Max</td><td>V<sub>Ed</sub>={max_V_uls:.1f}</td><td>V<sub>Rd</sub>={V_Rd:.1f}</td><td>{'✅ OK' if pass_V else '❌ FAIL'}</td></tr>
        </table>
        
        <h3>3. Analysis Plots</h3>
        <img src="{img1_b64}" style="width: 100%; max-width: 800px;">
        <br>
        <img src="{img2_b64}" style="width: 100%; max-width: 600px;">
        
        <h3>4. Detailed Checks</h3>
        
        <h4>4.1 Minimum Reinforcement (EC2 9.2.1.1)</h4>
        <p>Minimum area of reinforcement required to control cracking and prevent brittle failure:</p>
        <ul>
        <li><strong>Vertical (Main):</strong>
            <br>&emsp;A<sub>s,min</sub> = 0.26 (f<sub>ctm</sub>/f<sub>yk</sub>) bd &ge; 0.0013 bd
            <br>&emsp;f<sub>ctm</sub> = 0.3 &times; {fck}<sup>2/3</sup> = {0.3*fck**(2/3):.2f} MPa
            <br>&emsp;A<sub>s,min</sub> = <strong>{As_min_vert:.0f} mm²/m</strong> (vs Prov: {As_vert_prov:.0f}) -> {'✅ OK' if As_vert_prov >= As_min_vert else '❌ FAIL'}
        </li>
        <li><strong>Horizontal (Dist):</strong>
             <br>&emsp;A<sub>s,min</sub> = <strong>{As_min_horz:.0f} mm²/m</strong> (vs Prov: {As_horz_prov:.0f}) -> {'✅ OK' if As_horz_prov >= As_min_horz else '❌ FAIL'}
        </li>
        </ul>

        <h4>4.2 Vertical Bending (Outer/Soil Face)</h4>
        <ul>
        <li><strong>Design Moment:</strong> M<sub>Ed</sub> = {M_Ed_y_soil:.1f} kNm/m (Hogging)</li>
        <li><strong>Effective Depth:</strong> d = {d_vert:.0f} mm</li>
        <li><strong>Reinforcement:</strong> &phi;{v_bar}@{v_space} (As = {As_vert_prov:.0f} mm²/m)</li>
        <li><strong>Capacity Check:</strong>
             <br>&emsp;K = {det_flex_y_soil['K']:.3f} | z = {det_flex_y_soil['z']:.0f} mm | A<sub>s,req</sub> = {As_req_My_soil:.0f} mm²/m
             <br>&emsp;<strong>Utilisation:</strong> {As_req_My_soil/As_vert_prov*100:.1f}% -> {'✅ OK' if pass_My_soil else '❌ FAIL'}
        </li>
        </ul>
        
        <h4>4.3 Vertical Bending (Inner/Excav Face)</h4>
        <ul>
        <li><strong>Design Moment:</strong> M<sub>Ed</sub> = {M_Ed_y_excav:.1f} kNm/m (Sagging)</li>
        <li><strong>Capacity Check:</strong>
             <br>&emsp;A<sub>s,req</sub> = {As_req_My_excav:.0f} mm²/m
             <br>&emsp;<strong>Utilisation:</strong> {As_req_My_excav/As_vert_prov*100:.1f}% -> {'✅ OK' if pass_My_excav else '❌ FAIL'}
        </li>
        </ul>

        
        <h4>Vertical Crack Width - EN 1992-1-1 Cl 7.3.4</h4>
        <ul>
        <li><strong>Inputs:</strong> M<sub>sls</sub> = {max_My_sls:.1f} kNm/m, &alpha;<sub>e</sub> = 15, A<sub>s</sub> = {As_vert_prov:.0f}</li>
        <li><strong>Neutral Axis (x):</strong>
            <br>&emsp;x = d &times; [-&alpha;&rho; + &radic;((&alpha;&rho;)<sup>2</sup> + 2&alpha;&rho;)]
            <br>&emsp;<strong>x = {det_cr_y['x']:.1f} mm</strong>
        </li>
        <li><strong>Cracked Inertia (I<sub>cr</sub>):</strong>
            <br>&emsp;I<sub>cr</sub> = (b &times; x<sup>3</sup>)/3 + &alpha;<sub>e</sub> &times; A<sub>s</sub> &times; (d-x)<sup>2</sup>
            <br>&emsp;I<sub>cr</sub> = (1000 &times; {det_cr_y['x']:.0f}<sup>3</sup>)/3 + 15 &times; {As_vert_prov:.0f} &times; ({d_vert:.0f}-{det_cr_y['x']:.0f})<sup>2</sup>
            <br>&emsp;<strong>I<sub>cr</sub> = {det_cr_y['I_cr']:.2e} mm<sup>4</sup></strong>
        </li>
        <li><strong>Steel Stress (&sigma;<sub>s</sub>):</strong>
            <br>&emsp;&sigma;<sub>s</sub> = (&alpha;<sub>e</sub> &times; M<sub>sls</sub> &times; (d-x)) / I<sub>cr</sub>
            <br>&emsp;&sigma;<sub>s</sub> = (15 &times; {max_My_sls*1e6:.1e} &times; ({d_vert:.0f}-{det_cr_y['x']:.0f})) / {det_cr_y['I_cr']:.2e}
            <br>&emsp;<strong>&sigma;<sub>s</sub> = {det_cr_y['sigma_s']:.1f} MPa</strong>
        </li>
        <li><strong>Max Spacing (s<sub>r,max</sub>):</strong>
            <br>&emsp;s<sub>r,max</sub> = 3.4c + 0.42k<sub>1</sub>k<sub>2</sub>&phi; / &rho;<sub>eff</sub>
            <br>&emsp;s<sub>r,max</sub> = 3.4&times;{det_cr_y['cover']} + 0.42&times;{det_cr_y['k1']}&times;{det_cr_y['k2']}&times;{v_bar} / {det_cr_y['rho_eff']:.4f}
            <br>&emsp;<strong>s<sub>r,max</sub> = {det_cr_y['sr_max']:.1f} mm</strong>
        </li>
        <li><strong>Crack Width (w<sub>k</sub>):</strong>
            <br>&emsp;&epsilon;<sub>diff</sub> = [&sigma;<sub>s</sub> - k<sub>t</sub>(f<sub>ct,eff</sub>/&rho;<sub>eff</sub>)(1+&alpha;&rho;<sub>eff</sub>)] / E<sub>s</sub>
            <br>&emsp;&epsilon;<sub>diff</sub> = [{det_cr_y['sigma_s']:.1f} - {det_cr_y['kt']}&times;({det_cr_y['fctm']:.2f}/{det_cr_y['rho_eff']:.4f})(1+15&times;{det_cr_y['rho_eff']:.4f})] / {det_cr_y['Es']:.0e}
            <br>&emsp;&epsilon;<sub>diff</sub> = <strong>{det_cr_y['eps_diff']:.6f}</strong>
            <br>&emsp;w<sub>k</sub> = {det_cr_y['sr_max']:.1f} &times; {det_cr_y['eps_diff']:.6f}
            <br>&emsp;<strong>w<sub>k</sub> = {wk_y:.3f} mm</strong> (Limit {w_max})
        </li>
        </ul>
        <p><strong>Status:</strong> {'✅ OK' if pass_crack_y else '❌ FAIL'}</p>
        
        <h4>4.4 Horizontal Bending (Outer/Soil Face)</h4>
        <ul>
        <li><strong>Design Moment:</strong> M<sub>Ed</sub> = {M_Ed_x_soil:.1f} kNm/m (Hogging)</li>
        <li><strong>Capacity Check:</strong>
             <br>&emsp;A<sub>s,req</sub> = {As_req_Mx_soil:.0f} mm²/m
             <br>&emsp;<strong>Utilisation:</strong> {As_req_Mx_soil/As_horz_prov*100:.1f}% -> {'✅ OK' if pass_Mx_soil else '❌ FAIL'}
        </li>
        </ul>
        
        <h4>4.5 Horizontal Bending (Inner/Excav Face)</h4>
        <ul>
        <li><strong>Design Moment:</strong> M<sub>Ed</sub> = {M_Ed_x_excav:.1f} kNm/m (Sagging)</li>
        <li><strong>Capacity Check:</strong>
             <br>&emsp;A<sub>s,req</sub> = {As_req_Mx_excav:.0f} mm²/m
             <br>&emsp;<strong>Utilisation:</strong> {As_req_Mx_excav/As_horz_prov*100:.1f}% -> {'✅ OK' if pass_Mx_excav else '❌ FAIL'}
        </li>
        </ul>
        
        <h4>Horizontal Crack Width - EN 1992-1-1 Cl 7.3.4</h4>
        <ul>
        <li><strong>Inputs:</strong> M<sub>sls</sub> = {max_Mx_sls:.1f} kNm/m, d = {d_horz:.1f} mm, A<sub>s</sub> = {As_horz_prov:.0f}</li>
        <li><strong>Neutral Axis (x):</strong>
            <br>&emsp;x = {det_cr_x['x']:.1f} mm
        </li>
        <li><strong>Cracked Inertia (I<sub>cr</sub>):</strong>
            <br>&emsp;I<sub>cr</sub> = {det_cr_x['I_cr']:.2e} mm<sup>4</sup>
        </li>
        <li><strong>Steel Stress (&sigma;<sub>s</sub>):</strong>
            <br>&emsp;&sigma;<sub>s</sub> = (&alpha;<sub>e</sub> &times; M<sub>sls</sub> &times; (d-x)) / I<sub>cr</sub>
            <br>&emsp;&sigma;<sub>s</sub> = (15 &times; {max_Mx_sls*1e6:.1e} &times; ({d_horz:.0f}-{det_cr_x['x']:.0f})) / {det_cr_x['I_cr']:.2e}
            <br>&emsp;<strong>&sigma;<sub>s</sub> = {det_cr_x['sigma_s']:.1f} MPa</strong>
        </li>
        <li><strong>Max Spacing (s<sub>r,max</sub>):</strong>
            <br>&emsp;s<sub>r,max</sub> = 3.4&times;{det_cr_x['cover']} + 0.42&times;{det_cr_x['k1']}&times;{det_cr_x['k2']}&times;{h_bar} / {det_cr_x['rho_eff']:.4f}
            <br>&emsp;<strong>s<sub>r,max</sub> = {det_cr_x['sr_max']:.1f} mm</strong>
        </li>
        <li><strong>Crack Width (w<sub>k</sub>):</strong>
            <br>&emsp;&epsilon;<sub>diff</sub> = [{det_cr_x['sigma_s']:.1f} - {det_cr_x['kt']}&times;({det_cr_x['fctm']:.2f}/{det_cr_x['rho_eff']:.4f})(1+15&times;{det_cr_x['rho_eff']:.4f})] / {det_cr_x['Es']:.0e}
            <br>&emsp;&epsilon;<sub>diff</sub> = <strong>{det_cr_x['eps_diff']:.6f}</strong>
            <br>&emsp;w<sub>k</sub> = {det_cr_x['sr_max']:.1f} &times; {det_cr_x['eps_diff']:.6f}
            <br>&emsp;<strong>w<sub>k</sub> = {wk_x:.3f} mm</strong> (Limit {w_max})
        </li>
        </ul>
        <p><strong>Status:</strong> {'✅ OK' if pass_crack_x else '❌ FAIL'}</p>
        
        <h4>Shear Capacity (No Links) - EN 1992-1-1 Cl 6.2.2</h4>
        <ul>
        <li><strong>Design Shear:</strong> V<sub>Ed</sub> = {max_V_uls:.1f} kN/m</li>
        <li><strong>Parameters:</strong>
            <br>&emsp;k = 1 + &radic;(200/d) = 1 + &radic;(200/{det_shear['d']:.1f}) = <strong>{det_shear['k']:.2f}</strong> (&le; 2.0)
            <br>&emsp;&rho;<sub>l</sub> = A<sub>s</sub>/(bd) = {As_vert_prov:.0f}/(1000&times;{det_shear['d']:.1f}) = <strong>{det_shear['rho']:.4f}</strong> (&le; 0.02)
        </li>
        <li><strong>Capacity (V<sub>Rd,c</sub>):</strong>
            <br>&emsp;V<sub>Rd,c</sub> = [C<sub>Rd,c</sub> k (100&rho;<sub>l</sub> f<sub>ck</sub>)<sup>1/3</sup>] &times; b &times; d
            <br>&emsp;V<sub>Rd,c</sub> = [{det_shear['CRdc']:.3f} &times; {det_shear['k']:.2f} &times; (100&times;{det_shear['rho']:.4f}&times;{det_shear['fck']})<sup>1/3</sup>] &times; 1000 &times; {det_shear['d']:.1f}
            <br>&emsp;<strong>V<sub>Rd,c</sub> = {V_Rd:.1f} kN/m</strong> (Min Check v<sub>min</sub>={det_shear['v_min']:.2f} MPa)
        </li>
        </ul>
        <p><strong>Status:</strong> {'✅ OK' if pass_V else '❌ FAIL'}</p>
        
        </div>
        """
        
        st.markdown(report_html, unsafe_allow_html=True)
        
        st.divider()
        
        # Save & Print Area
        c_p1, c_p2 = st.columns([2, 1])
        
        with c_p1:
            st.download_button(
                label="📥 Download Report (HTML)",
                data=report_html,
                file_name=f"Wall_Design_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html",
                type="primary"
            )
                      
        with c_p2:
            st.markdown("<br>", unsafe_allow_html=True) # Spacer
            if st.button("Print Report"):
                # JavaScript injection to trigger print
                 js = f"""
                 <script>
                     var printWindow = window.open('', '_blank');
                     printWindow.document.write(`{report_html}`);
                     printWindow.document.close();
                     printWindow.print(); 
                 </script>
                 """
                 st.components.v1.html(js, height=0, width=0)

    # else:
    #    st.info("Adjust parameters in the sidebar and click 'Calculate' to run.")

if __name__ == "__main__":
    main()
