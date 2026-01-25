import streamlit as st
import pandas as pd
import os
import json
import io
import datetime
import math
import base64
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Page Config ---
st.set_page_config(
    page_title="Pile Cap Design Calculator",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load Custom CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Ensure assets directory exists and css file is there
# We assume the style.css is already present in the workspace/assets from the template
css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    local_css(css_path)
else:
    # Fallback CSS if file not found (Template requirement: strict adherence, so we should try to match)
    st.markdown("""
    <style>
    .main-header {
        font-family: 'Roboto', sans-serif;
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #0066cc;
    }
    .main-header h1 {
        color: #0066cc;
        margin: 0;
        font-size: 28px;
        font-weight: 700;
    }
    .main-header p {
        color: #666;
        margin: 5px 0 0;
        font-size: 14px;
    }
    .css-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header Section ---
def render_header():
    st.markdown("""
        <div class="main-header">
            <h1>Pile Cap Design Calculator</h1>
            <p>Professional Design Template</p>
        </div>
    """, unsafe_allow_html=True)

render_header()

# --- Helper Functions ---

def draw_pile_cap_diagram(pile_df, cx, cy, cap_params, col_params):
    """
    Draws the plan view of the pile cap.
    """
    Lx = cap_params['Lx']
    Ly = cap_params['Ly']
    dp = cap_params['dp']
    pile_shape = cap_params.get('pile_shape', 'Circular')
    off_x = cap_params.get('off_x', 0)
    off_y = cap_params.get('off_y', 0)
    shape = cap_params.get('shape', 'Rectangle')
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 1. Pile Cap Boundary
    if shape == 'Triangular' and len(pile_df) == 3:
        # Standard Triangular Cap for 3 piles
        # We use an offset-based polygon that bounds the piles + edge_dist
        # Simplification: Use the piles coordinates directly as vertices, offset outwards?
        # Actually, let's just make it a triangle that bounds the piles with min_edge_dist
        piles = pile_df.to_dict('records')
        # Sort piles to ensure consistent poly
        # Vertex 1: Top, 2: Bottom Right, 3: Bottom Left
        verts = []
        edge_dist = cap_params.get('edge_dist', 150)
        
        # We calculate the vertices of the triangular cap
        # This is a bit complex for arbitrary 3-pile, but for equilateral:
        # Vertex i = Pile i + vector pointing away from centroid
        for _, p in pile_df.iterrows():
            px, py = p['x'], p['y']
            # Direction from 0,0 to pile
            mag = (px**2 + py**2)**0.5
            if mag > 0:
                vx = px / mag
                vy = py / mag
            else:
                vx, vy = 0, 1 # fallback
            
            # Additional logic for vertices: they are corners of the triangular cap
            # To strictly bound circles, we need to extend the triangle edges.
            # Simplified for UI: Use a slightly enlarged triangle
            # dist = dp/2 + edge_dist. We need to project this along the bisectors.
            ext = (dp/2 + edge_dist) / math.sin(30 * math.pi/180) # for equilateral
            verts.append((px + vx*ext, py + vy*ext))
            
        poly = patches.Polygon(verts, closed=True, linewidth=2, edgecolor='#333', facecolor='#f0f2f6', label='Pile Cap')
        ax.add_patch(poly)
    else:
        # 1. Pile Cap Boundary (Concrete Rectangle)
        # Shifted by off_x, off_y
        rect_cap = patches.Rectangle((-Lx/2 + off_x, -Ly/2 + off_y), Lx, Ly, linewidth=2, edgecolor='#333', facecolor='#f0f2f6', label='Pile Cap')
        ax.add_patch(rect_cap)
    
    # 2. Column
    rect_col = patches.Rectangle((-cx/2, -cy/2), cx, cy, linewidth=2, edgecolor='#333', facecolor='#ddd', hatch='///', label='Column')
    ax.add_patch(rect_col)
    ax.text(0, 0, f"Col\n{int(cx)}x{int(cy)}", ha='center', va='center', fontsize=9, fontweight='bold', color='#333', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # 3. Piles
    for i, row in pile_df.iterrows():
        if pile_shape == 'Square':
            # Rectangle centered at (x,y) with width/height dp
            pile_patch = patches.Rectangle((row['x'] - dp/2, row['y'] - dp/2), dp, dp, edgecolor='#333', facecolor='#fff', linewidth=1.5)
        else:
            pile_patch = patches.Circle((row['x'], row['y']), radius=dp/2, edgecolor='#333', facecolor='#fff', linewidth=1.5)
        ax.add_patch(pile_patch)
        ax.text(row['x'], row['y'], f"P{int(row.name)+1}", ha='center', va='center', fontsize=8, color='blue')
    
    # --- Dimensions ---
    # --- Helper for Dimensions ---
    def draw_dim(p1, p2, text, offset=0, text_offset=0, vertical=False):
        # p1, p2 are tuples (x, y) on the element edge
        # offset is distance from element to dim line
        # text_offset is distance from dim line to text
        
        if vertical:
            x_dim = p1[0] - offset
            # Draw line
            ax.annotate('', xy=(x_dim, p1[1]), xytext=(x_dim, p2[1]),
                        arrowprops=dict(arrowstyle='|-|', color='gray', lw=0.8, shrinkA=0, shrinkB=0))
            # Draw text
            mid_y = (p1[1] + p2[1]) / 2
            ax.text(x_dim - text_offset, mid_y, text, ha='right', va='center', rotation=90, 
                    fontsize=8, color='gray', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))
        else:
            y_dim = p1[1] - offset
            # Draw line
            ax.annotate('', xy=(p1[0], y_dim), xytext=(p2[0], y_dim),
                        arrowprops=dict(arrowstyle='|-|', color='gray', lw=0.8, shrinkA=0, shrinkB=0))
            # Draw text
            mid_x = (p1[0] + p2[0]) / 2
            ax.text(mid_x, y_dim - text_offset, text, ha='center', va='top', 
                    fontsize=8, color='gray', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))

    # --- Dimensions ---
    # Horizontal Lx (Overall)
    draw_dim((-Lx/2 + off_x, -Ly/2 + off_y), (Lx/2 + off_x, -Ly/2 + off_y), f"Lx={int(Lx)}", offset=350, text_offset=10)
    
    # Vertical Ly (Overall)
    draw_dim((-Lx/2 + off_x, -Ly/2 + off_y), (-Lx/2 + off_x, Ly/2 + off_y), f"Ly={int(Ly)}", offset=350, text_offset=10, vertical=True)
    
    # --- Detailed Dimensions ---
    # Find unique X and Y coordinates
    unique_x = sorted(pile_df['x'].unique())
    unique_y = sorted(pile_df['y'].unique())
    
    # 1. Horizontal Chain (Bottom) - Offset 150
    dim_off_det = 150
    edge_left = -Lx/2 + off_x
    edge_right = Lx/2 + off_x
    
    if unique_x:
        # Edge to First
        if abs(unique_x[0] - edge_left) > 10: # Only if distinct
             draw_dim((edge_left, -Ly/2 + off_y), (unique_x[0], -Ly/2 + off_y), f"{int(unique_x[0] - edge_left)}", offset=dim_off_det, text_offset=5)
        
        # Inter-Pile
        for i in range(len(unique_x)-1):
            draw_dim((unique_x[i], -Ly/2 + off_y), (unique_x[i+1], -Ly/2 + off_y), f"{int(unique_x[i+1]-unique_x[i])}", offset=dim_off_det, text_offset=5)
            
        # Last to Edge
        if abs(edge_right - unique_x[-1]) > 10:
             draw_dim((unique_x[-1], -Ly/2 + off_y), (edge_right, -Ly/2 + off_y), f"{int(edge_right - unique_x[-1])}", offset=dim_off_det, text_offset=5)

    # 2. Vertical Chain (Left) - Offset 150
    edge_bot = -Ly/2 + off_y
    edge_top = Ly/2 + off_y
    
    if unique_y:
        # Edge to First
        if abs(unique_y[0] - edge_bot) > 10:
             draw_dim((-Lx/2 + off_x, edge_bot), (-Lx/2 + off_x, unique_y[0]), f"{int(unique_y[0] - edge_bot)}", offset=dim_off_det, text_offset=5, vertical=True)
             
        # Inter-Pile
        for i in range(len(unique_y)-1):
            draw_dim((-Lx/2 + off_x, unique_y[i]), (-Lx/2 + off_x, unique_y[i+1]), f"{int(unique_y[i+1]-unique_y[i])}", offset=dim_off_det, text_offset=5, vertical=True)
            
        # Last to Edge
        if abs(edge_top - unique_y[-1]) > 10:
             draw_dim((-Lx/2 + off_x, unique_y[-1]), (-Lx/2 + off_x, edge_top), f"{int(edge_top - unique_y[-1])}", offset=dim_off_det, text_offset=5, vertical=True)

    # Pile Dia (Label first pile)
    if not pile_df.empty:
        p0 = pile_df.iloc[0]
        lbl = "w=" if pile_shape == 'Square' else "dp="
        ax.annotate(f"{lbl}{int(dp)}", xy=(p0['x'] + dp/2, p0['y']), xytext=(p0['x'] + dp/2 + 200, p0['y'] + 200),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=0.5), fontsize=8, color='blue')

    # --- Axes & Load Directions ---
    # Draw X-Y Axes (dashed)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(Lx/2 + off_x + 100, 0, "X", color='gray', fontsize=8, fontweight='bold', va='center')
    ax.text(0, Ly/2 + off_y + 100, "Y", color='gray', fontsize=8, fontweight='bold', ha='center')

    # Vx, Vy Arrows (at Column)
    ax.annotate('', xy=(cx/2 + 300, 0), xytext=(cx/2 + 50, 0), arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.text(cx/2 + 350, 0, "Vx", color='red', fontsize=9, fontweight='bold', va='center')
    
    ax.annotate('', xy=(0, cy/2 + 300), xytext=(0, cy/2 + 50), arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.text(0, cy/2 + 350, "Vy", color='red', fontsize=9, fontweight='bold', ha='center')
    
    # Mx, My Curved Arrows (Rotating ABOUT the axis)
    # Mx: Rotation ABOUT the X-axis (causes tension/comp along Y)
    # Shown as a single-headed arc centered on X-axis
    ax.annotate("", xy=(cx/2 + 200, 200), xytext=(cx/2 + 200, -200),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-1.2", color='darkred', lw=1.5))
    ax.text(cx/2 + 350, 100, "Mx", color='darkred', fontsize=9, fontweight='bold')
    
    # My: Rotation ABOUT the Y-axis (causes tension/comp along X)
    # Shown as a single-headed arc centered on Y-axis
    ax.annotate("", xy=(200, cy/2 + 200), xytext=(-200, cy/2 + 200),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-1.2", color='darkred', lw=1.5))
    ax.text(100, cy/2 + 350, "My", color='darkred', fontsize=9, fontweight='bold')

    # Dimensions
    # Auto-adjust limits
    ax.set_xlim(-Lx/2 + off_x - 600, Lx/2 + off_x + 600)
    ax.set_ylim(-Ly/2 + off_y - 600, Ly/2 + off_y + 600)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Pile Cap Plan View ({shape})", fontsize=10)
    
    return fig

def save_state():
    """Serialize session state to JSON string."""
    keys_to_save = [
        "project_title", "project_number", "designer", "design_date",
        "concrete_class", "fyk",
        "cx", "cy",
        "N_Ed", "V_x", "V_y", "M_x", "M_y",
        "h_cap", "cover",
        "dp", "pile_cap_comp", "pile_cap_tens", "pile_shape",
        "d_bot_x", "s_bot_x", "d_bot_y", "s_bot_y",
        "pile_coords", # This needs special handling as it's a DF usually, but we'll adapt
        "Lx_input", "Ly_input", "min_edge_dist", "cap_shape"
    ]
    
    state_data = {}
    for k in keys_to_save:
        if k in st.session_state:
            val = st.session_state[k]
            if isinstance(val, pd.DataFrame):
                state_data[k] = val.to_dict(orient='records')
            else:
                state_data[k] = val

    def json_serial(obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    return json.dumps(state_data, default=json_serial)

def load_state(uploaded_file):
    """Load state from JSON file."""
    if uploaded_file is not None:
        data = json.load(uploaded_file)
        for k, v in data.items():
            if k == 'design_date' and isinstance(v, str):
                try:
                    st.session_state[k] = datetime.date.fromisoformat(v)
                except ValueError:
                    st.session_state[k] = v
            elif k == 'pile_coords':
                st.session_state[k] = pd.DataFrame(v)
            else:
                st.session_state[k] = v

# Try to import tkinter for local file dialogs
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

# --- Application Layout ---
def main():
    # Sidebar
    with st.sidebar:
        # App Logo
        st.markdown("""
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="color: #0066cc; font-size: 28px; margin: 0; font-weight: 800;">C&S <span style="font-weight: 300;">Calc Pro</span></h1>
                <p style="font-size: 12px; color: #666; margin-top: 5px;">Structural Design Suite</p>
            </div>
            <hr style="margin-top: 0; margin-bottom: 20px;">
        """, unsafe_allow_html=True)

        st.header("Project Details")
        st.text_input("Project Title", value="New Project", key="project_title")
        st.text_input("Project Number", value="PC-2024-001", key="project_number")
        st.text_input("Designer", value="Engineer", key="designer")
        st.date_input("Date", key="design_date")
        
        st.markdown("---")
        st.header("Design Inputs")
        
        # 1. Material
        st.markdown("##### Material Properties")
        concrete_options = {
            "C20/25": 20, "C25/30": 25, "C30/37": 30, "C32/40": 32,
            "C35/45": 35, "C40/50": 40, "C45/55": 45, "C50/60": 50
        }
        selected_class = st.selectbox("Concrete Class (EC2)", options=list(concrete_options.keys()), index=2, key="concrete_class")
        fck = concrete_options[selected_class]
        st.info(f"Selected fck = {fck} MPa")
        fyk = st.number_input("Steel Yield Strength fyk (MPa)", value=500, step=10, key="fyk")
        
        # 2. Geometry
        st.markdown("##### Column & Cap Geometry")
        c1, c2 = st.columns(2)
        with c1:
            cx = st.number_input("Col Width cx (mm)", value=400, step=50, key="cx")
            h_cap = st.number_input("Cap Depth h (mm)", value=800, step=50, key="h_cap")
        with c2:
            cy = st.number_input("Col Depth cy (mm)", value=400, step=50, key="cy")
            cover = st.number_input("Cover (mm)", value=50, step=5, key="cover") # Higher cover for foundations

        d_avg = h_cap - cover - 20 # Approximation for initial calc
        st.caption(f"Approx d = {d_avg} mm")

        # 3. Piles (Flexible Input)
        st.markdown("##### Pile Configuration")
        
        # Moved Pile Data Here for Logic Flow
        st.caption("Pile Properties")
        r1_c1, r1_c2 = st.columns(2)
        dp = r1_c1.number_input("Pile Size/Dia (mm)", value=600, step=50, key="dp")
        pile_shape = r1_c2.selectbox("Pile Shape", ["Circular", "Square"], key="pile_shape")
        
        r2_c1, r2_c2 = st.columns(2)
        pile_cap_comp = r2_c1.number_input("Pile Cap. Comp (kN)", value=3000.0, step=100.0, key="pile_cap_comp")
        pile_cap_tens = r2_c2.number_input("Pile Cap. Tens (kN)", value=200.0, step=50.0, key="pile_cap_tens")
        
        # Auto-Scale Logic
        if 'last_dp' not in st.session_state:
            st.session_state.last_dp = dp
        
        if dp != st.session_state.last_dp:
            scale_ratio = dp / st.session_state.last_dp
            if 'pile_coords' in st.session_state and not st.session_state.pile_coords.empty:
                st.session_state.pile_coords['x'] = (st.session_state.pile_coords['x'] * scale_ratio).round(0)
                st.session_state.pile_coords['y'] = (st.session_state.pile_coords['y'] * scale_ratio).round(0)
                st.toast(f"Piles scaled by {scale_ratio:.2f}x to maintain spacing ratio")
            st.session_state.last_dp = dp
            st.rerun()

        st.markdown("---")
        min_edge_dist = st.number_input("Min Edge Dist (mm)", value=150, step=25, key="min_edge_dist")
        
        # Default Pile Data
        default_piles = pd.DataFrame([
            {'x': -900, 'y': -900},
            {'x': 900, 'y': -900}, # Updated default to match 3*600 = 1800 spacing
            {'x': -900, 'y': 900},
            {'x': 900, 'y': 900}
        ])
        
        if 'pile_coords' not in st.session_state:
            st.session_state.pile_coords = default_piles
            
        # Pattern Generator
        with st.expander("🧩 Pile Pattern Generator"):
            gen_col1, gen_col2, gen_col3 = st.columns([1, 1, 1])
            with gen_col1:
                gen_n = st.number_input("No. of Piles", min_value=2, max_value=9, value=4, step=1, key="gen_n")
            with gen_col2:
                # Auto-update logic: If dp changes, we might want to update this?
                # Simplest way: default value is 3*dp. 
                # If key is constant, Streamlit keeps old value unless we force update.
                # Use a specific key dependent on dp? No, that resets it too often.
                # Just set value=3*dp. Streamlit will use this on first load.
                # Usage: User changes dp -> re-runs -> value=... 
                # BUT Streamlit preserves widget state unless we manually update it.
                # To force update when dp changes, we need to check state.
                
                default_sp = float(3*dp)
                if f"sp_last_dp_{dp}" not in st.session_state:
                     st.session_state.gen_spacing = default_sp
                     st.session_state[f"sp_last_dp_{dp}"] = True
                     
                gen_spacing = st.number_input("Spacing c/c (mm)", value=default_sp, step=50.0, key="gen_spacing")
            with gen_col3:
                st.write("") # Spacer
                if st.button("Generate Group"):
                    # Standard Patterns
                    coords = []
                    s = gen_spacing
                    
                    if gen_n == 2:
                        coords = [{'x': -s/2, 'y': 0}, {'x': s/2, 'y': 0}]
                    elif gen_n == 3:
                        h = s * math.sqrt(3)/2
                        coords = [{'x': -s/2, 'y': -h/3}, {'x': s/2, 'y': -h/3}, {'x': 0, 'y': 2*h/3}]
                    elif gen_n == 4:
                        coords = [{'x': -s/2, 'y': -s/2}, {'x': s/2, 'y': -s/2}, 
                                  {'x': -s/2, 'y': s/2}, {'x': s/2, 'y': s/2}]
                    elif gen_n == 5:
                        coords = [{'x': -s*0.707, 'y': -s*0.707}, {'x': s*0.707, 'y': -s*0.707}, 
                                  {'x': -s*0.707, 'y': s*0.707}, {'x': s*0.707, 'y': s*0.707}, {'x': 0, 'y': 0}]
                        # Or classic sq 4 + 1 center? If we keep corner at same pos as 4, Spacing between C and corner is reduced.
                        # User wants 3*dia spacing.
                        # If center is present, closest neighbor is corner. Dist = sqrt(x^2+y^2).
                        # We need Dist >= s. So Corner >= s/sqrt(2) approx 0.707s.
                        # We use 0.707s for valid 5-pile
                    elif gen_n == 6:
                        coords = [{'x': -s/2, 'y': -s}, {'x': s/2, 'y': -s},
                                  {'x': -s/2, 'y': 0}, {'x': s/2, 'y': 0},
                                  {'x': -s/2, 'y': s}, {'x': s/2, 'y': s}]
                    elif gen_n == 7: # Hexagonal
                        coords = [{'x': 0, 'y': 0}]
                        for i in range(6):
                            angle = i * 60 * math.pi / 180
                            coords.append({'x': s*math.cos(angle), 'y': s*math.sin(angle)})
                    elif gen_n == 8:
                         coords = [{'x': -1.5*s, 'y': -s/2}, {'x': -0.5*s, 'y': -s/2}, {'x': 0.5*s, 'y': -s/2}, {'x': 1.5*s, 'y': -s/2},
                                   {'x': -1.5*s, 'y': s/2}, {'x': -0.5*s, 'y': s/2}, {'x': 0.5*s, 'y': s/2}, {'x': 1.5*s, 'y': s/2}]
                    elif gen_n == 9:
                        coords = []
                        for i in [-1, 0, 1]:
                            for j in [-1, 0, 1]:
                                coords.append({'x': i*s, 'y': j*s})
                                
                    st.session_state.pile_coords = pd.DataFrame(coords)
                    st.rerun()

        st.markdown("**Pile Coordinates (from Center)**")
        
        # Prepare Data for Editor (Add Label Column)
        # We need to manage the DataFrame carefully. 
        # If we just display it, edits return a new DF.
        # We want to allow adding/deleting rows directly in the editor.
        
        # We won't add the "Pile Not" text column as a fixed part of the editor data 
        # because it makes adding new rows complicated (user has to type "P5").
        # Instead, we'll let the index guide them or just show X/Y.
        
        df_editor_input = st.session_state.pile_coords.copy()
        # Add labels as Index for display
        df_editor_input.index = [f"P{i+1}" for i in range(len(df_editor_input))]
        
        edited_df = st.data_editor(
            df_editor_input, 
            key="pile_editor",
            column_config={
                "_index": st.column_config.Column("Pile No", disabled=True),
                "x": st.column_config.NumberColumn("x (mm)", format="%d", required=True),
                "y": st.column_config.NumberColumn("y (mm)", format="%d", required=True),
            },
            num_rows="dynamic",
            use_container_width=True
        )
        
        # Immediate Sync
        # We need to strip the index before saving, as session state expects simple integer index or no index concern
        # When user adds a row, the index might be weird, so we reset_index(drop=True)
        edited_df_clean = edited_df.reset_index(drop=True)
        
        if not edited_df_clean.equals(st.session_state.pile_coords):
            st.session_state.pile_coords = edited_df_clean
            st.rerun()
            
        edited_piles = st.session_state.pile_coords
        
        # Smart Tools (Optional Helper)
        with st.expander("🛠️ Smart Tools (optional)"):
             b_col1, b_col2 = st.columns([1, 1])
             with b_col1:
                 if st.button("➕ Smart Add Pile"):
                     # Smart Placement Logic
                     spacing = 3 * dp 
                     new_x, new_y = 0, 0
                     
                     if not edited_piles.empty:
                         # Candidates
                         candidates = [(0, 0)] # Center is Prio 1
                         for _, p in edited_piles.iterrows():
                             px, py = p['x'], p['y']
                             candidates.extend([(px+spacing, py), (px-spacing, py), (px, py+spacing), (px, py-spacing)])
                         
                         # Sort candidates by distance from Center
                         candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
                         
                         found = False
                         for cx_cand, cy_cand in candidates:
                             collision = False
                             for _, p in edited_piles.iterrows():
                                 dist = ((cx_cand - p['x'])**2 + (cy_cand - p['y'])**2)**0.5
                                 if dist < 10.0: # Identical overlap check only
                                     collision = True
                                     break
                             
                             if not collision:
                                 new_x, new_y = cx_cand, cy_cand
                                 found = True
                                 break
                                 
                         if not found: # Should not happen with candidates
                              last = edited_piles.iloc[-1]
                              new_x, new_y = last['x'] + spacing, last['y']
     
                     new_pile = pd.DataFrame([{'x': new_x, 'y': new_y}])
                     edited_piles = pd.concat([edited_piles, new_pile], ignore_index=True)
                     st.session_state.pile_coords = edited_piles
                     st.rerun() 
                 
             with b_col2:
                 if st.button("🗑️ Remove Last Pile"):
                     if not edited_piles.empty:
                         edited_piles = edited_piles.iloc[:-1]
                         st.session_state.pile_coords = edited_piles
                         st.rerun()

        # Sync back to session state
        st.session_state.pile_coords = edited_piles
        
        st.subheader("Pile Data")
        st.caption("(Moved to top of section)")

        # Check Spacing Validation
        if not edited_piles.empty and len(edited_piles) > 1:
            min_s_found = float('inf')
            closest_pair = ""
            piles_arr = edited_piles.to_dict('records')
            for i in range(len(piles_arr)):
                for j in range(i+1, len(piles_arr)):
                    p1 = piles_arr[i]
                    p2 = piles_arr[j]
                    dist = ((p1['x']-p2['x'])**2 + (p1['y']-p2['y'])**2)**0.5
                    if dist < min_s_found:
                        min_s_found = dist
                        closest_pair = f"P{i+1}-P{j+1}"
            
            if min_s_found < 3 * dp - 1.0: # Tolerance
                 st.warning(f"⚠️ Spacing Warning: {closest_pair} dist {min_s_found:.0f}mm < {3*dp:.0f}mm (3xDia)")
        
        # Calculate Bounding Box for Cap Dimensions
        if not edited_piles.empty:
            min_x, max_x = edited_piles['x'].min(), edited_piles['x'].max()
            min_y, max_y = edited_piles['y'].min(), edited_piles['y'].max()
            
            # Auto-size cap: Extent + Pile Dia/2 + min_edge_dist
            # We want ensuring at least min_edge_dist from pile edge
            # Extent of pile edge = max(|x|) + dp/2
            # Req Cap Dim = 2 * (Max Extent + min_edge_dist)
            
            # Correct Tight Bounding:
            max_p_x = edited_piles['x'].max()
            min_p_x = edited_piles['x'].min()
            max_p_y = edited_piles['y'].max()
            min_p_y = edited_piles['y'].min()
            
            req_Lx = (max_p_x - min_p_x) + dp + 2 * min_edge_dist
            req_Ly = (max_p_y - min_p_y) + dp + 2 * min_edge_dist
            
            st.metric("Min Cap Size (Tight Fit)", f"{req_Lx:.0f} x {req_Ly:.0f} mm")
            
            # Auto-Resize Logic
            auto_resize = st.checkbox("Auto-Resize Cap to Min Edge Dist", value=True, key="auto_resize_cap")
            
            if auto_resize:
                Lx = req_Lx
                Ly = req_Ly
                # Update session state to match
                st.session_state.Lx_input = float(req_Lx)
                st.session_state.Ly_input = float(req_Ly)
            else:
                Lx = st.session_state.get('Lx_input', float(req_Lx))
                Ly = st.session_state.get('Ly_input', float(req_Ly))
            
        else:
            Lx, Ly = 1000.0, 1000.0
            auto_resize = False
            
        # Let's add Manual Cap Dimension inputs allowing override
        st.markdown("**Cap Configuration**")
        cap_shape = st.selectbox("Cap Shape", ["Rectangle", "Triangular"], index=0, key="cap_shape")
        
        dim_c1, dim_c2 = st.columns(2)
        with dim_c1:
            Lx = st.number_input("Cap Length Lx (mm)", value=float(Lx), step=50.0, key="Lx_input", disabled=auto_resize)
        with dim_c2:
            Ly = st.number_input("Cap Width Ly (mm)", value=float(Ly), step=50.0, key="Ly_input", disabled=auto_resize)
            
        # Calculate Offsets (Center of cap relative to column 0,0)
        off_x, off_y = 0.0, 0.0
        if not edited_piles.empty:
            min_x_p, max_x_p = edited_piles['x'].min(), edited_piles['x'].max()
            min_y_p, max_y_p = edited_piles['y'].min(), edited_piles['y'].max()
            off_x = (min_x_p + max_x_p) / 2
            off_y = (min_y_p + max_y_p) / 2
            
        # Now we can check edge distance
        edge_warn = []
        if not edited_piles.empty:
            for i, row in edited_piles.iterrows():
                # Distance from center to Pile Edge
                # Handle rectangle logic check
                dist_x = (Lx/2) - abs(row['x'] - off_x) - dp/2
                dist_y = (Ly/2) - abs(row['y'] - off_y) - dp/2
                
                if dist_x < -1.0 or dist_y < -1.0: # Outside
                     edge_warn.append(f"Pile {i+1} (Outside)")
                elif dist_x < min_edge_dist or dist_y < min_edge_dist:
                    edge_warn.append(f"Pile {i+1}")
        
        if edge_warn:
            st.warning(f"⚠️ Edge Distance Warning: {', '.join(edge_warn)} are closer than {min_edge_dist}mm to the edge!")


        # Pile Capacity moved up

        # 4. Loading
        st.markdown("##### Design Loads (ULS)")
        N_Ed = st.number_input("Axial Load N_Ed (kN)", value=5000.0, step=100.0, key="N_Ed")
        
        l1, l2 = st.columns(2)
        with l1:
            M_x = st.number_input("Moment Mx (kNm)", value=200.0, step=10.0, key="M_x")
            V_y = st.number_input("Shear Vy (kN)", value=100.0, step=10.0, key="V_y")
        with l2:
            M_y = st.number_input("Moment My (kNm)", value=100.0, step=10.0, key="M_y")
            V_x = st.number_input("Shear Vx (kN)", value=50.0, step=10.0, key="V_x")

        # 5. Reinforcement
        st.markdown("##### Reinforcement")
        st.caption("Provide details for both directions")
        
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**X-Direction (Bot)**")
            d_bot_x = st.selectbox("Bar Dia X (mm)", [12,16,20,25,32,40], index=3, key="d_bot_x")
            s_bot_x = st.number_input("Spacing X (mm)", value=150, step=25, key="s_bot_x")
        with r2:
            st.markdown("**Y-Direction (Bot)**")
            d_bot_y = st.selectbox("Bar Dia Y (mm)", [12,16,20,25,32,40], index=3, key="d_bot_y")
            s_bot_y = st.number_input("Spacing Y (mm)", value=150, step=25, key="s_bot_y")
            
    # Pile Area
    if pile_shape == "Square":
        Ap = (dp/1000.0)**2
    else:
        Ap = math.pi * (dp/1000.0)**2 / 4
        
    st.sidebar.markdown("---")
    st.sidebar.header("File Operations")
    if HAS_TKINTER:
        if st.button("💾 Save Project"):
            try:
                root = tk.Tk(); root.withdraw(); root.wm_attributes('-topmost', 1)
                file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")], initialfile="pilecap_proj.json")
                root.destroy()
                if file_path:
                    with open(file_path, "w") as f: f.write(save_state())
                    st.success(f"Saved to {file_path}")
            except Exception as e: st.error(f"Save failed: {e}")
    else:
            st.sidebar.download_button("💾 Save Project", save_state(), "pilecap_proj.json", "application/json")
    
    uploaded_file = st.sidebar.file_uploader("📂 Load Project", type=["json"])
    if uploaded_file is not None:
            st.sidebar.button("Confirm Load", on_click=load_state, args=(uploaded_file,))

    # Main Content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Design Calculations")
        tabs = st.tabs(["Calculation", "Detailed Report"])
        
        with tabs[0]:
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            st.write("### Pile Reaction Analysis")
            
            # --- Logic ---
            if edited_piles.empty:
                st.error("Please add piles.")
            else:
                n_piles = len(edited_piles)
                
                # Check Centroid
                # If pile group is not centered, we should technically account for eccentricity of N_Ed
                # For this tool, we assume N is applied at (0,0) and handle moments Mx, My
                
                # Rigid Cap Method
                # Ixx = Sum y^2, Iyy = Sum x^2
                Ixx = (edited_piles['y']**2).sum()
                Iyy = (edited_piles['x']**2).sum()
                
                # Pile Forces
                # P = N/n +/- My*x/Iyy +/- Mx*y/Ixx
                # Sign convention: 
                # Mx: Moment about X-axis. Positive creates comp in +y?? 
                # Let's assume Right Hand Rule. Mx vector along X. +Mx compresses +y (Top). 
                # Actually standard structural:
                # P_i = N/n + (Mx * y_i / Sum y^2) + (My * x_i / Sum x^2)
                # Note units: M in kNm, coords in mm. Consistent units: M -> Nmm, coords -> mm. Res -> N. /1000 -> kN
                
                P_axial = N_Ed / n_piles
                
                # Calculate forces
                # We need to being careful with signs.
                # Usually Mx vector is along X axis. +Mx puts +Y in compression? Or Tension?
                # Let's display coords and forces table
                
                pile_res = []
                for idx, row in edited_piles.iterrows():
                    x = row['x']
                    y = row['y']
                    
                    term_My = (M_y * 1e6 * x) / Iyy if Iyy != 0 else 0
                    term_Mx = (M_x * 1e6 * y) / Ixx if Ixx != 0 else 0
                    
                    # Superposition (Signs depend on direction of M)
                    # Let's assume inputs are Design Moments creating Copmression in + quadrants
                    # So +My pushes +X, +Mx pushes +Y
                    P_val = P_axial + (term_My/1000.0) + (term_Mx/1000.0)
                    
                    status = "OK"
                    if P_val > pile_cap_comp: status = "Overload (Comp)"
                    if P_val < -pile_cap_tens: status = "Overload (Tens)"
                    
                    pile_res.append({
                        "Pile": idx+1,
                        "x": x, "y": y,
                        "Force (kN)": round(P_val, 1),
                        "Status": status
                    })
                
                res_df = pd.DataFrame(pile_res)
                st.dataframe(res_df, use_container_width=True)
                
                max_P = res_df['Force (kN)'].max()
                min_P = res_df['Force (kN)'].min()
                
                # Checks
                pile_check = max_P <= pile_cap_comp and min_P >= -pile_cap_tens
                if pile_check:
                    st.success(f"Pile Capacity OK. Max Load: {max_P} kN")
                else:
                    st.error(f"Pile Capacity Exceeded! Max: {max_P} kN")

                st.markdown("---")
                st.write("### Flexural Design (Bottom Reinforcement)")
                
                # Flexure Logic
                # Calculate Moment at face of column
                # Critical section at face of column
                
                # Mx Design (Bending about X axis -> Y reinforcement) - Wait, usually:
                # Moment about Y-axis (My) => varying x => Requires X-reinforcement (As_x) to resist
                # Let's stick to: M_xx = Moment causing bending in X-direction (around Y axis)
                
                # Calculate Moments at column faces
                # Foreach pile i:
                # If x_i > cx/2: Lever arm lx = x_i - cx/2. Moment += P_i * lx
                # If x_i < -cx/2: Lever arm lx = |x_i| - cx/2. Moment += P_i * lx
                
                M_design_x = 0.0 # Creates tension in x-bars (bending in x-plane)
                M_design_y = 0.0 # Creates tension in y-bars
                
                for idx, row in edited_piles.iterrows():
                    P_pile = res_df.loc[idx, 'Force (kN)']
                    x = row['x']
                    y = row['y']
                    
                    # Only calculate if Pile is in compression (upward reaction on cap). 
                    # If tension, it pulls down, helping? Conservatively ignore tension piles for main steel or treat appropriately.
                    # Usually pile caps are designed for upward soil reaction.
                    if P_pile > 0:
                        # X-Direction Moment (about Y axis section)
                        lx = abs(x) - cx/2
                        if lx > 0:
                            M_design_x += P_pile * (lx / 1000.0) # kN * m = kNm
                            
                        # Y-Direction Moment
                        ly = abs(y) - cy/2
                        if ly > 0:
                            M_design_y += P_pile * (ly / 1000.0)
                
                # Only take moments from one half?
                # The total moment is the sum of moments from all piles on one side of the section? 
                # NO. We need to check the Critical Section. 
                # For a symm cap, check section at +cx/2 (Sum P * (x - cx/2) for all x > cx/2)
                # And check section at -cx/2. Take Max.
                
                # Refined Moment Calc
                # Section Right (+X face): Sum P_i * (x_i - cx/2) for all x_i > cx/2
                M_x_pos = sum([r['Force (kN)'] * (r['x'] - cx/2)/1000.0 for i, r in res_df.iterrows() if r['x'] > cx/2])
                M_x_neg = sum([r['Force (kN)'] * (abs(r['x']) - cx/2)/1000.0 for i, r in res_df.iterrows() if r['x'] < -cx/2])
                M_Ed_x = max(M_x_pos, M_x_neg)
                
                # Section Top (+Y face): Sum P_i * (y_i - cy/2) for all y_i > cy/2
                M_y_pos = sum([r['Force (kN)'] * (r['y'] - cy/2)/1000.0 for i, r in res_df.iterrows() if r['y'] > cy/2])
                M_y_neg = sum([r['Force (kN)'] * (abs(r['y']) - cy/2)/1000.0 for i, r in res_df.iterrows() if r['y'] < -cy/2])
                M_Ed_y = max(M_y_pos, M_y_neg)
                
                c_calc1, c_calc2 = st.columns(2)
                c_calc1.metric("Design Moment M_Ed,x", f"{M_Ed_x:.1f} kNm")
                c_calc2.metric("Design Moment M_Ed,y", f"{M_Ed_y:.1f} kNm")
                
                # As Req Calc (EC2)
                # d_x = h - cover - d_bot_x/2
                # d_y = h - cover - d_bot_x - d_bot_y/2 (Stacking)
                d_x = h_cap - cover - d_bot_x/2
                d_y = h_cap - cover - d_bot_x - d_bot_y/2
                
                def calc_As(M, d, b_width):
                    if M <= 0: return 0
                    K = (M * 1e6) / (b_width * d**2 * fck)
                    if K > 0.167: z = 0.5 * d # Simplified
                    else: z = d * (0.5 + (0.25 - K/1.134)**0.5)
                    z = min(z, 0.95*d)
                    fyd = fyk / 1.15
                    return (M * 1e6) / (fyd * z)
                
                As_req_x = calc_As(M_Ed_x, d_x, Ly) # Width is Ly for X-bending
                As_req_y = calc_As(M_Ed_y, d_y, Lx) # Width is Lx for Y-bending
                
                # As Prov
                # Spacing s_bot_x -> Area per m = 1000/s * A_bar
                # Total Width = Ly. Total Bars = Ly/s + 1
                n_bars_x = math.floor(Ly / s_bot_x) + 1
                As_prov_x = n_bars_x * (math.pi * d_bot_x**2 / 4)
                
                n_bars_y = math.floor(Lx / s_bot_y) + 1
                As_prov_y = n_bars_y * (math.pi * d_bot_y**2 / 4)
                
                st.write(f"**X-Direction Reinforcement ({n_bars_x} H{d_bot_x})**")
                st.markdown(f"As_req: {As_req_x:.0f} mm² | As_prov: {As_prov_x:.0f} mm² " + 
                           ("✅" if As_prov_x >= As_req_x else "❌"))
                           
                st.write(f"**Y-Direction Reinforcement ({n_bars_y} H{d_bot_y})**")
                st.markdown(f"As_req: {As_req_y:.0f} mm² | As_prov: {As_prov_y:.0f} mm² " + 
                           ("✅" if As_prov_y >= As_req_y else "❌"))

                flexure_ok = (As_prov_x >= As_req_x) and (As_prov_y >= As_req_y)

                # --- Enhanced Shear Check (EC2 Cl 6.2.2(6)) ---
                st.markdown("---")
                st.write("### Shear Check (Enhanced Resistance @ Face)")
                st.caption("Using $\\beta = a_v / 2d$ reduction for piles within $2.0d$ of column face.")

                def check_enhanced_shear(side_df, d_val, b_width, face_coord, is_x=True):
                    # V_Ed,adj = Sum(P_i * beta_i)
                    V_Ed_adj = 0
                    piles_info = []
                    
                    for i, r in side_df.iterrows():
                        p_val = r['Force (kN)']
                        if p_val <= 0: continue
                        
                        # av = distance from col face to pile face (Edge-to-Edge)
                        # EC2 6.2.2(6): Distance to edge of load
                        dist_center = abs(r['x' if is_x else 'y'] - face_coord)
                        av = dist_center - (dp / 2)
                        
                        # Constraint: av must be >= 0 (overlap treated as 0)
                        if av < 0: av = 0

                        # Detailed info string
                        coord_val = r['x' if is_x else 'y']
                        calc_str = f"|{coord_val:.0f} - {face_coord:.0f}| - {dp/2:.0f}"
                        
                        # EC2 6.2.2(6) enhancement factor beta = av / 2d
                        beta = min(max(av / (2 * d_val), 0.25), 1.0)
                        
                        V_Ed_adj += p_val * beta
                        piles_info.append(f"P{i+1}: $a_v = {calc_str} = {av:.0f}$ mm $\\rightarrow \\beta = {av:.0f}/(2 \\cdot {d_val:.0f}) = {beta:.2f}$")
                    
                    # Resistance V_Rd,c
                    k_size = min(1 + (200/d_val)**0.5, 2.0)
                    rho = min( (As_prov_x if is_x else As_prov_y) / (b_width * d_val), 0.02)
                    v_min = 0.035 * k_size**1.5 * fck**0.5
                    v_rdc_base = max(0.12 * k_size * (100 * rho * fck)**(1/3), v_min)
                    V_Rdc = v_rdc_base * b_width * d_val / 1000.0
                    
                    return V_Ed_adj, V_Rdc, piles_info

                # Section Right (+X face)
                df_right = res_df[res_df['x'] > cx/2]
                V_Ed_x_adj, V_Rdc_x, info_x = check_enhanced_shear(df_right, d_x, Ly, cx/2, True)
                
                # Section Top (+Y face)
                df_top = res_df[res_df['y'] > cy/2]
                V_Ed_y_adj, V_Rdc_y, info_y = check_enhanced_shear(df_top, d_y, Lx, cy/2, False)

                sc1, sc2 = st.columns(2)
                with sc1:
                    st.metric("Adj. Shear V_Ed,x", f"{V_Ed_x_adj:.1f} kN")
                    st.metric("Capacity V_Rd,c,x", f"{V_Rdc_x:.1f} kN", delta="OK" if V_Rdc_x >= V_Ed_x_adj else "Fail")
                    if info_x: st.info("\n".join(info_x))
                
                with sc2:
                    st.metric("Adj. Shear V_Ed,y", f"{V_Ed_y_adj:.1f} kN")
                    st.metric("Capacity V_Rd,c,y", f"{V_Rdc_y:.1f} kN", delta="OK" if V_Rdc_y >= V_Ed_y_adj else "Fail")
                    if info_y: st.info("\n".join(info_y))

                shear_ok = (V_Rdc_x >= V_Ed_x_adj) and (V_Rdc_y >= V_Ed_y_adj)
                
            st.markdown('</div>', unsafe_allow_html=True)

        with tabs[1]:
            st.markdown("### Detailed Design Report")
            
            # Generate Figure
            fig_rep = draw_pile_cap_diagram(edited_piles, cx, cy, {'Lx': Lx, 'Ly': Ly, 'dp': dp, 'pile_shape': pile_shape, 'off_x': off_x, 'off_y': off_y, 'shape': cap_shape, 'edge_dist': min_edge_dist}, {})
            buf = io.BytesIO()
            fig_rep.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode()
            
            # HTML Template (Adapted)
             # CSS for A4 Printed Look
            html_style = """
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
                @import url('https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.css');
                .katex-display { text-align: left !important; margin-left: 0 !important; }
                body { font-family: 'Roboto', sans-serif; color: #333; max-width: 210mm; margin: 0 auto; padding: 20px; }
                header { border-bottom: 2px solid #0066cc; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; }
                h2 { color: #0066cc; font-size: 18px; border-left: 5px solid #0066cc; padding-left: 10px; background: #f8f9fa; padding: 8px 10px; margin-top: 30px;}
                h3 { font-size: 16px; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 20px;}
                table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                table th, table td { border: 1px solid #eee; padding: 6px; text-align: left; font-size: 14px; }
                table th { background: #f8f9fa; }
                .status-pass { color: green; font-weight: bold; }
                .status-fail { color: red; font-weight: bold; }
                .formula-box { background: #fdfdfd; border: 1px solid #eee; padding: 8px; border-left: 3px solid #ddd; font-family: 'Times New Roman'; }
            </style>
            """
            
            # Create Moment Breakdown Strings
            # X-Direction (About Y-axis)
            if M_x_pos >= M_x_neg:
                m_x_terms = [f"{r['Force (kN)']} \\cdot {(r['x'] - cx/2)/1000.0:.3f}" for i, r in res_df.iterrows() if r['x'] > cx/2]
            else:
                m_x_terms = [f"{r['Force (kN)']} \\cdot {(abs(r['x']) - cx/2)/1000.0:.3f}" for i, r in res_df.iterrows() if r['x'] < -cx/2]
            M_Ed_x_str = " + ".join(m_x_terms) if m_x_terms else "0"

            # Y-Direction (About X-axis)
            if M_y_pos >= M_y_neg:
                m_y_terms = [f"{r['Force (kN)']} \\cdot {(r['y'] - cy/2)/1000.0:.3f}" for i, r in res_df.iterrows() if r['y'] > cy/2]
            else:
                m_y_terms = [f"{r['Force (kN)']} \\cdot {(abs(r['y']) - cy/2)/1000.0:.3f}" for i, r in res_df.iterrows() if r['y'] < -cy/2]
            M_Ed_y_str = " + ".join(m_y_terms) if m_y_terms else "0"

            report_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Pile Cap Design Report</title>
                {html_style}
                <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">
                <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.js"></script>
                <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/contrib/auto-render.min.js"></script>
                <script>
                    document.addEventListener("DOMContentLoaded", function() {{
                        renderMathInElement(document.body, {{
                            delimiters: [ {{left: "$$", right: "$$", display: true}}, {{left: "$", right: "$", display: false}} ]
                        }});
                    }});
                </script>
            </head>
            <body>
                <header>
                    <div>
                        <h1 style="color: #0066cc; margin:0;">Pile Cap Design Calculation</h1>
                        <p style="margin:0;">Eurocode 2 & 7 Compliant</p>
                    </div>
                    <div style="text-align:right;">
                        <strong>Date:</strong> {st.session_state.get('design_date', 'N/A')}<br>
                        <strong>Ref:</strong> {st.session_state.get('project_number', 'N/A')}
                    </div>
                </header>
                
                <div style="text-align: center; margin-bottom: 20px;">
                    <img src="data:image/png;base64,{img_b64}" style="max-width: 60%; border: 1px solid #eee; padding: 10px;">
                    <p>Figure 1: Pile Cap Layout</p>
                </div>
                
                <h2>1. Design Basis & Materials</h2>
                <table>
                    <tr><th>Parameter</th><th>Formula / Ref</th><th>Value</th></tr>
                    <tr><td>Concrete Class</td><td>EN 1992-1-1 Table 3.1</td><td>{selected_class} ($f_{{ck}}$ = {fck} MPa)</td></tr>
                    <tr><td>Pile Shape</td><td>-</td><td>{pile_shape}</td></tr>
                    <tr><td>Pile Size</td><td>-</td><td>{dp} mm</td></tr>
                    <tr><td>Design Steel Strength</td><td>$f_{{yd}} = f_{{yk}} / \\gamma_S$</td><td>{fyk/1.15:.1f} MPa</td></tr>
                </table>

                <h2>2. Pile Reaction Analysis (Rigid Cap Method)</h2>
                <div class="formula-box">
                    General Formula: $$P_i = \\frac{{N_{{Ed}}}}{{n}} \\pm \\frac{{M_x \\cdot y_i}}{{\\sum y^2}} \\pm \\frac{{M_y \\cdot x_i}}{{\\sum x^2}}$$
                </div>
                <h3>Geometric Properties</h3>
                <p>Second Moment of Area of Pile Group:</p>
                <p>$\\sum x^2 = {' + '.join([f'({r:.2f})^2' for r in edited_piles['x']/1000])} = {Iyy/1e6:.4f} \\text{{ m}}^2$</p>
                <p>$\\sum y^2 = {' + '.join([f'({r:.2f})^2' for r in edited_piles['y']/1000])} = {Ixx/1e6:.4f} \\text{{ m}}^2$</p>
                
                <h3>Pile Forces Calculation</h3>
                <table>
                    <thead><tr><th>Pile No.</th><th>$x$ (mm)</th><th>$y$ (mm)</th><th>$N/n$ (kN)</th><th>$M_x y / \\sum y^2$ (kN)</th><th>$M_y x / \\sum x^2$ (kN)</th><th>Total $P_i$ (kN)</th></tr></thead>
                    <tbody>
            """
            
            for _, r in res_df.iterrows():
                tx = (M_y * 1e6 * r['x']) / Iyy if Iyy != 0 else 0
                ty = (M_x * 1e6 * r['y']) / Ixx if Ixx != 0 else 0
                report_html += f"<tr><td>P{int(r['Pile'])}</td><td>{r['x']}</td><td>{r['y']}</td><td>{N_Ed/len(res_df):.1f}</td><td>{ty/1000.0:.1f}</td><td>{tx/1000.0:.1f}</td><td style='font-weight:bold;'>{r['Force (kN)']}</td></tr>"
                
            report_html += f"""
                    </tbody>
                </table>
                <p>Max Pile Load: <strong>{max_P} kN</strong> vs Capacity: <strong>{pile_cap_comp} kN</strong> 
                <span class="{'status-pass' if pile_check else 'status-fail'}">{'OK' if pile_check else 'FAIL'}</span></p>

                <h4>Sample Calculation (Critical Pile P{int(res_df.iloc[res_df['Force (kN)'].argmax()]['Pile'])})</h4>
                <p>Using Pile Coordinates $x = {res_df.iloc[res_df['Force (kN)'].argmax()]['x']:.0f}$ mm, $y = {res_df.iloc[res_df['Force (kN)'].argmax()]['y']:.0f}$ mm:</p>
                $$P_{{i}} = \\frac{{{N_Ed}}}{{{len(edited_piles)}}} + \\frac{{{M_x} \\cdot {res_df.iloc[res_df['Force (kN)'].argmax()]['y']/1000:.2f}}}{{{Ixx/1e6:.4f}}} + \\frac{{{M_y} \\cdot {res_df.iloc[res_df['Force (kN)'].argmax()]['x']/1000:.2f}}}{{{Iyy/1e6:.4f}}}$$
                $$P_{{i}} = {N_Ed/len(edited_piles):.1f} + {((M_x*1e6*res_df.iloc[res_df['Force (kN)'].argmax()]['y'])/Ixx)/1000.0:.1f} + {((M_y*1e6*res_df.iloc[res_df['Force (kN)'].argmax()]['x'])/Iyy)/1000.0:.1f} = {max_P} \\text{{ kN}}$$
                
                <h2>3. Flexural Design (Bottom) (EN 1992-1-1 Cl 6.1)</h2>
                <p>Design for flexure at the column face using the simplified rectangular stress block.</p>
                
                <h3>X-Direction Bending (About Y-Axis)</h3>
                <div class="formula-box">
                    Design Moment: $M_{{Ed,x}} = \\sum P_i \\cdot a_i = {M_Ed_x_str} = {M_Ed_x:.1f} \\text{{ kNm}}$
                </div>
                <p>Effective Depth $d_x = h - c - \\phi/2 = {d_x:.1f} \\text{{ mm}}$</p>
                <p><strong>Bending Parameter K:</strong></p>
                $$K = \\frac{{M_{{Ed}}}}{{b \\cdot d^2 \\cdot f_{{ck}}}} = \\frac{{{M_Ed_x:.1f} \\cdot 10^6}}{{{Ly} \\cdot {d_x:.0f}^2 \\cdot {fck}}} = {(M_Ed_x*1e6)/(Ly*d_x**2*fck) if (Ly*d_x**2*fck)!=0 else 0:.4f}$$
                <p><strong>Lever Arm z:</strong> (limit $z \\le 0.95d$)</p>
                $$z = d \\cdot [0.5 + \\sqrt{{0.25 - K/1.134}}] = {min(d_x*(0.5 + (0.25 - (M_Ed_x*1e6)/(Ly*d_x**2*fck)/1.134)**0.5), 0.95*d_x) if (Ly*d_x**2*fck)!=0 else 0.95*d_x:.1f} \\text{{ mm}}$$
                <p><strong>Required Steel Area:</strong></p>
                $$A_{{s,req}} = \\frac{{M_{{Ed}}}}{{f_{{yd}} \\cdot z}} = \\frac{{{M_Ed_x:.1f} \\cdot 10^6}}{{{fyk/1.15:.1f} \\cdot {min(d_x*(0.5 + (0.25 - (M_Ed_x*1e6)/(Ly*d_x**2*fck)/1.134)**0.5), 0.95*d_x) if (Ly*d_x**2*fck)!=0 else 0.95*d_x:.1f}}} = {As_req_x:.0f} \\text{{ mm}}^2$$
                <p>Provided: {n_bars_x} H{d_bot_x} $\\rightarrow A_{{s,prov}} = {As_prov_x:.0f} \\text{{ mm}}^2$ ({'OK' if As_prov_x >= As_req_x else 'FAIL'})</p>

                <h3>Y-Direction Bending (About X-Axis)</h3>
                <div class="formula-box">
                    Design Moment: $M_{{Ed,y}} = \\sum P_i \\cdot b_i = {M_Ed_y_str} = {M_Ed_y:.1f} \\text{{ kNm}}$
                </div>
                <p>Effective Depth $d_y = {d_y:.1f} \\text{{ mm}}$ | Required $A_{{s,req}} = {As_req_y:.0f} \\text{{ mm}}^2$</p>
                <p>Provided: {n_bars_y} H{d_bot_y} $\\rightarrow A_{{s,prov}} = {As_prov_y:.0f} \\text{{ mm}}^2$ ({'OK' if As_prov_y >= As_req_y else 'FAIL'})</p>

                <h2>4. Shear Verification (EN 1992-1-1 Cl 6.2.2 & 6.2.2(6))</h2>
                <p>Shear enhancement according to EC2 Cl 6.2.2(6) for piles within $2.0d$ of column face.</p>
                <div class="formula-box">
                    Adjusted Design Shear: $V_{{Ed,adj}} = \\sum (P_i \\cdot \\beta)$ where $\\beta = a_v / 2d$ (limit $0.25 \\le \\beta \\le 1.0$)
                </div>

                <h3>Concrete Shear Resistance ($V_{{Rd,c}}$)</h2>
                <p>Ref: EN 1992-1-1 Cl 6.2.2, Eq 6.2.a:</p>
                $$V_{{Rd,c}} = [C_{{Rd,c}} \\cdot k \\cdot (100 \\rho_l f_{{ck}})^{{1/3}}] \\cdot b_w \\cdot d$$
                <p>Substitutions ($\alpha$-direction):</p>
                <ul>
                    <li>$k = 1 + \\sqrt{{200/d}} = {min(1+(200/d_x)**0.5, 2.0):.3f}$ (limit 2.0)</li>
                    <li>$\\rho_l = A_{{sl}} / (b_w d) = {min(As_prov_x/(Ly*d_x), 0.02):.4f}$ (limit 0.02)</li>
                    <li>$C_{{Rd,c}} = 0.18 / \\gamma_c = 0.12$</li>
                </ul>
                <p>Capacity $V_{{Rd,c,x}} = {V_Rdc_x:.1f} \\text{{ kN}}$</p>
                <p>Adjusted Shear $V_{{Ed,adj,x}} = {V_Ed_x_adj:.1f} \\text{{ kN}}$ | Status: <span class="{'status-pass' if V_Rdc_x >= V_Ed_x_adj else 'status-fail'}">{'OK' if V_Rdc_x >= V_Ed_x_adj else 'FAIL'}</span></p>
                <p style="font-size:12px; color:#666;">{'; '.join(info_x)}</p>

                <h3>Y-Direction Shear Check</h2>
                <p>Capacity $V_{{Rd,c,y}} = {V_Rdc_y:.1f} \\text{{ kN}}$ | Adjusted Shear $V_{{Ed,adj,y}} = {V_Ed_y_adj:.1f} \\text{{ kN}}$</p>
                <p>Status: <span class="{'status-pass' if V_Rdc_y >= V_Ed_y_adj else 'status-fail'}">{'OK' if V_Rdc_y >= V_Ed_y_adj else 'FAIL'}</span></p>
                <p style="font-size:12px; color:#666;">{'; '.join(info_y)}</p>

                <footer>Generated by Pile Cap Design Calculator</footer>
            </body>
            </html>
            """
            
            st.info("Report Available.")
            st.download_button("📥 Download PDF Report (HTML)", report_html, "pile_cap_report.html", "text/html")
            
            with st.expander("Preview"):
                st.components.v1.html(report_html, height=600, scrolling=True)

    with col2:
        st.subheader("Visualization")
        st.caption("Plan View")
        st.pyplot(draw_pile_cap_diagram(edited_piles, cx, cy, {
                'Lx': Lx, 'Ly': Ly, 'dp': dp, 
                'pile_shape': pile_shape,
                'off_x': off_x, 'off_y': off_y, 
                'shape': cap_shape, 'edge_dist': min_edge_dist
            }, {}))
        
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.metric("Total Piles", f"{len(edited_piles)}")
        st.metric("Max Pile Load", f"{max_P:.1f} kN")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
