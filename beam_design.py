import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --- PART 1: THE ENGINEERING LOGIC ---
def analyze_beam(span_m, load_kNm):
    """
    Calculates shear and moment arrays for a simply supported beam.
    Returns: x_coordinates, shear_forces, bending_moments, max_moment
    """
    # Create 100 points along the beam (0m to L)
    x = np.linspace(0, span_m, 100)
    
    # Calculate Reactions (Symmetrical UDL)
    reaction = (load_kNm * span_m) / 2
    
    # Shear Force Equation: V(x) = Ra - w*x
    shear = reaction - (load_kNm * x)
    
    # Bending Moment Equation: M(x) = Ra*x - (w*x^2)/2
    moment = (reaction * x) - (load_kNm * x**2 / 2)
    
    max_moment = (load_kNm * span_m**2) / 8
    
    return x, shear, moment, max_moment

# --- PART 2: THE USER INTERFACE ---
st.set_page_config(page_title="Structural Design App")
st.title("🏗️ RC Beam Analysis Tool")

# --- UPDATED INPUTS ---
st.sidebar.header("Geometry & Loading")
span = st.sidebar.number_input("Beam Span (m)", min_value=1.0, value=6.0)
load = st.sidebar.number_input("Ultimate Load (kN/m)", value=15.0)

st.sidebar.header("Material Properties") # <--- NEW
d = st.sidebar.number_input("Effective Depth d (mm)", value=450.0) # <--- NEW
fy = st.sidebar.number_input("Steel Yield fy (MPa)", value=500.0) # <--- NEW

# Run Analysis
x, V, M, M_max = analyze_beam(span, load)

# --- NEW: DESIGN CALCULATION ---
# 1. Convert Moment from kNm to Nmm (multiply by 1,000,000)
M_design_Nmm = M_max * 1e6

# 2. Assume lever arm z = 0.95 * d
z = 0.95 * d

# 3. Calculate As = M / (0.87 * fy * z)
As_req = M_design_Nmm / (0.87 * fy * z)

# --- UPDATED OUTPUT DISPLAY ---
col1, col2, col3 = st.columns(3) # <--- Changed to 3 columns
col1.metric("Max Shear", f"{max(abs(V)):.2f} kN")
col2.metric("Max Moment", f"{M_max:.2f} kNm")
col3.metric("Required Steel (As)", f"{As_req:.0f} mm²") # <--- NEW
# --- PART 4: VISUALIZATION ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Plot Shear
ax1.fill_between(x, V, color='salmon', alpha=0.3)
ax1.plot(x, V, color='red')
ax1.set_ylabel("Shear (kN)")
ax1.grid(True)
ax1.set_title("Shear Force Diagram")

# Plot Moment
ax2.fill_between(x, M, color='lightblue', alpha=0.3)
ax2.plot(x, M, color='blue')
ax2.set_ylabel("Moment (kNm)")
ax2.set_xlabel("Length (m)")
ax2.grid(True)
ax2.set_title("Bending Moment Diagram")

st.pyplot(fig)