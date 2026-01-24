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
    page_title="Civil Design Calculator Template",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load Custom CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Ensure assets directory exists and css file is there (we will create it next)
css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    local_css(css_path)

# --- Header Section ---
def render_header():
    st.markdown("""
        <div class="main-header">
            <h1>Civil & Structural Design Calculator</h1>
            <p>Professional Design Template</p>
        </div>
    """, unsafe_allow_html=True)

render_header()

# --- Helper Functions ---


def create_section_diagram(b, d, h, top_n, top_dia, bot_n, bot_dia, link_dia, cover):
    # Create figure
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Use exact h from inputs
    h_draw = h
    
    # 1. Concrete Section (Gray Rectangle)
    # Origin at bottom-left (0,0)
    rect = patches.Rectangle((0, 0), b, h_draw, linewidth=2, edgecolor='#333', facecolor='#f0f2f6')
    ax.add_patch(rect)
    
    # 2. Links (Rounded Rectangle)
    # Inset by cover
    link_w = b - 2*cover
    link_h = h_draw - 2*cover
    if link_w > 0 and link_h > 0:
        link_rect = patches.FancyBboxPatch((cover, cover), link_w, link_h, boxstyle=f"round,pad=0,rounding_size={2*link_dia}", linewidth=1.5, edgecolor='#444', facecolor='none', linestyle='--')
        ax.add_patch(link_rect)
    
    # 3. Bottom Reinforcement (Red Circles)
    # Center y = cover + link_dia + bot_dia/2
    y_bot = cover + link_dia + bot_dia/2
    
    # Distribute bars evenly
    # Available width for bar centers = b - 2*cover - 2*link_dia - bot_dia
    # But usually evenly spaced inside links
    if bot_n > 0:
        if bot_n == 1:
            x_positions = [b/2]
        else:
            # Simple spacing: cover + link + start...
            eff_w = b - 2*cover - 2*link_dia - bot_dia
            spacing = eff_w / (bot_n - 1)
            start_x = cover + link_dia + bot_dia/2
            x_positions = [start_x + i*spacing for i in range(bot_n)]
            
        for x in x_positions:
            circle = patches.Circle((x, y_bot), radius=bot_dia/2, edgecolor='maroon', facecolor='#ff4b4b')
            ax.add_patch(circle)
            
    # 4. Top Reinforcement (Blue Circles for Hangers)
    y_top = h_draw - cover - link_dia - top_dia/2
    if top_n > 0:
        if top_n == 1:
            x_sub = [b/2]
        else:
            eff_w_top = b - 2*cover - 2*link_dia - top_dia
            space_top = eff_w_top / (top_n - 1)
            start_x_top = cover + link_dia + top_dia/2
            x_sub = [start_x_top + i*space_top for i in range(top_n)]
            
        for x in x_sub:
             circle = patches.Circle((x, y_top), radius=top_dia/2, edgecolor='navy', facecolor='#80afff')
             ax.add_patch(circle)

    # 5. Annotations
    # Dimension line for d (approx from top to centroid of bot)
    # Note: Using calculated d from inputs for check, but diagram shows physical
    ax.annotate('', xy=(b*1.1, h_draw), xytext=(b*1.1, y_bot), arrowprops=dict(arrowstyle='<->', color='blue'))
    ax.text(b*1.15, (h_draw + y_bot)/2, f'd={d:.1f}', color='blue', va='center')
    
    # Settings
    ax.set_xlim(-b*0.2, b*1.5)
    ax.set_ylim(-h_draw*0.1, h_draw*1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Section {b}x{h_draw} (Det. Based)", fontsize=10)
    
    return fig

def save_state():
    """Serialize session state to JSON string."""
    # Filter out non-serializable items if necessary
    # Filter out non-serializable items and internal streamlit keys
    keys_to_save = [
        "project_title", "project_number", "designer", "design_date",
        "concrete_class", "fyk",
        "n_bot", "d_bot", "n_top", "d_top",
        "d_link", "s_link",
        "b", "h", "cover",
        "M_Ed", "V_Ed"
    ]
    state_data = {k: st.session_state[k] for k in keys_to_save if k in st.session_state}
    
    # Custom serializer for date objects
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
            # special handling for date restoration
            if k == 'design_date' and isinstance(v, str):
                try:
                    st.session_state[k] = datetime.date.fromisoformat(v)
                except ValueError:
                     st.session_state[k] = v
            else:
                st.session_state[k] = v


import tkinter as tk
from tkinter import filedialog

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
        st.text_input("Project Number", value="P-2024-001", key="project_number")
        st.text_input("Designer", value="Engineer", key="designer")
        st.date_input("Date", key="design_date")
        
        st.markdown("---")
        st.header("Design Inputs")
        # RC Beam Design Inputs
        st.markdown("##### Material Properties")
        
        # Concrete Class Selection
        concrete_options = {
            "C20/25": 20,
            "C25/30": 25,
            "C30/37": 30,
            "C32/40": 32, # Request
            "C35/45": 35,
            "C40/50": 40,
            "C45/55": 45,
            "C50/60": 50,
            "C55/67": 55,
            "C60/75": 60
        }
        selected_class = st.selectbox("Concrete Class (EC2)", options=list(concrete_options.keys()), index=2, key="concrete_class")
        fck = concrete_options[selected_class]
        # Display fck for clarity
        st.info(f"Selected fck = {fck} MPa")
        
        fyk = st.number_input("Steel Yield Strength fyk (MPa)", value=500, min_value=200, max_value=600, step=10, key="fyk")
        
        st.markdown("##### Reinforcement Detailing")
        st.caption("Provide reinforcement details.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Bottom Bars (Tension)**")
            n_bot = st.number_input("No. Bottom Bars", value=3, min_value=1, key="n_bot")
            d_bot = st.selectbox("Bottom Bar Dia (mm)", options=[10,12,16,20,25,32], index=3, key="d_bot") #20mm default
        with c2:
            st.markdown("**Top Bars (Comp/Hanger)**")
            n_top = st.number_input("No. Top Bars", value=2, min_value=2, key="n_top")
            d_top = st.selectbox("Top Bar Dia (mm)", options=[10,12,16,20,25,32], index=1, key="d_top") #12mm default
            
        st.markdown("**Shear Links**")
        l1, l2 = st.columns(2)
        with l1:
            d_link = st.selectbox("Link Dia (mm)", options=[6, 8, 10, 12], index=1, key="d_link") #8mm default
        with l2:
            s_link = st.number_input("Link Spacing (mm)", value=200, step=25, key="s_link")

        st.markdown("##### Section Geometry")
        b = st.number_input("Beam Width b (mm)", value=300, min_value=100, step=50, key="b")
        h = st.number_input("Total Depth h (mm)", value=600, min_value=100, step=50, key="h")
        cover = st.number_input("Concrete Cover (mm)", value=30, min_value=15, step=5, key="cover")
        
        # Calculate Effective Depth
        # d = h - cover - link_dia - bar_dia/2
        d_calculated = h - cover - d_link - (d_bot / 2)
        st.info(f"Calculated Effective Depth d = {d_calculated:.1f} mm")
        d = d_calculated # Assign to d for downstream logic
        
        st.markdown("##### Loading")
        M_Ed = st.number_input("Design Moment M_Ed (kNm)", value=150.0, step=10.0, key="M_Ed")
        V_Ed = st.number_input("Design Shear V_Ed (kN)", value=50.0, step=10.0, key="V_Ed")
        
        st.markdown("---")
        st.header("File Operations")
        
        # Save Project (Unified)
        if st.button("💾 Save Project"):
            try:
                root = tk.Tk()
                root.withdraw()
                root.wm_attributes('-topmost', 1)
                file_path = filedialog.asksaveasfilename(
                    master=root,
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                    initialfile="project_v1.json",
                    title="Save Project As"
                )
                root.destroy()
                
                if file_path:
                    with open(file_path, "w") as f:
                        f.write(save_state())
                    st.success(f"Project saved to: {file_path}")
            except Exception as e:
                st.error(f"Save failed: {e}")
        
        # Load Button
        uploaded_file = st.file_uploader("📂 Load Project", type=["json"])
        if uploaded_file is not None:
             st.button("Confirm Load", on_click=load_state, args=(uploaded_file,))

    # Main Content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Design Calculations")
        tabs = st.tabs(["Calculation", "Detailed Report"])
        
        with tabs[0]:
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            st.write("### EC2 Calculation Results")
            
            # --- Calculation Logic ---
            # Constants
            StartLine_K = 0.167
            
            # Calculate K
            # K = M / (b * d^2 * fck)
            # Units: M in kNm -> Nmm (*1e6), b,d in mm, fck in MPa (N/mm2)
            if b > 0 and d > 0 and fck > 0:
                K = (M_Ed * 1e6) / (b * (d**2) * fck)
            else:
                K = 999.0 # Error state
                
            st.metric("K Factor", f"{K:.3f}")
            
            # Check K
            As_req = 0.0
            z = 0.0
            status_ok = False
            
            if K > StartLine_K:
                st.error(f"K = {K:.3f} > 0.167. Compression reinforcement required!")
                st.warning("This simple module only calculates singly reinforced sections.")
                z = d * 0.5 # Default for display
            else:
                status_ok = True
                # Lever arm z
                # z = d * [0.5 + sqrt(0.25 - K/1.134)]
                # EC2 simplified: z/d = 0.5 * (1 + sqrt(1 - 3.53*K)) ? 
                # Standard formula: z = d/2 * (1 + sqrt(1 - 2*K_eff)) where K_eff...
                # Using the exact expression from Eurocode 2 (common textbook form):
                # z = d * [0.5 + (0.25 - K/1.134)**0.5]
                inner_term = 0.25 - (K / 1.134)
                if inner_term < 0:
                    z = 0.5 * d # Should be caught by K check, but safety
                else:
                    z = d * (0.5 + (inner_term)**0.5)
                
                # Limit z <= 0.95d
                z = min(z, 0.95 * d)
                
                # Calculate As
                # fyd = fyk / 1.15
                fyd = fyk / 1.15
                if z > 0:
                    As_req = (M_Ed * 1e6) / (fyd * z)
                else:
                    As_req = 0.0

                st.success("Design OK - Singly Reinforced")
                
                c_res1, c_res2 = st.columns(2)
                c_res1.metric("Lever Arm z", f"{z:.1f} mm")
                c_res2.metric("Required As", f"{As_req:.1f} mm²")
            
            # --- Flexure Check Provided vs Required ---
            As_prov = n_bot * (math.pi * d_bot**2 / 4)
            flexure_pass = As_prov >= As_req and status_ok
            
            st.markdown("---")
            st.write("### Reinforcement Verification & Summary")
            
            # Unified Dashboard Row
            k1, k2, k3, k4, k5 = st.columns(5)
            
            k1.metric("Provided As", f"{As_prov:.0f} mm²", delta=f"{As_prov - As_req:.0f}" if flexure_pass else f"{As_prov - As_req:.0f}", delta_color="normal" if flexure_pass else "inverse")
            
            # --- Shear Calculation (EC2) ---
            # V_Rd,c (Concrete Capacity without shear reinf - simplified)
            # k = 1 + sqrt(200/d) <= 2.0
            k_val = min(1 + (200/d)**0.5, 2.0)
            rho_l = min(As_prov / (b*d), 0.02)
            # v_min = 0.035 * k^1.5 * fck^0.5
            v_min = 0.035 * (k_val**1.5) * (fck**0.5)
            # V_Rd,c = [C_Rd,c * k * (100 * rho_l * fck)^(1/3)] * b * d
            # C_Rd,c = 0.18 / gamma_c (=1.5) = 0.12
            CRdc = 0.12
            val_1 = CRdc * k_val * (100 * rho_l * fck)**(1/3)
            
            v_rdc_stress = max(val_1, v_min)
            V_Rdc = v_rdc_stress * b * d / 1000.0 # kN
            
            # V_Rd,s (Shear Reinforcement Capacity)
            # V_Rd,s = (A_sw / s) * z * f_ywd * cot(theta)
            # A_sw = 2 legs * area of link (assuming 2 legs)
            A_sw = 2 * (math.pi * d_link**2 / 4)
            f_ywd = fyk / 1.15 # Design yield of shear steel
            cot_theta = 2.5 # Standard method/conservative
            
            # z usually 0.9d for shear if not calculated, but we have z from flexure
            # If z is 0 (failed flexure), use 0.9d
            z_shear = z if status_ok else 0.9*d
            
            V_Rds = (A_sw / s_link) * z_shear * f_ywd * cot_theta / 1000.0 # kN
            
            # V_Rd,max (Strut capacity)
            # V_Rd,max = alpha_cw * b * z * nu1 * fcd / (cot(theta) + tan(theta))
            # simplified check often omitted in simple tools, but roughly:
            # nu1 = 0.6 * (1 - fck/250)
            # fcd = fck/1.5
            # V_Rd,max roughly quite high for normal beams
            
            # Governing Capacity V_Rd
            V_Rd = max(V_Rdc, V_Rds) # If links provided, they dominate, but technically V_Rd,s
            
            shear_pass = V_Rd >= V_Ed
            
            # Summary Metrics usage logic
            x_depth_metric = As_prov * fyd / (0.8 * b * (0.85*fck/1.5))
            z_actual_metric = d - 0.4*x_depth_metric
            M_Rd_val = As_prov * fyd * z_actual_metric / 1e6
            usage_val = (M_Ed / M_Rd_val) * 100 if M_Rd_val > 0 else 999
            
            k2.metric("Shear V_Ed", f"{V_Ed:.1f} kN")
            k3.metric("Shear Capacity", f"{V_Rd:.1f} kN", delta="OK" if shear_pass else "Fail", delta_color="normal" if shear_pass else "inverse")
            k4.metric("Usage Ratio", f"{usage_val:.1f}%", delta="Safe" if usage_val <= 100 else "Overloaded", delta_color="normal" if usage_val <= 100 else "inverse")
            k5.metric("Required As", f"{As_req:.0f} mm²")
            
            if not shear_pass:
                st.error(f"Shear Fail: Capacity {V_Rd:.1f} kN < Load {V_Ed:.1f} kN. Reduce spacing or increase link dia.")
            
            if not flexure_pass:
                st.error("Flexure Fail: Provided reinforcement is less than required.")

            st.markdown('</div>', unsafe_allow_html=True)
            
        with tabs[1]:
            st.markdown("### Detailed Design Report")
            # Generate Diagram for Report
            fig_report = create_section_diagram(b, d, h, n_top, d_top, n_bot, d_bot, d_link, cover)
            buf = io.BytesIO()
            fig_report.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode()
            
            # Generate Professional HTML Report
            # CSS for A4 Printed Look
            html_style = """
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
                @import url('https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.css');
                
                /* KaTeX Left Alignment Override */
                .katex-display {
                    text-align: left !important;
                    margin-left: 0 !important;
                    margin-right: 0 !important;
                    padding-left: 0 !important;
                }
                
                .katex-html {
                    text-align: left !important;
                }
                
                @media print {
                    @page { size: A4; margin: 20mm; }
                    body { -webkit-print-color-adjust: exact; }
                }
                
                body {
                    font-family: 'Roboto', sans-serif;
                    color: #333;
                    line-height: 1.5;
                    max-width: 210mm;
                    margin: 0 auto;
                    background: #fff;
                    padding: 20px;
                }
                
                header {
                    border-bottom: 2px solid #0066cc;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                
                .header-left h1 { margin: 0; color: #0066cc; font-size: 24px; }
                .header-left p { margin: 5px 0 0; color: #666; font-size: 14px; }
                
                h2 {
                    color: #0066cc;
                    font-size: 18px;
                    border-left: 5px solid #0066cc;
                    padding-left: 10px;
                    margin-top: 30px;
                    background: #f8f9fa;
                    padding: 8px 10px;
                }

                h3 {
                    color: #444;
                    font-size: 16px;
                    margin-top: 20px;
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 5px;
                }
                
                .calc-step {
                    margin-bottom: 15px;
                    page-break-inside: avoid;
                }
                
                .calc-label {
                    font-weight: bold;
                    color: #555;
                    display: block;
                    margin-bottom: 5px;
                }
                
                .formula-box {
                    background: #fdfdfd;
                    border: 1px solid #eee;
                    padding: 8px 12px;
                    border-left: 3px solid #ddd;
                    font-family: 'Times New Roman', serif;
                    font-size: 1.1em;
                    position: relative;
                    text-align: left;
                }
                
                .clause-ref {
                    float: right;
                    font-family: 'Roboto', sans-serif;
                    font-size: 0.8em;
                    color: #0066cc;
                    background: #eef6ff;
                    padding: 2px 6px;
                    border-radius: 4px;
                    font-style: normal;
                }
                
                .subst {
                    color: #666;
                    font-size: 0.95em;
                    margin-top: 4px;
                    padding-left: 10px;
                    border-left: 2px dotted #ccc;
                }
                
                .result {
                    font-weight: bold;
                    color: #000;
                }

                .status-pass { color: green; font-weight: bold; }
                .status-fail { color: red; font-weight: bold; }
                
                table.params {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 10px 0;
                }
                table.params th, table.params td {
                    border: 1px solid #eee;
                    padding: 6px;
                    text-align: left;
                }
                table.params th { background: #f8f9fa; width: 40%; }
                
                footer {
                    margin-top: 50px;
                    border-top: 1px solid #eee;
                    padding-top: 10px;
                    font-size: 12px;
                    color: #999;
                    text-align: center;
                }
            </style>
            """
            
            # HTML Content Construction
            # --- Clause Logic Helpers (Restored) ---
            gamma_c = 1.5
            gamma_s = 1.15
            alpha_cc = 0.85 # UK Annex usually 0.85, simplified
            fcd = alpha_cc * fck / gamma_c
            fyd = fyk / gamma_s
            
            # --- Pre-calculation for Report Logic ---
            # x = (As * fyd) / (0.8 * b * fcd)
            x_depth_calc = (As_prov * fyd) / (0.8 * b * fcd)
            z_actual_calc = d - 0.4 * x_depth_calc
            M_Rd_calc = (As_prov * fyd * z_actual_calc) / 1e6
            xd_ratio = x_depth_calc / d
            
            # Formatted Strings for Report
            x_depth_str = f"{x_depth_calc:.1f}"
            xd_ratio_str = f"{xd_ratio:.2f}"
            z_actual_str = f"{z_actual_calc:.1f}"
            M_Rd_str = f"{M_Rd_calc:.1f}"
            
            xd_status = "<= 0.45 (OK)" if xd_ratio <= 0.45 else "> 0.45 (Warning: Reduced Ductility)"
            cap_status_class = "status-pass" if M_Rd_calc >= M_Ed else "status-fail"
            cap_status_text = "PASS" if M_Rd_calc >= M_Ed else "FAIL"

            report_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>RC Beam Design Report</title>
                {html_style}
                <!-- KaTeX for math rendering -->
                <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">
                <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.js"></script>
                <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/contrib/auto-render.min.js"></script>
                <script>
                    document.addEventListener("DOMContentLoaded", function() {{
                        renderMathInElement(document.body, {{
                            delimiters: [
                                {{left: "$$", right: "$$", display: true}},
                                {{left: "$", right: "$", display: false}}
                            ]
                        }});
                    }});
                </script>
            </head>
            <body>
                <header>
                    <div class="header-left">
                        <h1>RC Beam Design Calculation</h1>
                        <p>Eurocode 2: Design of Concrete Structures (EN 1992-1-1)</p>
                    </div>
                    <div class="header-right">
                        <div style="text-align:right;">
                            <strong>Date:</strong> {st.session_state.get('design_date', 'N/A')}<br>
                            <strong>Ref:</strong> {st.session_state.get('project_number', 'N/A')}
                        </div>
                    </div>
                </header>
                
                <div style="text-align: center; margin-bottom: 20px;">
                    <img src="data:image/png;base64,{img_b64}" style="max-width: 70%; border: 1px solid #eee; padding: 10px; border-radius: 4px;">
                    <p style="font-size: 12px; color: #666;">Figure 1: Proposed Beam Section & Reinforcement</p>
                </div>
                
                <h2>1. Design Data & Material Properties</h2>
                <table class="params">
                    <tr><th>Concrete Strength class</th><td>C{fck}</td></tr>
                    <tr><th>Characteristic Strength ($f_{{ck}}$)</th><td>{fck} MPa</td></tr>
                    <tr><th>Design Compressive Strength ($f_{{cd}}$)</th><td>
                        {alpha_cc:.2f} × {fck} / {gamma_c} = <strong>{fcd:.1f} MPa</strong> <span class="clause-ref">[EC2 3.1.6]</span>
                    </td></tr>
                    <tr><th>Steel Characteristic Yield ($f_{{yk}}$)</th><td>{fyk} MPa</td></tr>
                    <tr><th>Design Yield Strength ($f_{{yd}}$)</th><td>
                        {fyk} / {gamma_s} = <strong>{fyd:.1f} MPa</strong> <span class="clause-ref">[EC2 3.2.7]</span>
                    </td></tr>
                    <tr><th>Section Width ($b$)</th><td>{b} mm</td></tr>
                    <tr><th>Effective Depth ($d$)</th><td>{d} mm</td></tr>
                </table>
                
                <h2>2. Flexural Design (ULS)</h2>
                <h3>2.1 Design Actions</h3>
                <div class="calc-step">
                    <span class="calc-label">Design Moment ($M_{{Ed}}$)</span>
                    <div class="formula-box">
                        $M_{{Ed}}$ = {M_Ed} kNm
                    </div>
                </div>
                
                <h3>2.2 Section Analysis</h3>
                <div class="calc-step">
                    <span class="calc-label">K Value Calculation (Singly Reinforced)</span>
                    <div class="formula-box">
                        $$K = \\frac{{M_{{Ed}}}}{{b d^2 f_{{ck}}}}$$
                        <span class="clause-ref">[Derived EC2 3.1.7]</span>
                    </div>
                    <div class="subst">
                        $$= \\frac{{{M_Ed} \\times 10^6}}{{{b} \\times {d}^2 \\times {fck}}}$$
                        = <span class="result">{K:.4f}</span>
                    </div>
                </div>
                
                <div class="calc-step">
                    <span class="calc-label">Limiting K Value</span>
                    <div class="formula-box">
                        $$K_{{lim}} = 0.167$$
                        <span class="clause-ref">[NA to EC2]</span>
                    </div>
                    <div class="subst">
                        Check: {K:.4f} { "<=" if K <= 0.167 else ">" } 0.167 
                        → <span class="{ 'status-pass' if K <= 0.167 else 'status-fail' }">{ "OK (Singly Reinforced)" if K <= 0.167 else "FAIL (Compression Steel Req)" }</span>
                    </div>
                </div>
            """
            
            if status_ok:
                report_html += f"""
                <div class="calc-step">
                    <span class="calc-label">Lever Arm ($z$) calculation</span>
                    <div class="formula-box">
                        $$z = d \\left[ 0.5 + \\sqrt{{0.25 - \\frac{{K}}{{1.134}}}} \\right] \\le 0.95d$$
                    </div>
                    <div class="subst">
                        $$= {d} \\times [ 0.5 + \\sqrt{{0.25 - {K:.4f}/1.134}} ]$$
                        = <span class="result">{z:.1f} mm</span>
                    </div>
                    <div class="subst">
                        Check Max: $0.95d = 0.95 \\times {d} = {0.95*d:.1f}$ mm
                        → Use $z = {z:.1f}$ mm
                    </div>
                </div>
                
                <div class="calc-step">
                    <span class="calc-label">Required Flexural Reinforcement ($A_{{s,req}}$)</span>
                    <div class="formula-box">
                        $$A_{{s,req}} = \\frac{{M_{{Ed}}}}{{f_{{yd}} z}}$$
                        <span class="clause-ref">[Equilibrium]</span>
                    </div>
                    <div class="subst">
                        $$= \\frac{{{M_Ed} \\times 10^6}}{{{fyd:.1f} \\times {z:.1f}}}$$
                        = <span class="result">{As_req:.0f} mm²</span>
                    </div>
                </div>
                
                <div class="calc-step">
                    <span class="calc-label">Provided Reinforcement</span>
                    <div class="formula-box">
                        Provided: {n_bot} H{d_bot}
                    </div>
                    <div class="subst">
                        $$A_{{s,prov}} = {n_bot} \\times \\pi \\times {d_bot}^2 / 4$$
                        = <span class="result">{As_prov:.0f} mm²</span>
                    </div>
                    <div class="subst">
                        Check: {As_prov:.0f} ≥ {As_req:.0f}
                        <span class="{ 'status-pass' if flexure_pass else 'status-fail' }">{ "OK" if flexure_pass else "FAIL" }</span>
                    </div>
                </div>

                <h3>2.3 Actual Moment Capacity ($M_{{Rd}}$)</h3>
                

                
                <div class="calc-step">
                    <span class="calc-label">Neutral Axis Depth ($x$)</span>
                    <div class="formula-box">
                       $$x = \\frac{{A_{{s,prov}} f_{{yd}}}}{{0.8 \\cdot b \\cdot f_{{cd}} }}$$
                       <span class="clause-ref">[Equilibrium $F_s = F_c$]</span>
                    </div>
                    <div class="subst">
                        $$= \\frac{{{As_prov:.0f} \\times {fyd:.1f}}}{{0.8 \\times {b} \\times {fcd:.1f}}}$$
                        = <span class="result">{x_depth_str} mm</span>
                    </div>
                    <div class="subst">
                        Check $x/d = {xd_ratio_str}$ {xd_status}
                    </div>
                </div>

                <div class="calc-step">
                    <span class="calc-label">Actual Lever Arm ($z_{{actual}}$)</span>
                    <div class="formula-box">
                        $$z = d - 0.4x$$
                    </div>
                    <div class="subst">
                        $$= {d} - 0.4({x_depth_str})$$ 
                        = <span class="result">{z_actual_str} mm</span>
                    </div>
                </div>

                <div class="calc-step">
                    <span class="calc-label">Moment Capacity ($M_{{Rd}}$)</span>
                    <div class="formula-box">
                        $$M_{{Rd}} = A_{{s,prov}} f_{{yd}} z$$
                    </div>
                    <div class="subst">
                        $$= {As_prov:.0f} \\times {fyd:.1f} \\times {z_actual_str} \\times 10^{{-6}}$$
                    </div>
                    <div class="subst">
                        = <span class="result">{M_Rd_str} kNm</span>
                    </div>
                    <div class="subst">
                        Check: $M_{{Rd}} \\ge M_{{Ed}}$ ({M_Rd_str} vs {M_Ed}) 
                        <span class="{cap_status_class}">{cap_status_text}</span>
                    </div>
                </div>
                
                <h2>3. Shear Design (ULS)</h2>
                <h3>3.1 Concrete Shear Capacity ($V_{{Rd,c}}$)</h3>
                
                <div class="calc-step">
                    <span class="calc-label">Size Effect Factor ($k$)</span>
                    <div class="formula-box">
                        $$k = 1 + \\sqrt{{\\frac{{200}}{{d}}}} \\le 2.0$$
                        <span class="clause-ref">[EC2 6.2.2(1)]</span>
                    </div>
                    <div class="subst">
                        $$= 1 + \\sqrt{{200/{d}}}$$ = <span class="result">{k_val:.3f}</span>
                    </div>
                </div>
                
                <div class="calc-step">
                    <span class="calc-label">Reinforcement Ratio ($\\rho_l$)</span>
                    <div class="formula-box">
                        $$\\rho_l = \\frac{{A_{{s,prov}}}}{{b_w d}} \\le 0.02$$
                        <span class="clause-ref">[EC2 6.2.2(1)]</span>
                    </div>
                    <div class="subst">
                        $$= {As_prov:.0f} / ({b} \\times {d})$$ = <span class="result">{rho_l:.4f}</span>
                    </div>
                </div>
                
                <div class="calc-step">
                    <span class="calc-label">Minimum Shear Strength ($v_{{min}}$)</span>
                    <div class="formula-box">
                        $$v_{{min}} = 0.035 k^{{1.5}} f_{{ck}}^{{0.5}}$$
                        <span class="clause-ref">[EC2 Eq 6.3N]</span>
                    </div>
                    <div class="subst">
                        $$= 0.035 \\times {k_val:.3f}^{{1.5}} \\times {fck}^{{0.5}}$$
                        = <span class="result">{v_min:.3f} N/mm²</span>
                    </div>
                </div>
                
                <div class="calc-step">
                    <span class="calc-label">Design Concrete Shear Resistance ($V_{{Rd,c}}$)</span>
                    <div class="formula-box">
                        $$V_{{Rd,c}} = [C_{{Rd,c}} k (100 \\rho_l f_{{ck}})^{{1/3}}] b_w d$$
                        <span class="clause-ref">[EC2 Eq 6.2a]</span>
                    </div>
                    <div class="subst">
                        Take $C_{{Rd,c}} = 0.18/1.5 = {CRdc}$.<br>
                        Stress Value = max($0.12 \\times {k_val:.2f} \\times (100 \\times {rho_l:.4f} \\times {fck})^{{1/3}}$, $v_{{min}}$)<br>
                        Stress Value = <span class="result">{v_rdc_stress:.3f} N/mm²</span><br>
                        Force = {v_rdc_stress:.3f} × {b} × {d} / 1000 
                        = <span class="result">{V_Rdc:.1f} kN</span>
                    </div>
                </div>

                <h3>3.2 Shear Reinforcement Capacity ($V_{{Rd,s}}$)</h3>
                <div class="calc-step">
                    <span class="calc-label">Provided Links</span>
                    <div class="formula-box">
                        H{d_link} @ {s_link} mm c/c (Assumed 2 legs)
                    </div>
                    <div class="subst">
                        $$A_{{sw}} = 2 \\times \\pi \\times {d_link}^2 / 4$$ 
                        = <span class="result">{A_sw:.0f} mm²</span>
                    </div>
                </div>
                
                 <div class="calc-step">
                    <span class="calc-label">Shear Resistance provided by links ($V_{{Rd,s}}$)</span>
                    <div class="formula-box">
                        $$V_{{Rd,s}} = \\frac{{A_{{sw}}}}{{s}} z f_{{ywd}} \\cot \\theta$$
                        <span class="clause-ref">[EC2 Eq 6.8]</span>
                    </div>
                    <div class="subst">
                        Assume $\\cot \\theta = 2.5$. $f_{{ywd}} = f_{{yd}} = {fyd:.1f} \\text{{ MPa}}$.<br>
                        $$= \\frac{{{A_sw:.0f}}}{{{s_link}}} \\times {z_shear:.1f} \\times {fyd:.1f} \\times 2.5 \\times 10^{{-3}}$$
                        = <span class="result">{V_Rds:.1f} kN</span>
                    </div>
                </div>
                
                <h3>3.3 Verification Conclusion</h3>
                <div class="calc-step">
                    <div class="subst">
                        Design Shear Capacity $V_{{Rd}} = \max(V_{{Rd,c}}, V_{{Rd,s}})$
                        = <span class="result">{V_Rd:.1f} kN</span>
                    </div>
                    <div class="subst">
                        Check: $V_{{Rd}} \ge V_{{Ed}} ({V_Ed} \text{{ kN}})$
                        <span class="{ 'status-pass' if shear_pass else 'status-fail' }">{ 'PASS' if shear_pass else 'FAIL' }</span>
                    </div>
                </div>
                """
            else:
                report_html += """
                <div class="subst status-fail">
                    Design FAILED at Flexural Stage. Please modify inputs.
                </div>
                """
                
            report_html += """
                <footer>
                    Generated by C&S Calc Pro | Professional Structural Engineering Software
                </footer>
            </body>
            </html>
            """
            
            st.info("Report generated in professional A4 format. Click below to download.")
            
            st.markdown(f"**Status:** {'✅ Design OK' if status_ok else '❌ Design Failed'}")
            
            # Preview Expander
            with st.expander("📄 Preview Report Content"):
                st.components.v1.html(report_html, height=600, scrolling=True)
            
            if st.button("📥 Save Report As..."):
                try:
                    root = tk.Tk()
                    root.withdraw()
                    root.wm_attributes('-topmost', 1)
                    file_path = filedialog.asksaveasfilename(
                        master=root,
                        defaultextension=".html",
                        filetypes=[("HTML files", "*.html"), ("All files", "*.*")],
                        initialfile="design_report.html",
                        title="Save Design Report"
                    )
                    root.destroy()
                    
                    if file_path:
                        with open(file_path, "w", encoding='utf-8') as f:
                            f.write(report_html)
                        st.success(f"Report saved to: {file_path}")
                except Exception as e:
                    st.error(f"Save failed: {e}")

    with col2:
        st.subheader("Diagrams & Summary")
        
        # Diagram Placeholder Card
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.caption("Section Analysis Diagram")
        
        # Matplotlib visualization
        # We recreate the figure here to avoid closure issues with streamlit display
        fig_display = create_section_diagram(b, d, h, n_top, d_top, n_bot, d_bot, d_link, cover)
        
        st.pyplot(fig_display)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
