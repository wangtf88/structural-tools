import math
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pile Cap Design to Eurocode 2", page_icon="🏗️", layout="wide")


def steel_area(dia_mm: float) -> float:
    return math.pi * dia_mm**2 / 4.0


def pile_coordinates(nx: int, ny: int, sx: float, sy: float):
    xs = [((i - (nx - 1) / 2.0) * sx) for i in range(nx)]
    ys = [((j - (ny - 1) / 2.0) * sy) for j in range(ny)]
    return [(x, y) for y in ys for x in xs]


def beam_theory_design(coords, n_ed, cx, cy, d, fck, fyk, gamma_c=1.5, gamma_s=1.15):
    fyd = fyk / gamma_s
    n_pile = n_ed / len(coords)

    m_ed_x = sum(n_pile * max(abs(x) - cx / 2.0, 0.0) / 1000.0 for x, _ in coords)
    m_ed_y = sum(n_pile * max(abs(y) - cy / 2.0, 0.0) / 1000.0 for _, y in coords)

    z = min(0.95 * d, 0.9 * d)
    as_req_x = (m_ed_x * 1e6) / (fyd * z) if m_ed_x > 0 else 0.0
    as_req_y = (m_ed_y * 1e6) / (fyd * z) if m_ed_y > 0 else 0.0

    v_ed_x = sum(n_pile for x, _ in coords if abs(x) > cx / 2.0)
    v_ed_y = sum(n_pile for _, y in coords if abs(y) > cy / 2.0)

    c_rdc = 0.18 / gamma_c
    k = min(1.0 + math.sqrt(200.0 / d), 2.0)

    def vrdc(b, as_req):
        rho = min(as_req / (b * d), 0.02) if b * d > 0 else 0.0
        vmin = 0.035 * (k ** 1.5) * math.sqrt(fck)
        vrdc_mpa = max(c_rdc * k * ((100.0 * rho * fck) ** (1 / 3)), vmin)
        return vrdc_mpa * b * d / 1000.0

    b_x = max(cy, 1.0)
    b_y = max(cx, 1.0)

    return {
        "n_pile": n_pile,
        "MEd_x": m_ed_x,
        "MEd_y": m_ed_y,
        "As_req_x": as_req_x,
        "As_req_y": as_req_y,
        "VEd_x": v_ed_x,
        "VEd_y": v_ed_y,
        "VRdc_x": vrdc(b_x, as_req_x),
        "VRdc_y": vrdc(b_y, as_req_y),
    }


def strut_tie_design(nx, ny, sx, sy, d, n_ed, fck, fyk, gamma_c=1.5, gamma_s=1.15):
    fcd = 0.85 * fck / gamma_c
    fyd = fyk / gamma_s

    n_piles = nx * ny
    n_pile = n_ed / n_piles

    hx = (nx - 1) * sx / 2.0
    hy = (ny - 1) * sy / 2.0
    r = math.sqrt(hx**2 + hy**2)

    if r <= 0:
        theta = math.pi / 2
        strut_force = n_pile
    else:
        theta = math.atan(d / r)
        strut_force = n_pile / max(math.sin(theta), 1e-9)

    tie_x = n_ed * hx / max(d, 1e-9)
    tie_y = n_ed * hy / max(d, 1e-9)

    as_req_x = tie_x * 1e3 / fyd
    as_req_y = tie_y * 1e3 / fyd

    nu = 0.6 * (1.0 - fck / 250.0)
    sigma_rd_max = nu * fcd

    return {
        "theta_deg": math.degrees(theta),
        "strut_force": strut_force,
        "tie_x": tie_x,
        "tie_y": tie_y,
        "As_req_x": as_req_x,
        "As_req_y": as_req_y,
        "sigma_rd_max": sigma_rd_max,
    }


st.title("Pile Cap Design App (Eurocode 2)")
st.caption("Includes Beam Theory and Strut-and-Tie Method (STM) checks for preliminary design.")

with st.sidebar:
    st.header("Material")
    fck = st.selectbox("Concrete strength fck (MPa)", [20, 25, 30, 35, 40, 45, 50], index=2)
    fyk = st.selectbox("Steel fyk (MPa)", [400, 500], index=1)

    st.header("Geometry (mm)")
    nx = st.number_input("Number of piles in X", min_value=1, max_value=6, value=2, step=1)
    ny = st.number_input("Number of piles in Y", min_value=1, max_value=6, value=2, step=1)
    sx = st.number_input("Pile spacing sx", min_value=300.0, value=1800.0, step=50.0)
    sy = st.number_input("Pile spacing sy", min_value=300.0, value=1800.0, step=50.0)
    h = st.number_input("Cap thickness h", min_value=400.0, value=900.0, step=25.0)
    cover = st.number_input("Bottom cover", min_value=40.0, value=75.0, step=5.0)
    bar_dia = st.selectbox("Main bar diameter", [12, 16, 20, 25, 32], index=2)

    st.header("Column & actions")
    cx = st.number_input("Column size cx", min_value=200.0, value=500.0, step=50.0)
    cy = st.number_input("Column size cy", min_value=200.0, value=500.0, step=50.0)
    n_ed = st.number_input("Design axial load NEd (kN)", min_value=100.0, value=6000.0, step=100.0)

    st.header("Provided reinforcement")
    n_bars_x = st.number_input("Bars in X direction", min_value=2, value=10, step=1)
    n_bars_y = st.number_input("Bars in Y direction", min_value=2, value=10, step=1)


d = h - cover - bar_dia / 2.0
coords = pile_coordinates(int(nx), int(ny), sx, sy)
beam = beam_theory_design(coords, n_ed, cx, cy, d, fck, fyk)
stm = strut_tie_design(int(nx), int(ny), sx, sy, d, n_ed, fck, fyk)

as_prov_x = n_bars_x * steel_area(bar_dia)
as_prov_y = n_bars_y * steel_area(bar_dia)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Beam theory method")
    st.metric("Pile reaction per pile", f"{beam['n_pile']:.1f} kN")
    st.metric("MEd,x", f"{beam['MEd_x']:.1f} kNm")
    st.metric("MEd,y", f"{beam['MEd_y']:.1f} kNm")
    st.metric("As,req,x", f"{beam['As_req_x']:.0f} mm²")
    st.metric("As,req,y", f"{beam['As_req_y']:.0f} mm²")
    st.metric("VRdc,x / VEd,x", f"{beam['VRdc_x']:.0f} / {beam['VEd_x']:.0f} kN")
    st.metric("VRdc,y / VEd,y", f"{beam['VRdc_y']:.0f} / {beam['VEd_y']:.0f} kN")

with c2:
    st.subheader("Strut-and-tie method (STM)")
    st.metric("Strut angle θ", f"{stm['theta_deg']:.1f}°")
    st.metric("Strut compression force", f"{stm['strut_force']:.1f} kN")
    st.metric("Tie force T_x", f"{stm['tie_x']:.1f} kN")
    st.metric("Tie force T_y", f"{stm['tie_y']:.1f} kN")
    st.metric("As,req,x", f"{stm['As_req_x']:.0f} mm²")
    st.metric("As,req,y", f"{stm['As_req_y']:.0f} mm²")
    st.metric("Max node stress σRd,max", f"{stm['sigma_rd_max']:.2f} MPa")

st.markdown("---")
st.subheader("Reinforcement check")
req_x = max(beam["As_req_x"], stm["As_req_x"])
req_y = max(beam["As_req_y"], stm["As_req_y"])

ok_x = as_prov_x >= req_x
ok_y = as_prov_y >= req_y

summary = pd.DataFrame(
    [
        ["X", req_x, as_prov_x, "OK" if ok_x else "Increase bars"],
        ["Y", req_y, as_prov_y, "OK" if ok_y else "Increase bars"],
    ],
    columns=["Direction", "As required governing (mm²)", "As provided (mm²)", "Status"],
)
st.dataframe(summary, width="stretch")

st.info(
    "This app is intended for preliminary sizing to EN 1992-1-1 principles. "
    "For final design, include full detailing checks, punching checks, anchorage, and geotechnical verification."
)

st.subheader("Pile layout (model)")
coords_df = pd.DataFrame(coords, columns=["x (mm)", "y (mm)"])
coords_df.index = [f"Pile {i+1}" for i in range(len(coords_df))]
st.dataframe(coords_df, width="stretch")
