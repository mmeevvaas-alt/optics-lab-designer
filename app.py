
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# הגדרות כלליות
# -----------------------------
st.set_page_config(
    page_title="Optics Lab Designer",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] { direction: rtl; }
    .block-container { padding-top: 1.4rem; padding-bottom: 3rem; }
    .metric-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 16px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .small-note {
        font-size: 0.90rem;
        opacity: 0.85;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# בחירת ניסוי
# -----------------------------
EXPERIMENTS = ["דמות בעדשות", "התאבכות ועקיפה"]
experiment = st.radio("בחרי ניסוי", EXPERIMENTS, horizontal=True, key="experiment")

if experiment == "התאבכות ועקיפה":
    from interference import render_interference
    render_interference()
    st.stop()

LENSES = pd.DataFrame(
    [
        [30, 22.2, 11.0, 21.0, 25.4],
        [35, 27.8, 9.5, 21.0, 25.4],
        [40, 33.3, 8.5, 21.0, 25.4],
        [45, 38.8, 8.0, 21.0, 25.4],
        [50, 44.1, 7.5, 21.0, 25.4],
        [60, 54.7, 6.8, 21.0, 25.4],
        [75, 70.0, 6.3, 21.0, 25.4],
        [100, 95.7, 5.5, 21.0, 25.4],
        [150, 146.1, 5.0, 21.0, 25.4],
        [200, 196.3, 4.8, 21.0, 25.4],
        [250, 246.5, 4.6, 21.0, 25.4],
        [300, 296.6, 4.5, 21.0, 25.4],
        [400, 396.8, 4.3, 21.0, 25.4],
        [500, 496.9, 4.2, 21.0, 25.4],
    ],
    columns=["EFL_mm", "BFL_mm", "Tc_mm", "CA_mm", "Diameter_mm"],
)
LENSES["Lens"] = LENSES["EFL_mm"].map(lambda x: f"EFL {x:g} mm")


def safe_image_distance(f, u):
    """Signed thin-lens image distance. u>0 for a real object to the left."""
    den = (1.0 / f) - (1.0 / u)
    if abs(den) < 1e-12:
        return np.inf
    return 1.0 / den


def signed_magnification(u, v):
    if not np.isfinite(v) or abs(u) < 1e-12:
        return np.nan
    return -v / u


def lens_info_box(row):
    st.markdown(
        f"""
        <div class="metric-card">
        <b>עדשה נבחרת</b><br>
        EFL = {row.EFL_mm:g} mm &nbsp; | &nbsp;
        BFL = {row.BFL_mm:g} mm &nbsp; | &nbsp;
        Tc = {row.Tc_mm:g} mm<br>
        Clear aperture = {row.CA_mm:g} mm &nbsp; | &nbsp;
        קוטר מכני = {row.Diameter_mm:g} mm
        </div>
        """,
        unsafe_allow_html=True,
    )


def single_lens_figure(f, u, ho, ca, screen_h):
    v = safe_image_distance(f, u)
    m = signed_magnification(u, v)
    hi = m * ho if np.isfinite(m) else np.nan

    x_obj = -u
    x_lens = 0.0

    if np.isfinite(v):
        # Set plotting range sensibly for real or virtual image.
        if v > 0:
            x_right = max(v * 1.15, f * 2.2, 80)
            x_left = min(x_obj * 1.15, -f * 2.2)
        else:
            x_right = max(f * 2.5, 100)
            x_left = min(x_obj * 1.15, v * 1.15, -f * 2.2)
    else:
        x_right = max(f * 4, 200)
        x_left = min(x_obj * 1.15, -f * 2.2)

    fig = go.Figure()

    # Optical axis
    fig.add_shape(type="line", x0=x_left, x1=x_right, y0=0, y1=0,
                  line=dict(color="gray", width=1))

    # Lens
    half_ca = ca / 2
    fig.add_shape(type="line", x0=0, x1=0, y0=-half_ca, y1=half_ca,
                  line=dict(color="royalblue", width=5))
    fig.add_annotation(x=0, y=half_ca * 1.12, text=f"Lens<br>f={f:g} mm",
                       showarrow=False)

    # Focal points
    fig.add_trace(go.Scatter(
        x=[-f, f], y=[0, 0],
        mode="markers+text",
        marker=dict(size=7, color="royalblue"),
        text=["F", "F′"],
        textposition=["bottom center", "bottom center"],
        name="מוקדים"
    ))

    # Object arrow
    fig.add_annotation(
        x=x_obj, y=ho/2, ax=x_obj, ay=0,
        xref="x", yref="y", axref="x", ayref="y",
        text="עצם", showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=3,
    )

    # Image arrow / screen
    if np.isfinite(v):
        if v > 0:
            fig.add_shape(
                type="rect",
                x0=v-1.5, x1=v+1.5,
                y0=-screen_h/2, y1=screen_h/2,
                line=dict(color="black", width=1),
                fillcolor="rgba(0,0,0,0.04)"
            )
            fig.add_annotation(x=v, y=screen_h/2*1.10, text="מסך", showarrow=False)
        fig.add_annotation(
            x=v, y=hi/2, ax=v, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            text="דמות" + (" ממשית" if v > 0 else " מדומה"),
            showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=3,
        )

    # Rays from top of object: center ray and a parallel ray.
    y0 = ho / 2

    # Ray through center of lens.
    theta_center = (0 - y0) / u
    x_seg = np.array([x_obj, 0.0, x_right])
    y_seg = np.array([y0, 0.0, theta_center * x_right])
    fig.add_trace(go.Scatter(
        x=x_seg, y=y_seg, mode="lines",
        line=dict(width=2), name="קרן דרך מרכז העדשה"
    ))

    # Parallel ray to the lens at y=y0, then refracted.
    if abs(y0) <= half_ca:
        theta_after = -y0 / f
        x2 = np.array([x_obj, 0.0, x_right])
        y2 = np.array([y0, y0, y0 + theta_after*x_right])
        fig.add_trace(go.Scatter(
            x=x2, y=y2, mode="lines",
            line=dict(width=2), name="קרן מקבילה לציר"
        ))

        # Backward extensions for virtual image
        if np.isfinite(v) and v < 0:
            x_back = np.array([v, 0.0])
            y_back = np.array([hi/2, y0])
            fig.add_trace(go.Scatter(
                x=x_back, y=y_back, mode="lines",
                line=dict(width=1.5, dash="dash"),
                name="המשך לאחור"
            ))

    # Dynamic y range
    ys = [abs(ho/2), abs(half_ca), abs(screen_h/2)]
    if np.isfinite(hi):
        ys.append(abs(hi/2))
    y_lim = max(max(ys) * 1.35, 12)

    fig.update_layout(
        height=480,
        margin=dict(l=25, r=25, t=35, b=25),
        xaxis_title="מיקום לאורך הציר האופטי [mm]",
        yaxis_title="גובה [mm]",
        yaxis=dict(range=[-y_lim, y_lim], zeroline=False),
        xaxis=dict(range=[x_left, x_right], zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title="שרטוט קרניים — מודל עדשה דקה",
    )
    return fig


def reciprocal_graph(f, axis_mode="1/v מול 1/u", umax_factor=8):
    u_min = max(f * 1.05, f + 1e-6)
    u_max = max(f * umax_factor, u_min + 50)
    u = np.linspace(u_min, u_max, 250)
    v = np.array([safe_image_distance(f, uu) for uu in u])
    invu = 1/u
    invv = 1/v

    if axis_mode == "1/v מול 1/u":
        x, y = invu, invv
        xlab, ylab = "1/u [1/mm]", "1/v [1/mm]"
    else:
        x, y = invv, invu
        xlab, ylab = "1/v [1/mm]", "1/u [1/mm]"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="מודל תאורטי"))
    fig.update_layout(
        height=410,
        xaxis_title=xlab,
        yaxis_title=ylab,
        title="לינאריזציה של משוואת העדשה",
        margin=dict(l=25, r=25, t=50, b=25),
    )
    return fig


def simulated_measurements(f, n, sigma_u, sigma_v, seed):
    rng = np.random.default_rng(seed)
    u_true = np.linspace(1.25*f, 5.0*f, n)
    v_true = np.array([safe_image_distance(f, uu) for uu in u_true])

    u_meas = u_true + rng.normal(0, sigma_u, size=n)
    v_meas = v_true + rng.normal(0, sigma_v, size=n)

    x = 1/u_meas
    y = 1/v_meas
    slope, intercept = np.polyfit(x, y, 1)
    f_fit = 1/intercept if abs(intercept) > 1e-12 else np.nan

    y_fit = slope*x + intercept
    residuals = y-y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-y.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

    df = pd.DataFrame({
        "u_true_mm": u_true,
        "v_true_mm": v_true,
        "u_measured_mm": u_meas,
        "v_measured_mm": v_meas,
        "1/u [1/mm]": x,
        "1/v [1/mm]": y,
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="מדידות מסומלצות"))
    xs = np.linspace(x.min()*0.95, x.max()*1.05, 200)
    fig.add_trace(go.Scatter(
        x=xs, y=slope*xs+intercept, mode="lines",
        name=f"Fit: y={slope:.3f}x+{intercept:.5f}"
    ))
    fig.update_layout(
        height=400,
        xaxis_title="1/u [1/mm]",
        yaxis_title="1/v [1/mm]",
        title="מדידות מסומלצות + fit ליניארי",
        margin=dict(l=25, r=25, t=50, b=25),
    )

    return df, fig, slope, intercept, f_fit, r2


def two_lens_actual(f1, f2, u1, d):
    """Return paraxial thin-lens result for a chosen u1 and lens separation d."""
    v1 = safe_image_distance(f1, u1)
    if not np.isfinite(v1):
        return None
    m1 = -v1/u1
    u2 = d - v1  # signed object distance for lens 2
    if abs(u2) < 1e-10:
        return None
    v2 = safe_image_distance(f2, u2)
    if not np.isfinite(v2):
        return None
    m2 = -v2/u2
    mt = m1*m2
    return dict(v1=v1, m1=m1, u2=u2, v2=v2, m2=m2, M=mt)


def full_field_height_two_lens(u1, d, f1, ca1, ca2, M, screen_height):
    """
    Maximum centered object height for which the FULL bundle accepted by lens 1
    also clears lens 2 (100% unvignetted relative to lens 1), plus screen limit.
    Paraxial thin-lens model.
    """
    a1 = ca1/2
    a2 = ca2/2

    if d <= 0 or u1 <= 0:
        h_vig = 0.0
    else:
        coeff_y1 = 1 + d/u1 - d/f1
        margin = a2 - abs(coeff_y1)*a1
        if margin <= 0:
            h_vig = 0.0
        else:
            y0_max = margin * u1 / d
            h_vig = 2*y0_max

    h_screen = np.inf
    if abs(M) > 1e-12 and screen_height > 0:
        h_screen = screen_height/abs(M)

    h_full = min(h_vig, h_screen) if np.isfinite(h_screen) else h_vig
    return max(h_full, 0.0), max(h_vig, 0.0), h_screen


@st.cache_data(show_spinner=False)
def find_two_lens_solutions(
    target_M,
    tolerance_pct,
    bench_length,
    min_sep,
    min_screen,
    min_u1,
    position_step,
    require_real_intermediate,
    screen_height,
    lens1_ids,
    lens2_ids,
):
    rows = []

    # Search object distances on the actual lab positioning grid.
    max_u = max(min_u1 + position_step, bench_length - min_sep - min_screen)
    u_grid = np.arange(min_u1, max_u + 0.1*position_step, position_step)

    for i in lens1_ids:
        L1 = LENSES.loc[i]
        f1 = float(L1.EFL_mm)
        for j in lens2_ids:
            L2 = LENSES.loc[j]
            f2 = float(L2.EFL_mm)

            for u1 in u_grid:
                if abs(u1-f1) < 0.25*position_step:
                    continue

                v1 = safe_image_distance(f1, u1)
                if not np.isfinite(v1):
                    continue
                m1 = -v1/u1
                if abs(m1) < 1e-10:
                    continue

                desired_m2 = target_M/m1
                if abs(desired_m2) < 1e-10:
                    continue

                # For a thin lens, u = f(1 - 1/m)
                u2_desired = f2 * (1 - 1/desired_m2)
                d_ideal = v1 + u2_desired

                if not np.isfinite(d_ideal):
                    continue

                # Respect finite positioning resolution.
                d = round(d_ideal/position_step) * position_step
                if d < min_sep:
                    continue

                res = two_lens_actual(f1, f2, u1, d)
                if res is None:
                    continue

                v2 = res["v2"]
                if v2 < min_screen:
                    continue

                if require_real_intermediate:
                    if not (res["v1"] > 0 and res["u2"] > 0):
                        continue

                total_length = u1 + d + v2
                if total_length > bench_length:
                    continue

                err_pct = abs(res["M"]-target_M)/abs(target_M)*100
                if err_pct > tolerance_pct:
                    continue

                h_full, h_vig, h_screen = full_field_height_two_lens(
                    u1, d, f1, float(L1.CA_mm), float(L2.CA_mm),
                    res["M"], screen_height
                )

                rows.append({
                    "L1_EFL_mm": f1,
                    "L2_EFL_mm": f2,
                    "u1_mm": u1,
                    "d12_mm": d,
                    "v1_mm": res["v1"],
                    "u2_mm": res["u2"],
                    "screen_from_L2_mm": v2,
                    "M1": res["m1"],
                    "M2": res["m2"],
                    "M_total": res["M"],
                    "error_pct": err_pct,
                    "total_length_mm": total_length,
                    "full_object_height_mm": h_full,
                    "unvignetted_by_L2_height_mm": h_vig,
                    "screen_limited_object_height_mm": h_screen,
                    "CA1_mm": float(L1.CA_mm),
                    "CA2_mm": float(L2.CA_mm),
                    "Tc1_mm": float(L1.Tc_mm),
                    "Tc2_mm": float(L2.Tc_mm),
                })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["error_pct", "total_length_mm", "full_object_height_mm"],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    # Keep a manageable number of best candidates while preserving variety.
    return out.head(500)


def trace_ray_through_two_lenses(y_obj, y_at_l1, u1, d, f1, f2, v2, ca1, ca2):
    """Trace one paraxial ray; stop at lens 2 if clipped."""
    x_obj = 0
    x_l1 = u1
    x_l2 = u1 + d
    x_scr = x_l2 + v2

    theta0 = (y_at_l1-y_obj)/u1
    theta1 = theta0 - y_at_l1/f1
    y2 = y_at_l1 + d*theta1

    x = [x_obj, x_l1, x_l2]
    y = [y_obj, y_at_l1, y2]

    if abs(y_at_l1) > ca1/2:
        return x[:2], y[:2], True, "L1"

    if abs(y2) > ca2/2:
        return x, y, True, "L2"

    theta2 = theta1 - y2/f2
    yscr = y2 + v2*theta2
    x.append(x_scr)
    y.append(yscr)
    return x, y, False, None


def two_lens_figure(sol, object_height, screen_height):
    f1 = sol["L1_EFL_mm"]
    f2 = sol["L2_EFL_mm"]
    u1 = sol["u1_mm"]
    d = sol["d12_mm"]
    v1 = sol["v1_mm"]
    v2 = sol["screen_from_L2_mm"]
    M = sol["M_total"]
    ca1 = sol["CA1_mm"]
    ca2 = sol["CA2_mm"]

    x_obj = 0
    x_l1 = u1
    x_l2 = u1 + d
    x_scr = x_l2 + v2
    x_inter = x_l1 + v1

    fig = go.Figure()
    fig.add_shape(type="line", x0=-10, x1=x_scr*1.05, y0=0, y1=0,
                  line=dict(color="gray", width=1))

    # Lenses and screen
    for x, ca, label in [(x_l1, ca1, f"L1<br>f={f1:g}"), (x_l2, ca2, f"L2<br>f={f2:g}")]:
        fig.add_shape(type="line", x0=x, x1=x, y0=-ca/2, y1=ca/2,
                      line=dict(color="royalblue", width=5))
        fig.add_annotation(x=x, y=ca/2*1.22, text=label, showarrow=False)

    fig.add_shape(type="rect", x0=x_scr-1.5, x1=x_scr+1.5,
                  y0=-screen_height/2, y1=screen_height/2,
                  line=dict(color="black", width=1),
                  fillcolor="rgba(0,0,0,0.04)")
    fig.add_annotation(x=x_scr, y=screen_height/2*1.10, text="מסך", showarrow=False)

    # Object and final image
    y_obj = object_height/2
    y_img = M*y_obj
    fig.add_annotation(
        x=x_obj, y=y_obj, ax=x_obj, ay=0,
        xref="x", yref="y", axref="x", ayref="y",
        text="עצם", showarrow=True, arrowhead=3, arrowwidth=3
    )
    fig.add_annotation(
        x=x_scr, y=y_img, ax=x_scr, ay=0,
        xref="x", yref="y", axref="x", ayref="y",
        text="דמות סופית", showarrow=True, arrowhead=3, arrowwidth=3
    )

    # Intermediate image (real or virtual)
    m1 = sol["M1"]
    y_inter = m1*y_obj
    if np.isfinite(x_inter):
        fig.add_annotation(
            x=x_inter, y=y_inter,
            text="דמות ביניים",
            showarrow=True,
            arrowhead=2,
            ax=30, ay=-30
        )

    # Rays from top point of object, aimed at center and edges of L1
    for y1 in [-ca1/2, 0, ca1/2]:
        xs, ys, clipped, where = trace_ray_through_two_lenses(
            y_obj, y1, u1, d, f1, f2, v2, ca1, ca2
        )
        name = "קרן" + (f" — נחסמה ב-{where}" if clipped else "")
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(width=1.8, dash="dash" if clipped else "solid"),
            name=name,
            showlegend=True
        ))

    y_lim = max(ca1/2, ca2/2, abs(y_obj), abs(y_img), screen_height/2) * 1.45
    y_lim = max(y_lim, 12)

    fig.update_layout(
        height=520,
        xaxis_title="מיקום לאורך המסילה [mm]",
        yaxis_title="גובה [mm]",
        xaxis=dict(range=[-15, x_scr*1.05]),
        yaxis=dict(range=[-y_lim, y_lim]),
        title="מערכת שתי עדשות — מיקום רכיבים ושרטוט קרניים",
        margin=dict(l=25, r=25, t=55, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def recompute_measured_system(f1, f2, u1, d):
    res = two_lens_actual(f1, f2, u1, d)
    if res is None:
        return None
    return res


# -----------------------------
# כותרת
# -----------------------------
st.title("🔬 Optics Lab Designer")
st.caption("סימולטור לתכנון וניתוח ניסויי עדשות במעבדת אופטיקה")

with st.expander("הנחות המודל — חשוב לקרוא"):
    st.markdown(
        """
        הסימולציה משתמשת ב**אופטיקה פראקסיאלית ובמודל עדשה דקה**, כאשר אורך המוקד
        הוא ה־EFL של העדשה. נתוני BFL ועובי העדשה מוצגים למשתמש, אך אינם מספיקים
        לבדם כדי לשחזר במדויק את מיקום שני המישורים הראשיים של עדשה עבה.

        לכן הסימולציה מתאימה מאוד ל**תכנון מוקדם, חיזוי, sensitivity analysis והשוואה לניסוי**,
        אך היא אינה מחליפה מדידה של המערכת האמיתית, שבה קיימים עובי עדשה, סטיות,
        אי־יישור, אברציות ואי־ודאות.

        ה־Clear Aperture משמש לבדיקת vignetting במערכת של שתי עדשות.
        בעדשה יחידה אידאלית וממורכזת אין "גובה עצם מקסימלי" חד שנובע מאותה עדשה בלבד;
        ה־aperture משפיע בעיקר על שטף האור וה־NA. לכן בחלק 1 מוצג גם גבול המסך,
        ובחלק 2 מחושב גבול שדה שבו כל האלומה שהתקבלה ע"י L1 עוברת גם דרך L2.
        """
    )

tab1, tab2, tab3 = st.tabs([
    "חלק 1 — עדשה יחידה",
    "חלק 2 — תכנון שתי עדשות",
    "השוואה לניסוי"
])

# =========================================================
# חלק 1
# =========================================================
with tab1:
    st.subheader("עדשה יחידה: חיזוי, דמות, הגדלה ולינאריזציה")

    c1, c2 = st.columns([1, 2])

    with c1:
        lens_label = st.selectbox("בחרי עדשה", LENSES["Lens"].tolist(), index=4)
        row = LENSES[LENSES["Lens"] == lens_label].iloc[0]
        f = float(row.EFL_mm)
        lens_info_box(row)

        u_min = max(10.0, 0.35*f)
        u_max = min(1500.0, max(8*f, 250.0))
        default_u = min(max(2.5*f, u_min), u_max)

        u = st.slider(
            "מרחק עצם מן העדשה u [mm]",
            min_value=float(u_min),
            max_value=float(u_max),
            value=float(default_u),
            step=1.0,
        )

        ho = st.slider(
            "גובה העצם [mm]",
            min_value=2.0,
            max_value=80.0,
            value=20.0,
            step=1.0,
        )
        screen_h = st.slider(
            "גובה המסך הפעיל [mm]",
            min_value=10.0,
            max_value=150.0,
            value=60.0,
            step=1.0,
        )

        v = safe_image_distance(f, u)
        m = signed_magnification(u, v)
        hi = m*ho if np.isfinite(m) else np.nan

        if np.isfinite(v):
            image_type = "ממשית" if v > 0 else "מדומה"
            orientation = "הפוכה" if m < 0 else "ישרה"
            st.metric("מרחק הדמות v", f"{v:.2f} mm")
            st.metric("הגדלה M", f"{m:.3f}")
            st.metric("גובה הדמות", f"{hi:.2f} mm")
            st.info(f"הדמות {image_type} ו{orientation}.")
        else:
            st.warning("העצם נמצא למעשה במוקד: במודל האידאלי הדמות מתקבלת באינסוף.")

        if np.isfinite(m) and abs(m) > 1e-12:
            max_object_screen = screen_h / abs(m)
            st.metric("גובה עצם מרבי לפי גודל המסך", f"{max_object_screen:.1f} mm")

        fnum = f / float(row.CA_mm)
        theta = math.atan((float(row.CA_mm)/2)/u)
        na_obj = math.sin(theta)
        st.caption(f"f/# בקירוב: {fnum:.2f} | NA בצד העצם בקירוב: {na_obj:.3f}")

    with c2:
        st.plotly_chart(
            single_lens_figure(f, u, ho, float(row.CA_mm), screen_h),
            use_container_width=True
        )

    st.divider()

    g1, g2 = st.columns([1, 1])
    with g1:
        axis_mode = st.radio(
            "בחירת צירים",
            ["1/v מול 1/u", "1/u מול 1/v"],
            horizontal=True
        )
        st.plotly_chart(reciprocal_graph(f, axis_mode), use_container_width=True)

    with g2:
        st.markdown("#### סימולציית מדידות")
        enable_noise = st.checkbox("צור סדרת מדידות מסומלצות עם רעש", value=True)
        if enable_noise:
            npts = st.slider("מספר נקודות", 4, 15, 7)
            sigma_u = st.number_input("σ(u) [mm]", min_value=0.0, value=1.0, step=0.1)
            sigma_v = st.number_input("σ(v) [mm]", min_value=0.0, value=1.0, step=0.1)
            seed = st.number_input("Seed", min_value=0, value=42, step=1)

            df_sim, fig_sim, slope, intercept, f_fit, r2 = simulated_measurements(
                f, npts, sigma_u, sigma_v, int(seed)
            )
            st.plotly_chart(fig_sim, use_container_width=True)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("שיפוע", f"{slope:.4f}", help="תאורטית: -1")
            mc2.metric("f מתוך החיתוך", f"{f_fit:.2f} mm")
            mc3.metric("R²", f"{r2:.5f}")
            st.dataframe(df_sim.round(4), use_container_width=True)

            st.caption(
                "ה-fit כאן הוא OLS פשוט לצורכי הדגמה. בניסוי אמיתי, כאשר יש אי־ודאות "
                "משמעותית גם ב-u וגם ב-v, כדאי לדון בשיטת fit מתאימה יותר."
            )

# =========================================================
# חלק 2
# =========================================================
with tab2:
    st.subheader("תכנון מערכת שתי עדשות לפי הגדלה רצויה")

    left, right = st.columns([1, 2])

    with left:
        mag_abs = st.slider("ערך מוחלט של ההגדלה הרצויה |M|", 0.25, 8.0, 2.0, 0.05)
        orientation = st.radio(
            "כיוון הדמות הסופית",
            ["ישרה (M>0)", "הפוכה (M<0)"],
            horizontal=False
        )
        target_M = mag_abs if "M>0" in orientation else -mag_abs

        tolerance_pct = st.slider("סטייה מותרת בהגדלה [%]", 0.1, 15.0, 3.0, 0.1)
        bench_length = st.slider("אורך מסילה מרבי [mm]", 300, 2500, 1200, 10)
        min_sep = st.slider("מרחק מינימלי בין העדשות [mm]", 10, 150, 30, 5)
        min_screen = st.slider("מרחק מינימלי L2→מסך [mm]", 10, 300, 30, 5)
        min_u1 = st.slider("מרחק מינימלי עצם→L1 [mm]", 10, 300, 30, 5)
        position_step = st.select_slider(
            "רזולוציית מיקום מעשית [mm]",
            options=[0.5, 1.0, 2.0, 5.0, 10.0],
            value=1.0
        )
        screen_height_2 = st.slider("גובה מסך [mm]", 20, 200, 80, 5)
        require_real_intermediate = st.checkbox("דרוש שדמות הביניים תהיה ממשית", value=False)

        st.markdown("##### עדשות מותרות")
        all_efls = LENSES["EFL_mm"].tolist()
        chosen_l1 = st.multiselect("אפשרויות L1", all_efls, default=all_efls)
        chosen_l2 = st.multiselect("אפשרויות L2", all_efls, default=all_efls)

        l1_ids = LENSES[LENSES["EFL_mm"].isin(chosen_l1)].index.tolist()
        l2_ids = LENSES[LENSES["EFL_mm"].isin(chosen_l2)].index.tolist()

    with right:
        if not l1_ids or not l2_ids:
            st.warning("יש לבחור לפחות עדשה אחת עבור L1 ועדשה אחת עבור L2.")
            solutions = pd.DataFrame()
        elif abs(target_M) < 1e-9:
            st.warning("הגדלה אפס אינה מוגדרת כיעד אופטי.")
            solutions = pd.DataFrame()
        else:
            solutions = find_two_lens_solutions(
                target_M=target_M,
                tolerance_pct=tolerance_pct,
                bench_length=float(bench_length),
                min_sep=float(min_sep),
                min_screen=float(min_screen),
                min_u1=float(min_u1),
                position_step=float(position_step),
                require_real_intermediate=require_real_intermediate,
                screen_height=float(screen_height_2),
                lens1_ids=tuple(l1_ids),
                lens2_ids=tuple(l2_ids),
            )

        if solutions.empty:
            st.error(
                "לא נמצאו פתרונות עם האילוצים הנוכחיים. נסי להגדיל את אורך המסילה, "
                "להגדיל tolerance, להקטין מרחקי מינימום או לאפשר דמות ביניים וירטואלית."
            )
        else:
            st.success(f"נמצאו {len(solutions)} פתרונות מתאימים (מוצגים עד 500 הטובים ביותר).")

            display_cols = [
                "L1_EFL_mm", "L2_EFL_mm", "u1_mm", "d12_mm",
                "screen_from_L2_mm", "M_total", "error_pct",
                "total_length_mm", "full_object_height_mm"
            ]
            pretty = solutions[display_cols].copy()
            pretty.columns = [
                "L1 EFL [mm]", "L2 EFL [mm]", "עצם→L1 [mm]",
                "L1→L2 [mm]", "L2→מסך [mm]", "M בפועל",
                "שגיאת M [%]", "אורך כולל [mm]",
                "גובה עצם מלא ללא vignetting [mm]"
            ]
            st.dataframe(pretty.round(3), use_container_width=True, height=300)

            best_n = min(25, len(solutions))
            idx = st.selectbox(
                "בחרי פתרון להצגה",
                options=list(range(best_n)),
                format_func=lambda k: (
                    f"#{k+1}: f1={solutions.iloc[k]['L1_EFL_mm']:.0f}, "
                    f"f2={solutions.iloc[k]['L2_EFL_mm']:.0f} mm | "
                    f"M={solutions.iloc[k]['M_total']:.3f} | "
                    f"L={solutions.iloc[k]['total_length_mm']:.0f} mm"
                )
            )
            sol = solutions.iloc[idx].to_dict()

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("M כולל", f"{sol['M_total']:.4f}")
            s2.metric("שגיאת הגדלה", f"{sol['error_pct']:.2f}%")
            s3.metric("אורך מערכת", f"{sol['total_length_mm']:.1f} mm")
            s4.metric("גובה עצם מלא", f"{sol['full_object_height_mm']:.1f} mm")

            st.markdown(
                f"""
                **מיקומי מערכת נבחרת**
                - עצם → L1: **{sol['u1_mm']:.1f} mm**
                - L1 → L2: **{sol['d12_mm']:.1f} mm**
                - L2 → מסך: **{sol['screen_from_L2_mm']:.1f} mm**
                - דמות ביניים ביחס ל-L1: **v₁ = {sol['v1_mm']:.1f} mm**
                - מרחק העצם של L2 (חתום): **u₂ = {sol['u2_mm']:.1f} mm**
                - \(M_1={sol['M1']:.3f}\), \(M_2={sol['M2']:.3f}\)
                """
            )

            default_obj_h = min(
                20.0,
                max(2.0, sol["full_object_height_mm"]*0.7)
                if sol["full_object_height_mm"] > 0 else 10.0
            )
            object_h2 = st.slider(
                "גובה העצם לשרטוט [mm]",
                min_value=2.0,
                max_value=80.0,
                value=float(round(default_obj_h, 1)),
                step=1.0,
                key="obj_h_two"
            )
            st.plotly_chart(
                two_lens_figure(sol, object_h2, screen_height_2),
                use_container_width=True
            )

            if sol["full_object_height_mm"] <= 0:
                st.warning(
                    "בגאומטריה זו L2 מצמצמת אפילו את האלומה המלאה של נקודה על הציר. "
                    "עדיין יכולה להיווצר דמות, אך היא כבר אינה 100% unvignetted ביחס למפתח L1."
                )
            else:
                st.info(
                    f"לפי מודל הקרניים, עצם ממורכז בגובה עד כ-{sol['full_object_height_mm']:.1f} mm "
                    "יודגם במלואו ללא vignetting של האלומה שהתקבלה ע״י L1, ובתנאי שהוא גם נכנס במסך."
                )

            # Sensitivity analysis
            st.markdown("#### רגישות להזזת L2")
            delta_l2 = st.slider("הזזת L2 [mm]", -10.0, 10.0, 1.0, 0.5)
            res_shift = two_lens_actual(
                sol["L1_EFL_mm"], sol["L2_EFL_mm"],
                sol["u1_mm"], sol["d12_mm"] + delta_l2
            )
            if res_shift and np.isfinite(res_shift["v2"]) and res_shift["v2"] > 0:
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("M לאחר הזזה", f"{res_shift['M']:.4f}",
                           delta=f"{res_shift['M']-sol['M_total']:+.4f}")
                cc2.metric("מיקום מסך חדש מ-L2", f"{res_shift['v2']:.2f} mm",
                           delta=f"{res_shift['v2']-sol['screen_from_L2_mm']:+.2f} mm")
                cc3.metric("שינוי דמות ביניים?", "לא", help="L1 והעצם לא זזו.")
            else:
                st.warning("לאחר ההזזה שנבחרה לא מתקבלת דמות סופית ממשית במיקום סופי.")

# =========================================================
# השוואה לניסוי
# =========================================================
with tab3:
    st.subheader("השוואת תכנון לסידור שבוצע במעבדה")
    st.write(
        "החלק הזה נועד להמחיש את הרעיון הפדגוגי: הסימולציה מייצרת prediction, "
        "והמעבדה בודקת מה באמת התקבל."
    )

    c1, c2 = st.columns(2)
    with c1:
        f1_meas = st.selectbox("EFL של L1 [mm]", LENSES["EFL_mm"].tolist(), index=4, key="cmp_f1")
        f2_meas = st.selectbox("EFL של L2 [mm]", LENSES["EFL_mm"].tolist(), index=7, key="cmp_f2")
        u1_meas = st.number_input("עצם→L1 שנמדד [mm]", min_value=1.0, value=150.0, step=1.0)
        d_meas = st.number_input("L1→L2 שנמדד [mm]", min_value=1.0, value=250.0, step=1.0)
        h_obj_meas = st.number_input("גובה עצם [mm]", min_value=0.1, value=20.0, step=0.5)
        h_img_meas = st.number_input("גובה דמות שנמדד [mm]", value=40.0, step=0.5)

    res = recompute_measured_system(float(f1_meas), float(f2_meas), float(u1_meas), float(d_meas))
    with c2:
        if res is None or not np.isfinite(res["v2"]):
            st.error("הגאומטריה שנבחרה נמצאת בנקודה סינגולרית במודל.")
        else:
            M_meas_geom = h_img_meas/h_obj_meas
            # Orientation cannot be inferred from positive heights, so compare magnitude.
            err_mag = abs(abs(M_meas_geom)-abs(res["M"])) / max(abs(res["M"]), 1e-12) * 100

            st.metric("מיקום דמות חזוי מ-L2", f"{res['v2']:.2f} mm")
            st.metric("הגדלה חזויה M", f"{res['M']:.4f}")
            st.metric("|M| מהגובה שנמדד", f"{abs(M_meas_geom):.4f}")
            st.metric("פער יחסי ב-|M|", f"{err_mag:.2f}%")

            st.markdown(
                """
                **שאלות שמומלץ לשאול בדוח**
                1. האם הפער יכול להיות מוסבר באי־ודאות מדידת המרחקים?
                2. איזה מרחק במערכת הכי רגיש לשגיאה של 1 mm?
                3. האם עובי העדשות והמישורים הראשיים יכולים להסביר חלק מהפער?
                4. האם נצפה vignetting, ואם כן — האם הוא תואם לתחזית?
                """
            )

st.divider()
st.caption(
    "Optics Lab Designer — כלי הוראה לתכנון, חיזוי והשוואה לניסוי. "
    "המודל הפראקסיאלי הוא נקודת פתיחה, לא תחליף למדידה."
)
