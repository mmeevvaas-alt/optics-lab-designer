"""
ניסוי התאבכות ועקיפה — סימולציה של תמונת המסך הצפויה ובדיקת תוצאות סטודנטים.

כל המרחקים בתוך הקוד הם במילימטרים. אורך הגל מוזן בננומטרים ומומר ל-mm.
המודל: עקיפת פראונהופר (שדה רחוק), עם חישוב זווית מדויק sinθ = x / sqrt(x² + L²).
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HENE_NM = 632.8
NM_TO_MM = 1e-6

SCREEN_COLORSCALE = [
    [0.0, "#000000"],
    [0.35, "#4a0000"],
    [0.7, "#e01010"],
    [0.9, "#ff5a3c"],
    [1.0, "#ffe2d6"],
]


# -----------------------------
# פיזיקה
# -----------------------------
def sin_theta(x, L):
    """sinθ מדויק עבור נקודה במרחק x מהמרכז על מסך במרחק L."""
    return x / np.sqrt(x ** 2 + L ** 2)


def x_from_sin(s, L):
    """מיקום על המסך עבור sinθ נתון (|s| < 1)."""
    s = np.asarray(s, dtype=float)
    return L * np.tan(np.arcsin(s))


def single_slit_intensity(x, a, lam, L):
    s = sin_theta(x, L)
    return np.sinc(a * s / lam) ** 2  # np.sinc(u) = sin(πu)/(πu)


def double_slit_intensity(x, a, d, lam, L):
    s = sin_theta(x, L)
    return np.cos(np.pi * d * s / lam) ** 2 * np.sinc(a * s / lam) ** 2


def grating_order_positions(lines_per_mm, lam, L, tilt_deg=0.0):
    """
    מחזיר טבלה של הסדרים שנראים. tilt = זווית פגיעה ביחס לניצב לשריג.
    המיקום על המסך נמדד ביחס לציר האלומה (המסך ניצב לאלומה).
    """
    d = 1.0 / lines_per_mm
    ti = math.radians(tilt_deg)
    rows = []
    n_lim = int(math.floor(2 * d / lam)) + 1
    for n in range(-n_lim, n_lim + 1):
        s = math.sin(ti) + n * lam / d
        if abs(s) >= 1:
            continue
        th = math.asin(s) - ti  # זווית ביחס לאלומה
        if abs(th) >= math.radians(89):
            continue
        rows.append({"n": n, "theta_deg": math.degrees(th), "X_mm": L * math.tan(th)})
    return pd.DataFrame(rows), d


# -----------------------------
# גרפים
# -----------------------------
def n_points(half_width, feature):
    """מספר דגימות שמבטיח לפחות ~12 נקודות לכל פס."""
    if feature <= 0 or not np.isfinite(feature):
        return 3000
    return int(np.clip(12 * 2 * half_width / feature, 1500, 8000))


def screen_figure(x, intensity, exposure, title):
    """תמונה מדומה של המסך, כפי שהעין או המצלמה רואה אותה."""
    disp = np.clip(intensity * exposure, 0, 1) ** 0.6
    yy = np.linspace(-1, 1, 41)
    beam = np.exp(-(yy / 0.42) ** 2)
    img = beam[:, None] * disp[None, :]
    fig = go.Figure(
        go.Heatmap(
            z=img, x=x, y=yy, colorscale=SCREEN_COLORSCALE,
            zmin=0, zmax=1, showscale=False, hoverinfo="skip",
        )
    )
    fig.update_yaxes(visible=False)
    fig.update_xaxes(title="מיקום על המסך [mm]", showgrid=False)
    fig.update_layout(
        title=title, height=190, margin=dict(l=10, r=10, t=40, b=40),
        plot_bgcolor="black",
    )
    return fig


def intensity_figure(x, intensity, envelope=None, marks=None, log_scale=False, title=""):
    fig = go.Figure()
    if envelope is not None:
        fig.add_trace(go.Scatter(
            x=x, y=envelope, mode="lines", name="מעטפת סדק יחיד",
            line=dict(color="gray", dash="dash", width=1),
        ))
    fig.add_trace(go.Scatter(
        x=x, y=intensity, mode="lines", name="I / I₀",
        line=dict(color="#d62728", width=1.6),
    ))
    if marks:
        for xm, label in marks[:14]:
            fig.add_vline(x=xm, line=dict(color="rgba(80,80,80,0.6)", dash="dot", width=1))
            if label:
                fig.add_annotation(x=xm, y=1.06, yref="paper", text=label,
                                   showarrow=False, font=dict(size=10))
    fig.update_layout(
        title=title, height=330, margin=dict(l=10, r=10, t=50, b=40),
        legend=dict(orientation="h", y=-0.25),
        xaxis_title="מיקום על המסך [mm]", yaxis_title="עוצמה יחסית",
    )
    if log_scale:
        fig.update_yaxes(type="log", range=[-4, 0.05])
    return fig


def fraunhofer_note(size_mm, lam, L):
    """מספר פרנל: אם הוא קרוב ל-1 ומעלה, קירוב השדה הרחוק פחות מדויק."""
    F = size_mm ** 2 / (lam * L)
    if F > 1:
        st.warning(
            f"מספר פרנל a²/(λL) ≈ {F:.2f}. המסך קרוב מדי לקירוב השדה הרחוק, "
            "והתמונה בפועל עלולה להיות שונה מהסימולציה."
        )
    else:
        st.caption(f"מספר פרנל ≈ {F:.3f} (קטן מ-1, קירוב השדה הרחוק תקף).")


def pct_dev(measured, ref):
    return (measured - ref) / ref * 100.0


# -----------------------------
# לשוניות
# -----------------------------
def tab_young(lam_nm, exposure):
    lam = lam_nm * NM_TO_MM
    c1, c2 = st.columns([1, 2])
    with c1:
        d = st.number_input("מרחק בין הסדקים d [mm]", min_value=0.005, value=0.250,
                            step=0.005, format="%.3f", key="y_d")
        a = st.number_input("רוחב כל סדק a [mm]", min_value=0.002, value=0.040,
                            step=0.005, format="%.3f", key="y_a")
        L = st.number_input("מרחק שקופית–מסך L [mm]", min_value=10.0, value=550.0,
                            step=5.0, key="y_L")
        half = st.slider("רוחב תצוגה ± [mm]", 2.0, 150.0, 25.0, 1.0, key="y_half")
        show_marks = st.checkbox("סמן את פסי האור", value=True, key="y_marks")
        log_scale = st.checkbox("ציר עוצמה לוגריתמי", value=False, key="y_log")

    if a >= d:
        st.error("רוחב הסדק a חייב להיות קטן מהמרחק בין הסדקים d.")
        return

    dx_small = lam * L / d
    dx_exact = float(x_from_sin(lam / d, L))
    env_zero = float(x_from_sin(lam / a, L)) if lam / a < 1 else np.inf
    ratio = d / a
    inside = 2 * int(math.floor(ratio - 1e-9)) + 1

    with c1:
        st.metric("מרווח בין פסים Δx", f"{dx_exact:.3f} mm",
                  help=f"בקירוב זווית קטנה: λL/d = {dx_small:.3f} mm")
        if np.isfinite(env_zero):
            st.metric("אפס ראשון של המעטפת", f"± {env_zero:.2f} mm")
        st.metric("פסי התאבכות בתוך הפס המרכזי של המעטפת", f"{inside}")
        if abs(ratio - round(ratio)) < 0.02 * ratio:
            st.info(f"d/a ≈ {round(ratio)}: הסדרים ±{round(ratio)}, ±{2*round(ratio)} … חסרים "
                    "(נופלים על אפסי המעטפת).")
        fraunhofer_note(d, lam, L)

    x = np.linspace(-half, half, n_points(half, dx_exact))
    I = double_slit_intensity(x, a, d, lam, L)
    env = single_slit_intensity(x, a, lam, L)

    marks = []
    if show_marks:
        n_max = int(math.floor(min(half / dx_exact, 7)))
        for n in range(-n_max, n_max + 1):
            s = n * lam / d
            if abs(s) < 1:
                marks.append((float(x_from_sin(s, L)), f"{n}"))

    with c2:
        st.plotly_chart(screen_figure(x, I, exposure, "תמונת המסך הצפויה"),
                        use_container_width=True)
        st.plotly_chart(intensity_figure(x, I, envelope=env, marks=marks, log_scale=log_scale,
                                         title="עוצמה לאורך המסך"),
                        use_container_width=True)

    st.markdown("#### ערכים צפויים")
    t1, t2 = st.columns([3, 2])
    with t1:
        rows = []
        for n in range(0, 7):
            s = n * lam / d
            if abs(s) >= 1:
                break
            x_ex = float(x_from_sin(s, L))
            rows.append({
                "סדר n": n,
                "θ [°]": math.degrees(math.asin(s)),
                "Xₙ מדויק [mm]": x_ex,
                "Xₙ = nλL/d [mm]": n * dx_small,
                "בתוך המעטפת המרכזית": "כן" if abs(x_ex) < env_zero else "לא",
            })
        st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True, hide_index=True)
    with t2:
        N = st.number_input("מספר פסי אור שנספרו N", min_value=2, value=8, step=1, key="y_N")
        st.metric(f"D צפוי על פני {N} פסים (N−1 מרווחים)", f"{(N - 1) * dx_exact:.2f} mm")
        st.caption(f"אם הסטודנטים ספרו {N} מרווחים: D ≈ {N * dx_exact:.2f} mm.")


def tab_single(lam_nm, exposure):
    lam = lam_nm * NM_TO_MM
    c1, c2 = st.columns([1, 2])
    with c1:
        a = st.number_input("רוחב הסדק a [mm]", min_value=0.002, value=0.080,
                            step=0.005, format="%.3f", key="s_a")
        L = st.number_input("מרחק שקופית–מסך L [mm]", min_value=10.0, value=550.0,
                            step=5.0, key="s_L")
        half = st.slider("רוחב תצוגה ± [mm]", 2.0, 150.0, 30.0, 1.0, key="s_half")
        log_scale = st.checkbox("ציר עוצמה לוגריתמי (לראות פסים חלשים)", value=False, key="s_log")

    if lam / a >= 1:
        st.error("הסדק צר מאורך הגל: אין פסי חושך, האור מתפזר לכל הכיוונים.")
        return

    y1 = float(x_from_sin(lam / a, L))
    with c1:
        st.metric("y₁: מהמרכז לפס החושך הראשון", f"{y1:.2f} mm")
        st.metric("D₁ = W₀: רוחב הפס המרכזי", f"{2 * y1:.2f} mm")
        st.caption("הסטודנטים מודדים D (בין שני פסי החושך) ומחלקים ב-2 כדי לקבל y.")
        fraunhofer_note(a, lam, L)

    x = np.linspace(-half, half, n_points(half, y1))
    I = single_slit_intensity(x, a, lam, L)
    marks = []
    for n in range(1, 6):
        s = n * lam / a
        if abs(s) < 1:
            yn = float(x_from_sin(s, L))
            if yn <= half:
                marks += [(yn, f"{n}"), (-yn, f"{n}")]

    with c2:
        st.plotly_chart(screen_figure(x, I, exposure, "תמונת המסך הצפויה"),
                        use_container_width=True)
        st.plotly_chart(intensity_figure(x, I, marks=marks, log_scale=log_scale,
                                         title="עוצמה לאורך המסך (קווים מנוקדים: פסי חושך)"),
                        use_container_width=True)

    st.markdown("#### ערכים צפויים")
    rows = []
    for n in range(1, 6):
        s = n * lam / a
        if abs(s) >= 1:
            break
        yn = float(x_from_sin(s, L))
        rows.append({
            "פס חושך n": n,
            "θ [°]": math.degrees(math.asin(s)),
            "yₙ מדויק [mm]": yn,
            "yₙ = nλL/a [mm]": n * lam * L / a,
            "Dₙ = 2yₙ (מה שנמדד) [mm]": 2 * yn,
        })
    st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True, hide_index=True)


def tab_grating(lam_nm, exposure):
    lam = lam_nm * NM_TO_MM
    c1, c2 = st.columns([1, 2])
    with c1:
        lpm = st.number_input("צפיפות קווים [קווים למ\"מ]", min_value=1.0, value=50.0,
                              step=10.0, key="g_lpm")
        L = st.number_input("מרחק שריג–מסך L [mm]", min_value=10.0, value=400.0,
                            step=5.0, key="g_L")
        tilt = st.slider("הטיית השריג (זווית פגיעה) θᵢ [°]", -40.0, 40.0, 0.0, 1.0, key="g_tilt",
                         help="0 = האלומה ניצבת לשריג. הטיה הופכת את התמונה ללא סימטרית.")
        half = st.slider("רוחב תצוגה ± [mm]", 10.0, 400.0, 150.0, 5.0, key="g_half")

    df, d = grating_order_positions(lpm, lam, L, tilt)
    n_max = int(math.floor(d / lam))
    with c1:
        st.metric("קבוע השריג d", f"{d * 1000:.2f} µm")
        st.metric("הסדר הגבוה ביותר האפשרי (ללא הטיה)", f"±{n_max}")

    x = np.linspace(-half, half, 6000)
    I = np.zeros_like(x)
    sigma = 0.6  # רוחב כתם הלייזר על המסך, mm
    for _, r in df.iterrows():
        I += np.exp(-((x - r.X_mm) / sigma) ** 2)
    I = np.clip(I, 0, 1)

    hidden = df[np.abs(df.X_mm) > half]
    with c2:
        st.plotly_chart(screen_figure(x, I, max(exposure, 1.0), "תמונת המסך הצפויה (מיקומי הסדרים)"),
                        use_container_width=True)
        st.caption("העוצמה של כל סדר תלויה ברוחב החריצים, שאינו ידוע, ולכן כל הכתמים מוצגים באותה עוצמה. "
                   "מה שחשוב הוא המיקום.")
        if len(hidden):
            st.info(f"{len(hidden)} סדרים נמצאים מחוץ לרוחב התצוגה. הגדילי את רוחב התצוגה כדי לראות אותם.")

    st.markdown("#### ערכים צפויים")
    visible = df[np.abs(df.X_mm) <= half]
    st.caption(f"הטבלה מציגה רק סדרים שנמצאים בטווח התצוגה (±{half:.0f} mm).")
    if tilt == 0:
        rows = []
        for _, r in visible[visible.n >= 0].iterrows():
            n = int(r.n)
            x_ex = r.X_mm
            x_sa = n * lam * L / d
            rows.append({
                "סדר n": n,
                "θₙ [°]": r.theta_deg,
                "Xₙ מדויק [mm]": x_ex,
                "Dₙ = 2Xₙ [mm]": 2 * x_ex,
                "Xₙ בקירוב זווית קטנה [mm]": x_sa,
                "שגיאת הקירוב [%]": pct_dev(x_sa, x_ex) if n else 0.0,
            })
        st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True, hide_index=True)
        st.caption("שגיאת הקירוב מראה כמה ישתבש d אם הסטודנטים משתמשים ב-sinθ ≈ X/L.")
    else:
        out = visible.copy()
        out.columns = ["סדר n", "זווית ביחס לאלומה [°]", "X על המסך [mm]"]
        st.dataframe(out.round(3), use_container_width=True, hide_index=True)
        st.caption("עם הטיה הסדרים +n ו-−n כבר לא במרחק שווה מהמרכז.")


def tab_hair(lam_nm, exposure):
    lam = lam_nm * NM_TO_MM
    c1, c2 = st.columns([1, 2])
    with c1:
        t_um = st.number_input("עובי השערה [µm]", min_value=5.0, value=60.0, step=5.0, key="h_t")
        L = st.number_input("מרחק שערה–מסך L [mm]", min_value=10.0, value=550.0,
                            step=5.0, key="h_L")
        half = st.slider("רוחב תצוגה ± [mm]", 2.0, 150.0, 40.0, 1.0, key="h_half")
    a = t_um / 1000.0
    y1 = float(x_from_sin(lam / a, L))
    with c1:
        st.metric("y₁", f"{y1:.2f} mm")
        st.metric("D₁ = 2y₁", f"{2 * y1:.2f} mm")
        st.caption("לפי עיקרון בבינה, התמונה זהה לסדק באותו רוחב, פרט לכתם המרכזי "
                   "שבו עובר רוב האור הישיר.")

    x = np.linspace(-half, half, n_points(half, y1))
    I = single_slit_intensity(x, a, lam, L)
    I_direct = np.exp(-(x / 0.6) ** 2)  # האלומה הישירה שלא נחסמה
    I_show = np.maximum(I * 0.25, I_direct)
    marks = []
    for n in range(1, 6):
        s = n * lam / a
        if s < 1:
            yn = float(x_from_sin(s, L))
            if yn <= half:
                marks += [(yn, f"{n}"), (-yn, f"{n}")]
    with c2:
        st.plotly_chart(screen_figure(x, I_show, exposure * 2, "תמונת המסך הצפויה"),
                        use_container_width=True)
        st.plotly_chart(intensity_figure(x, I, marks=marks, title="תבנית העקיפה (ללא האלומה הישירה)"),
                        use_container_width=True)

    rows = []
    for n in range(1, 6):
        s = n * lam / a
        if s >= 1:
            break
        yn = float(x_from_sin(s, L))
        rows.append({"פס חושך n": n, "yₙ [mm]": yn, "Dₙ = 2yₙ [mm]": 2 * yn})
    st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True, hide_index=True)


def tab_check(lam_nm):
    st.markdown("מזינים את הנתונים מהדוח ומקבלים את התוצאה הנכונה, הסטייה וסימנים לטעויות נפוצות.")
    lam0 = lam_nm * NM_TO_MM
    kind = st.radio("איזה חלק בודקים?", ["ניסוי יאנג", "סדק יחיד", "שריג", "שערה"],
                    horizontal=True, key="c_kind")
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("##### אי-ודאות במדידה")
        dD = st.number_input("δD (קריאת סרגל) [mm]", min_value=0.0, value=0.5, step=0.1, key="c_dD")
        dL = st.number_input("δL [mm]", min_value=0.0, value=2.0, step=0.5, key="c_dL")

    if kind == "ניסוי יאנג":
        with c1:
            st.markdown("##### נתוני הדוח")
            d = st.number_input("d [mm]", min_value=0.001, value=0.250, step=0.005, format="%.3f", key="c_y_d")
            L = st.number_input("L [mm]", min_value=1.0, value=550.0, step=5.0, key="c_y_L")
            D = st.number_input("D: מרחק שנמדד על המסך [mm]", min_value=0.01, value=21.0, step=0.5, key="c_y_D")
            N = st.number_input("מספר פסי אור שנספרו N", min_value=2, value=15, step=1, key="c_y_N")
            conv = st.radio("איך חולק", ["N−1 מרווחים (נכון כשסופרים פסים)", "N מרווחים"],
                            key="c_y_conv")
        k = N - 1 if conv.startswith("N−1") else N
        dx = D / k
        lam_m = dx * d / L
        rel = math.sqrt((dD / D) ** 2 + (dL / L) ** 2)
        other = D / (N if k == N - 1 else N - 1) * d / L
        with c2:
            show_result_lambda(lam_m, lam0, rel, L)
            st.caption(f"עם החלוקה השנייה ({'N' if k == N - 1 else 'N−1'}) היה מתקבל "
                       f"λ = {other / NM_TO_MM:.1f} nm.")

    elif kind == "סדק יחיד":
        with c1:
            st.markdown("##### נתוני הדוח")
            a = st.number_input("a [mm]", min_value=0.001, value=0.080, step=0.005, format="%.3f", key="c_s_a")
            L = st.number_input("L [mm]", min_value=1.0, value=550.0, step=5.0, key="c_s_L")
            n = st.number_input("סדר פס החושך n", min_value=1, value=1, step=1, key="c_s_n")
            D = st.number_input("Dₙ: בין פסי החושך משני הצדדים [mm]", min_value=0.01, value=9.5,
                                step=0.5, key="c_s_D")
        y = D / 2
        lam_m = a * (y / math.sqrt(y ** 2 + L ** 2)) / n
        rel = math.sqrt(((dD / 2) / y) ** 2 + (dL / L) ** 2)
        with c2:
            show_result_lambda(lam_m, lam0, rel, L)
            lam_nodiv = a * (D / math.sqrt(D ** 2 + L ** 2)) / n
            st.caption(f"בלי לחלק ב-2 היה מתקבל λ = {lam_nodiv / NM_TO_MM:.1f} nm.")

    elif kind == "שריג":
        with c1:
            st.markdown("##### נתוני הדוח")
            lpm_nom = st.number_input("צפיפות רשומה על השריג [קווים למ\"מ]", min_value=1.0, value=50.0,
                                      step=10.0, key="c_g_lpm")
            L = st.number_input("L [mm]", min_value=1.0, value=230.0, step=5.0, key="c_g_L")
            n = st.number_input("סדר n", min_value=1, value=1, step=1, key="c_g_n")
            D = st.number_input("Dₙ: בין הנקודות +n ו-−n [mm]", min_value=0.01, value=10.0,
                                step=0.5, key="c_g_D")
        X = D / 2
        s = X / math.sqrt(X ** 2 + L ** 2)
        d_m = n * lam0 / s
        d_sa = n * lam0 * L / X
        d_nom = 1.0 / lpm_nom
        cos2 = L ** 2 / (X ** 2 + L ** 2)
        rel = cos2 * math.sqrt(((dD / 2) / X) ** 2 + (dL / L) ** 2)
        with c2:
            st.markdown("##### תוצאה")
            m1, m2 = st.columns(2)
            m1.metric("d מחושב (עם 632.8 nm)", f"{d_m * 1000:.2f} µm",
                      delta=f"{pct_dev(d_m, d_nom):+.1f}% מהרשום", delta_color="off")
            m2.metric("צפיפות קווים מחושבת", f"{1 / d_m:.1f} למ\"מ")
            st.metric("אי-ודאות", f"± {rel * d_m * 1000:.2f} µm ({rel * 100:.1f}%)")
            st.caption(f"הערך הרשום: d = {d_nom * 1000:.2f} µm. "
                       f"בקירוב זווית קטנה היה מתקבל d = {d_sa * 1000:.2f} µm.")
            dev = abs(pct_dev(d_m, d_nom))
            if dev > 3 * rel * 100 and dev > 5:
                st.warning("הסטייה מהערך הרשום גדולה בהרבה מאי-הוודאות. כדאי לבדוק את L, "
                           "את זיהוי הסדר n, ואם הם חילקו את D ב-2.")
            else:
                st.success("התוצאה מתיישבת עם הערך הרשום בתוך אי-הוודאות.")

    else:  # שערה
        with c1:
            st.markdown("##### נתוני הדוח")
            L = st.number_input("L [mm]", min_value=1.0, value=550.0, step=5.0, key="c_h_L")
            n = st.number_input("סדר פס החושך n", min_value=1, value=1, step=1, key="c_h_n")
            D = st.number_input("Dₙ: בין פסי החושך משני הצדדים [mm]", min_value=0.01, value=15.0,
                                step=0.5, key="c_h_D")
            lam_used = st.number_input("λ שהסטודנטים השתמשו בו [nm]", min_value=100.0,
                                       value=float(lam_nm), step=1.0, key="c_h_lam")
        y = D / 2
        s = y / math.sqrt(y ** 2 + L ** 2)
        a_true = n * lam0 / s
        a_their = n * lam_used * NM_TO_MM / s
        rel = math.sqrt(((dD / 2) / y) ** 2 + (dL / L) ** 2)
        with c2:
            st.markdown("##### תוצאה")
            st.metric("עובי השערה (עם 632.8 nm)", f"{a_true * 1000:.1f} µm",
                      help=f"אי-ודאות ± {rel * a_true * 1000:.1f} µm")
            if abs(lam_used - lam_nm) > 0.5:
                st.metric("עם ה-λ שהסטודנטים השתמשו בו", f"{a_their * 1000:.1f} µm")
            if not 15 <= a_true * 1000 <= 200:
                st.warning("העובי מחוץ לטווח הסביר לשערה (בערך 20–180 µm). "
                           "כדאי לבדוק את החלוקה ב-2 ואת היחידות.")
            else:
                st.success("העובי בטווח הסביר לשערה אנושית.")


def show_result_lambda(lam_m, lam0, rel, L):
    """תצוגת תוצאה משותפת לניסויים שבהם מחשבים את אורך הגל."""
    dev = pct_dev(lam_m, lam0)
    st.markdown("##### תוצאה")
    m1, m2 = st.columns(2)
    m1.metric("λ מחושב", f"{lam_m / NM_TO_MM:.1f} nm",
              delta=f"{dev:+.1f}% מ-{lam0 / NM_TO_MM:.1f}", delta_color="off")
    m2.metric("אי-ודאות", f"± {rel * lam_m / NM_TO_MM:.1f} nm ({rel * 100:.1f}%)")

    ratio = lam_m / lam0
    if 1.8 < ratio < 2.2:
        st.error("התוצאה גדולה פי 2 בערך. כנראה שלא חילקו את D ב-2.")
    elif 0.45 < ratio < 0.55:
        st.error("התוצאה קטנה פי 2 בערך. כנראה שחילקו ב-2 פעם אחת יותר מדי.")
    elif abs(dev) > 3 * rel * 100 and abs(dev) > 3:
        L_eff = L * lam_m / lam0
        st.warning(
            f"הסטייה ({dev:+.1f}%) גדולה בהרבה מאי-הוודאות ({rel * 100:.1f}%), ולכן יש שגיאה שיטתית. "
            f"למשל, L = {L_eff:.0f} mm היה נותן בדיוק {lam0 / NM_TO_MM:.1f} nm. "
            "אם כל החלקים בדוח סוטים באותו כיוון, כנראה ש-L נמדד מנקודת ייחוס שגויה."
        )
    else:
        st.success("התוצאה מתיישבת עם הערך הידוע בתוך אי-הוודאות.")


# -----------------------------
# נקודת כניסה
# -----------------------------
def render_interference():
    st.title("🔴 התאבכות ועקיפה")
    st.caption("סימולציה של תמונת המסך הצפויה, ובדיקת התוצאות שהסטודנטים קיבלו")

    with st.expander("הנחות המודל — חשוב לקרוא"):
        st.markdown(
            """
            הסימולציה משתמשת ב**עקיפת פראונהופר (שדה רחוק)**, עם חישוב זווית מדויק
            sinθ = x / √(x² + L²), כך שגם זוויות גדולות (שריג) מחושבות נכון.

            - **שני סדקים:** I = I₀·cos²(πd·sinθ/λ)·[sinβ/β]², כאשר β = πa·sinθ/λ.
            - **סדק יחיד:** I = I₀·[sinβ/β]².
            - **שריג:** מוצגים מיקומי הסדרים לפי d·sinθ = nλ. העוצמה היחסית של הסדרים
              תלויה ברוחב החריצים, שאינו ידוע, ולכן אינה מוצגת.
            - **שערה:** לפי עיקרון בבינה, אותה תבנית כמו סדק באותו רוחב, פרט לכתם המרכזי.

            התמונה המדומה מוצגת עם "חשיפה" מתכווננת, כמו מצלמה: חשיפה גבוהה
            מראה את הפסים החלשים ומרוויה את המרכז, כמו בצילומים שהסטודנטים מצרפים.
            """
        )

    g1, g2 = st.columns(2)
    with g1:
        lam_nm = st.number_input("אורך גל λ [nm]", min_value=300.0, max_value=1100.0,
                                 value=HENE_NM, step=0.1, key="lam_nm",
                                 help="לייזר הליום-ניאון: 632.8 nm")
    with g2:
        exposure = st.slider("חשיפה (בהירות התמונה המדומה)", 1.0, 60.0, 8.0, 1.0, key="exposure")

    t1, t2, t3, t4, t5 = st.tabs([
        "שני סדקים (יאנג)",
        "סדק יחיד",
        "שריג",
        "שערה",
        "בדיקת תוצאות מדוח",
    ])
    with t1:
        tab_young(lam_nm, exposure)
    with t2:
        tab_single(lam_nm, exposure)
    with t3:
        tab_grating(lam_nm, exposure)
    with t4:
        tab_hair(lam_nm, exposure)
    with t5:
        tab_check(lam_nm)

    st.divider()
    st.caption("Optics Lab Designer — הסימולציה היא תחזית. תמונה אמיתית מושפעת גם מאיכות הסדקים, "
               "מרוחב האלומה וממיקוד הלייזר.")
