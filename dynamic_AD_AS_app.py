# dynamic_AD_AS_app.py
# Streamlit app for the Dynamic AD-AS model
# Controls for monetary policy rule parameters: phi_pi, phi_Y, and pi_star (target)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Dynamic AD–AS: Monetary Policy Explorer", layout="wide")

# -------------------------
# Model parameters (fixed, not policy)
# -------------------------
alpha = 1.0
rho   = 2.0
phi   = 0.25  # slope in the inflation update (Phillips side)
T_default = 65
Ybar0 = 100.0  # natural output (baseline)

# -------------------------
# Core model functions
# -------------------------
def pi_update(pi_tm1, v_t, eps_t, pi_star_t, phi_pi, phi_Y):
    denom = 1.0 + alpha * (phi_Y + phi * phi_pi)
    A = (1.0 + alpha * phi_Y) / denom
    B =  phi / denom
    C = (phi * alpha * phi_pi) / denom
    return A*(pi_tm1 + v_t) + B*eps_t + C*pi_star_t

def Y_from_DAD(pi_t, eps_t, Ybar_t, pi_star_t, phi_pi, phi_Y):
    factor = 1.0 + alpha * phi_Y
    return Ybar_t - (alpha*phi_pi)/factor * (pi_t - pi_star_t) + (1.0/factor)*eps_t

def i_from_rule(pi_t, Y_t, Ybar_t, pi_star_t, phi_pi, phi_Y):
    # Interest-rate rule:
    # i_t = pi_t + rho + phi_pi*(pi_t - pi_star_t) + phi_Y*(Y_t - Ybar_t)
    return pi_t + rho + phi_pi*(pi_t - pi_star_t) + phi_Y*(Y_t - Ybar_t)

def simulate(T, Ybar0, pi_star_base, phi_pi, phi_Y,
             Ybar_path=None, pi_star_path=None,
             v_shock=None, eps_shock=None):
    Ybar_t    = np.full(T, Ybar0) if Ybar_path    is None else np.array(Ybar_path, dtype=float)
    pi_star_t = np.full(T, pi_star_base) if pi_star_path is None else np.array(pi_star_path, dtype=float)
    v_t   = np.zeros(T)
    eps_t = np.zeros(T)

    if v_shock:
        for k, v in v_shock.items():
            if 0 <= k < T:
                v_t[k] = v
    if eps_shock:
        for k, v in eps_shock.items():
            if 0 <= k < T:
                eps_t[k] = v

    pi = np.zeros(T); Y = np.zeros(T); i = np.zeros(T); r = np.zeros(T)

    # t=0 (start in steady state at given Ybar_0 and pi_star_0)
    pi[0] = pi_star_t[0]
    Y[0]  = Y_from_DAD(pi[0], eps_t[0], Ybar_t[0], pi_star_t[0], phi_pi, phi_Y)
    i[0]  = i_from_rule(pi[0], Y[0], Ybar_t[0], pi_star_t[0], phi_pi, phi_Y)
    r[0]  = i[0] - pi[0]

    for t in range(1, T):
        pi[t] = pi_update(pi[t-1], v_t[t], eps_t[t], pi_star_t[t], phi_pi, phi_Y)
        Y[t]  = Y_from_DAD(pi[t], eps_t[t], Ybar_t[t], pi_star_t[t], phi_pi, phi_Y)
        i[t]  = i_from_rule(pi[t], Y[t], Ybar_t[t], pi_star_t[t], phi_pi, phi_Y)
        r[t]  = i[t] - pi[t]

    return {
        "t": np.arange(T),
        "v_t": v_t, "eps_t": eps_t, "pi_t": pi, "Y_t": Y,
        "Y_n": Ybar_t, "pi_star": pi_star_t, "i_t": i, "r_t": r
    }

# -------------------------
# Diagnostics (amplitude & persistence)
# -------------------------
def compute_metrics(res):
    """Peak deviations and time-to-band (persistence) for Y and pi."""
    Y_gap = res["Y_t"] - res["Y_n"]
    pi_gap = res["pi_t"] - res["pi_star"]  # gap vs target at each t

    amp_Y  = float(np.max(np.abs(Y_gap)))
    amp_pi = float(np.max(np.abs(pi_gap)))

    # time to return within small band around steady state (0.01)
    band_Y  = 1e-2
    band_pi = 1e-2

    T = len(res["t"])
    tt_Y  = next((int(t) for t in range(T) if abs(Y_gap[t]) < band_Y), None)
    tt_pi = next((int(t) for t in range(T) if abs(pi_gap[t]) < band_pi), None)

    return {
        "amp_Y": amp_Y, "amp_pi": amp_pi,
        "time_to_band_Y": tt_Y, "time_to_band_pi": tt_pi
    }

# -------------------------
# Plotting
# -------------------------
def plot_four_panel(res, title):
    t = res["t"]

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.6), constrained_layout=True)

    # Output
    axs[0,0].plot(t, res["Y_t"], label="Y_t")
    axs[0,0].axhline(res["Y_n"][0], ls="--", c="gray", lw=1)
    axs[0,0].set_title(f"{title}: Output Y_t")
    axs[0,0].set_xlabel("t"); axs[0,0].set_ylabel("Y_t"); axs[0,0].grid(True); axs[0,0].legend()

    # Inflation
    axs[0,1].plot(t, res["pi_t"], label="pi_t", color="C1")
    axs[0,1].axhline(res["pi_star"][0], ls="--", c="gray", lw=1)
    axs[0,1].set_title(f"{title}: Inflation pi_t")
    axs[0,1].set_xlabel("t"); axs[0,1].set_ylabel("pi_t"); axs[0,1].grid(True); axs[0,1].legend()

    # Nominal rate
    axs[1,0].plot(t, res["i_t"], label="i_t", color="C2")
    axs[1,0].set_title(f"{title}: Nominal rate i_t")
    axs[1,0].set_xlabel("t"); axs[1,0].set_ylabel("i_t"); axs[1,0].grid(True); axs[1,0].legend()

    # Real rate
    axs[1,1].plot(t, res["r_t"], label="r_t", color="C3")
    axs[1,1].set_title(f"{title}: Real rate r_t")
    axs[1,1].set_xlabel("t"); axs[1,1].set_ylabel("r_t"); axs[1,1].grid(True); axs[1,1].legend()

    return fig

# -------------------------
# UI
# -------------------------
st.title("Dynamic AD–AS: Monetary Policy Explorer")
st.caption("Note: In the code/figures, Ybar corresponds to \\bar{Y} and pi_star to \\pi^*.")

with st.sidebar:
    st.header("Policy rule parameters (central bank)")
    phi_pi = st.slider("phi_pi (response to inflation gap)", 0.0, 2.0, 0.5, 0.05)
    phi_Y  = st.slider("phi_Y (response to output gap)",     0.0, 1.0, 0.5, 0.05)

    st.header("Inflation target (pi_star)")
    pi_star_base = st.number_input("Initial target pi_star at t=0", value=2.0, step=0.1, format="%.1f")
    change_pi_star = st.checkbox("Permanent change in pi_star from a chosen period", value=False)
    t_change = 1
    pi_star_new = pi_star_base
    if change_pi_star:
        t_change = st.number_input("Change period (t >= 1)", min_value=1, max_value=500, value=1, step=1)
        pi_star_new = st.number_input("New target (pi_star) from t_change onward", value=1.0, step=0.1, format="%.1f")

    st.header("Scenario and horizon")
    scenario = st.radio(
        "Shock scenario",
        ["No shock (policy only)",
         "Increase in Ybar (permanent from t=3 to 110)",
         "Supply shock: one-period v_t = +1 at t=1",
         "Demand shock: five-period eps_t = +1 (t=1..5)",
         "Policy only: permanent reduction in pi_star (from t=1)"],
        index=0
    )
    T = st.number_input("Time horizon T", min_value=10, max_value=500, value=T_default, step=5)

# Build scenario paths
Ybar_path = None
pi_star_path = None
v_shock = None
eps_shock = None
title = "Baseline"

if scenario == "No shock (policy only)":
    title = "Policy change only" if change_pi_star else "No shock"

elif scenario == "Increase in Ybar (permanent from t=3 to 110)":
    Ybar_path = np.full(T, Ybar0)
    if T > 3:
        Ybar_path[3:] = 110.0
    title = "Increase in Ybar"

elif scenario == "Supply shock: one-period v_t = +1 at t=1":
    v_shock = {1: 1.0} if T > 1 else None
    title = "Supply shock (1 period)"

elif scenario == "Demand shock: five-period eps_t = +1 (t=1..5)":
    eps_shock = {t: 1.0 for t in range(1, min(6, T))}
    title = "Demand shock (5 periods)"

elif scenario == "Policy only: permanent reduction in pi_star (from t=1)":
    change_pi_star = True
    t_change = 1
    pi_star_new = min(pi_star_base, 1.0)  # suggestive default
    title = "Permanent pi* reduction"

# Build pi_star path if policy change is requested
if change_pi_star:
    pi_star_path = np.full(T, pi_star_base)
    if t_change < T:
        pi_star_path[t_change:] = pi_star_new

# Run simulation
res = simulate(T=T, Ybar0=Ybar0, pi_star_base=pi_star_base,
               phi_pi=phi_pi, phi_Y=phi_Y,
               Ybar_path=Ybar_path, pi_star_path=pi_star_path,
               v_shock=v_shock, eps_shock=eps_shock)

# Show metrics and table
metrics = compute_metrics(res)
left, right = st.columns([1,1])

with left:
    st.subheader("Amplitude and persistence (diagnostics)")
    st.markdown(
        f"- Peak |Y_t - Ybar_t|: **{metrics['amp_Y']:.4f}**  \n"
        f"- Peak |pi_t - pi_star|: **{metrics['amp_pi']:.4f}**  \n"
        f"- Time to |Y gap| < 0.01: **{metrics['time_to_band_Y']}**  \n"
        f"- Time to |pi gap| < 0.01: **{metrics['time_to_band_pi']}**"
    )

with right:
    st.subheader("First 6 periods")
    df6 = pd.DataFrame({
        "v_t":     res["v_t"][:6],
        "eps_t":   res["eps_t"][:6],
        "pi_t":    res["pi_t"][:6],
        "Y_t":     res["Y_t"][:6],
        "Y_n":     res["Y_n"][:6],
        "pi_star": res["pi_star"][:6],
        "i_t":     res["i_t"][:6],
        "r_t":     res["r_t"][:6],
    })
    st.dataframe(df6.style.format(precision=4), use_container_width=True)

# Plots
st.subheader(f"Dynamic paths — {title}")
fig = plot_four_panel(res, title)
st.pyplot(fig, clear_figure=True)

# Notes for Exercise 11.2
st.markdown("""
**How to use for Exercise 11.2 (Interpretation of monetary policy)**  
- Increase `phi_pi` to see how a stronger response to the inflation gap reduces the peak of inflation and shortens its persistence, usually at the cost of larger short-run movements in the interest rate.  
- Increase `phi_Y` to see how reacting to the output gap stabilizes output but may slow the disinflation path under supply shocks.  
- Change `pi_star` (permanently) to study how lowering the target transmits to activity (`Y_t`) and inflation (`pi_t`): output dips below potential temporarily, inflation gradually converges to the new target, and the nominal rate settles at `rho + pi_star` in the long run.  
""")
