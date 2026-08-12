import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as rx
from plotly.subplots import make_subplots

# Asetetaan sivun leveä asettelu
st.set_page_config(page_title="Salkun Monte Carlo -simulaattori", layout="wide")

np.random.seed(42)

# --- SIVUPALKKI: PARAMETRIT ---
st.sidebar.header("Simulaation parametrit")

years = st.sidebar.slider("Vuodet (years)", min_value=5, max_value=50, value=30, step=1)
months = years * 12

monthly_inflation_rate = st.sidebar.number_input(
    "Kuukausi-inflaatio (monthly_inflation_rate)",
    min_value=0.0, max_value=0.01, value=0.0016, step=0.0001, format="%.4f"
)

n_simulations = st.sidebar.number_input(
    "Simulaatioiden määrä (n_simulations)",
    min_value=100, max_value=5000, value=1000, step=100
)

hide_euros = st.sidebar.checkbox("Piilota euroarvot (hide_euros)", value=False)

initial_capital = st.sidebar.number_input(
    "Alkupääoma (€)",
    min_value=0, value=1000000, step=10000
)

# --- SIVUPALKKI: SKENAARIOT ---
st.sidebar.subheader("Skenaariot")
default_scenarios = pd.DataFrame([
    {"Nimi": "40/40/20 Nollatuotto", "Tuotto (%)": 0.0, "Kulu (%)": 0.36, "Volatiteetti (%)": 0.0},
    {"Nimi": "Pankkitalletus", "Tuotto (%)": 2.0, "Kulu (%)": 0.00, "Volatiteetti (%)": 0.0},
    {"Nimi": "S-Varainhoito 100", "Tuotto (%)": 7.08, "Kulu (%)": 1.11, "Volatiteetti (%)": 15.23},
    {"Nimi": "Nordnet rohkea", "Tuotto (%)": 7.08, "Kulu (%)": 0.50, "Volatiteetti (%)": 14.76},
    {"Nimi": "Villen salkku", "Tuotto (%)": 6.16, "Kulu (%)": 1.49, "Volatiteetti (%)": 9.52},
    {"Nimi": "40/40/20 ETF & Rohkea & Talletus", "Tuotto (%)": 6.33, "Kulu (%)": 0.33, "Volatiteetti (%)": 11.8},
    {"Nimi": "80/20 Rohkea & Talletus", "Tuotto (%)": 6.16, "Kulu (%)": 0.44, "Volatiteetti (%)": 11.5},
    {"Nimi": "80/20 ETF & Talletus", "Tuotto (%)": 7.54, "Kulu (%)": 0.17, "Volatiteetti (%)": 15.23}
])

scenarios_df = st.sidebar.data_editor(
    default_scenarios,
    num_rows="dynamic",
    key="scenarios_editor"
)

# --- PÄÄSIVU: TAULUKOIDEN MUOKKAUS ---
st.title("Salkun Monte Carlo -simulaattori")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Kuukausittaiset Menot")
    default_expenses = pd.DataFrame([
        {"Nimi": "Peruselämisen kulut", "Määrä (€)": 4000, "Alku (kk)": 1, "Loppu (kk)": 0, "Inflaatiokorjaus": True}#,
        #{"Nimi": "Kulu2", "Määrä (€)": 1000, "Alku (kk)": 1, "Loppu (kk)": 2, "Inflaatiokorjaus": False}
    ])
    st.caption("Aseta 'Loppu (kk)' arvoksi 0, jos meno jatkuu simulaation loppuun asti.")
    expenses_df = st.data_editor(default_expenses, num_rows="dynamic", key="expenses_editor")

with col2:
    st.subheader("Kuukausittaiset Tulot / Eläkkeet")
    default_incomes = pd.DataFrame([
        {"Nimi": "V Varhennettu eläke1", "Määrä (€)": 1097, "Alku (kk)": 4 * 12 + 4, "Loppu (kk)": 7 * 12 + 4},
        {"Nimi": "V Vanhuuseläke", "Määrä (€)": 2198, "Alku (kk)": 7 * 12 + 5, "Loppu (kk)": 0} ,
        {"Nimi": "T Varhennettu eläke", "Määrä (€)": 500, "Alku (kk)": 7 * 12 + 4, "Loppu (kk)": 10 * 12 + 4},
        {"Nimi": "T vanhuuseläke", "Määrä (€)": 900, "Alku (kk)": 10 * 12 + 5, "Loppu (kk)": 0}
    ])
    st.caption("Aseta 'Loppu (kk)' arvoksi 0, jos tulo jatkuu simulaation loppuun asti.")
    incomes_df = st.data_editor(default_incomes, num_rows="dynamic", key="incomes_editor")


# --- SIMULAATIOALGORITMI ---
def run_monte_carlo(start_sum, exp_df, inc_df, ann_ret, ann_cost, ann_vol, periods, n_sims):
    net_ann_ret = ann_ret - ann_cost
    dt = 1 / 12
    monthly_drift = (net_ann_ret - 0.5 * (ann_vol ** 2)) * dt
    monthly_vol = ann_vol * np.sqrt(dt)

    random_shocks = np.random.normal(0, 1, size=(n_sims, periods))
    monthly_returns = np.exp(monthly_drift + monthly_vol * random_shocks) - 1

    balances = np.zeros((n_sims, periods + 1))
    balances[:, 0] = start_sum

    for m in range(1, periods + 1):
        current_balance = balances[:, m - 1].copy()

        # Menot
        for _, row in exp_df.iterrows():
            start = int(row["Alku (kk)"])
            end = int(row["Loppu (kk)"])
            if m >= start and (end == 0 or m <= end):
                amt = row["Määrä (€)"]
                cost = amt + (amt * monthly_inflation_rate * m) if row["Inflaatiokorjaus"] else amt
                current_balance -= cost

        # Tulot
        for _, row in inc_df.iterrows():
            start = int(row["Alku (kk)"])
            end = int(row["Loppu (kk)"])
            if m >= start and (end == 0 or m <= end):
                current_balance += row["Määrä (€)"]

        # Kasvu
        current_balance *= (1 + monthly_returns[:, m - 1])
        balances[:, m] = np.maximum(0, current_balance)

    return balances


# --- LASKENTA ---
mc_results = {}
for _, row in scenarios_df.iterrows():
    name = row["Nimi"]
    ret = row["Tuotto (%)"] / 100
    cost = row["Kulu (%)"] / 100
    vol = row["Volatiteetti (%)"] / 100

    sim_data = run_monte_carlo(
        initial_capital, expenses_df, incomes_df,
        ret, cost, vol, months, int(n_simulations)
    )

    sim_pct_changes = ((sim_data - initial_capital) / initial_capital) * 100

    mc_results[name] = {
        "median_pct": np.median(sim_pct_changes, axis=0),
        "p10_pct": np.percentile(sim_pct_changes, 10, axis=0),
        "p90_pct": np.percentile(sim_pct_changes, 90, axis=0),
        "median_abs": np.median(sim_data, axis=0),
        "p10_abs": np.percentile(sim_data, 10, axis=0),
        "p90_abs": np.percentile(sim_data, 90, axis=0)
    }

# --- VISUALISOINTI (PLOTLY SUBPLOTS) ---
st.markdown("---")
st.subheader("Simulaation tulokset")

time_axis = np.arange(months + 1) / 12

# Luodaan ruudukko kuvaajille
n_scen = len(mc_results)
cols = 2
rows = (n_scen + 1) // 2

fig = make_subplots(rows=rows, cols=cols, subplot_titles=list(mc_results.keys()))

for idx, (name, data) in enumerate(mc_results.items()):
    r = (idx // cols) + 1
    c = (idx % cols) + 1

    # 10% ja 90% rajat (täytetty alue)
    fig.add_trace(
        rx.Scatter(
            x=np.concatenate([time_axis, time_axis[::-1]]),
            y=np.concatenate([data["p90_pct"], data["p10_pct"][::-1]]),
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
        row=r, col=c
    )

    # Medianaviiva
    fig.add_trace(
        rx.Scatter(
            x=time_axis, y=data["median_pct"],
            mode='lines',
            name=f"{name} (Mediana)",
            line=dict(width=2)
        ),
        row=r, col=c
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=r, col=c)

fig.update_layout(height=300 * rows, width=1100, title_text="Salkun prosentuaalinen kehitys (0% = lähtötaso)",
                  showlegend=False)
fig.update_yaxes(ticksuffix=" %")
fig.update_xaxes(title_text="Vuodet")

st.plotly_chart(fig, use_container_width=True)

# --- NUMEERISET TULOKSET TAULUKOSSA ---
st.subheader(f"Lopputulokset {years} vuoden kohdalla")
summary_rows = []

for name, data in mc_results.items():
    if hide_euros:
        summary_rows.append({
            "Skenaario": name,
            "Mediana (%)": f"{data['median_pct'][-1]:+.1f} %",
            "10% Alalaita (%)": f"{data['p10_pct'][-1]:+.1f} %",
            "90% Ylälaita (%)": f"{data['p90_pct'][-1]:+.1f} %"
        })
    else:
        summary_rows.append({
            "Skenaario": name,
            "Mediana (€)": f"{data['median_abs'][-1]:,.0f} € ({data['median_pct'][-1]:+.1f} %)",
            "10% Alalaita (€)": f"{data['p10_abs'][-1]:,.0f} € ({data['p10_pct'][-1]:+.1f} %)",
            "90% Ylälaita (€)": f"{data['p90_abs'][-1]:,.0f} € ({data['p90_pct'][-1]:+.1f} %)"
        })

st.table(pd.DataFrame(summary_rows))
