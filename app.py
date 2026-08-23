import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as rx
from plotly.subplots import make_subplots
import os

# Yritetään tuoda openai-kirjasto
try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Asetetaan sivun leveä asettelu
st.set_page_config(page_title="Manilaattori - taloutesi tulevaisuus", layout="wide")

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

# UUSI VALINTA: Oletuksena False (näytetään tulevat eurot ilman inflaatiodiskonttausta)
use_real_value = st.sidebar.checkbox("Näytä arvot rahan nykyarvossa (reaaliarvo / ostovoima)", value=False)

initial_capital = st.sidebar.number_input(
    "Alkupääoma (€)",
    min_value=0, value=300000, step=10000
)

# --- SIVUPALKKI: TEKOÄLYN KEHOTE (PROMPT) ---
st.sidebar.subheader("Tekoälyn kehote (Prompt)")
default_prompt = (
    "Analysoi syöttetyt tulo- ja menotiedot ja arvioi miltä taloudellinen tulevaisuuteni näyttää simulaation eri vaihtoehdoissa.\n"
    "Pyri vastaamaan seuraaviin kysymyksiin:\n"
    "- Riittääkö varallisuuteni elämiseen simulaation ajan ja paljonko varallisuutta on jäljellä simulaation lopussa?\n"
    "- Onko kulutasoni oikeassa suhteessa tuloihin ja sijoitustuottoihin?\n"
    "Vertaile eri esimerkkisalkkujen mediaani, 10% ja 90% -tuloksia tuloksen ja riskitason osalta.\n"
    "Selitä mitä eri tulosvaihtoehdot tarkoittavat siten että maallikko ymmärtää vaihtoehtojen erot. \n"
    "Muodosta koko analyysi suomen kielellä ja käytä maallikon ymmärtämiä termejä.\n\n"
    "{summary_prompt_data}"
)

ai_prompt_template = st.sidebar.text_area(
    "Muokkaa tekoälylle lähetettävää kehotetta",
    value=default_prompt,
    height=220,
    help="Voit muokata ohjeita tekoälylle. Pidä tekstillä paikkamerkki {summary_prompt_data}, johon simulaation tulokset sijoitetaan."
)

# --- SIVUPALKKI: SKENAARIOT ---
st.sidebar.subheader("Skenaariot")
default_scenarios = pd.DataFrame([
    {"Nimi": "Sukan varsi", "Tuotto (%)": 0.0, "Kulu (%)": 0.0, "Volatiteetti (%)": 0.0},
    {"Nimi": "Pankkitalletus", "Tuotto (%)": 2.0, "Kulu (%)": 0.00, "Volatiteetti (%)": 0.0},
    {"Nimi": "S-Varainhoito 100", "Tuotto (%)": 7.08, "Kulu (%)": 1.11, "Volatiteetti (%)": 15.23},
    {"Nimi": "Nordnet rohkea", "Tuotto (%)": 7.08, "Kulu (%)": 0.50, "Volatiteetti (%)": 14.76},
    {"Nimi": "Sekasalkku", "Tuotto (%)": 6.16, "Kulu (%)": 1.49, "Volatiteetti (%)": 9.52},
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
st.title("Manilaattori - taloutesi tulevaisuus")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Kuukausittaiset Menot")
    default_expenses = pd.DataFrame([
        {"Nimi": "Peruselämisen kulut", "Määrä (€)": 3100, "Alku (kk)": 1, "Loppu (kk)": 0, "Inflaatiokorjaus": True}
    ])
    st.caption("Aseta 'Loppu (kk)' arvoksi 0, jos meno jatkuu simulaation loppuun asti.")
    expenses_df = st.data_editor(default_expenses, num_rows="dynamic", key="expenses_editor")

with col2:
    st.subheader("Kuukausittaiset Tulot / Eläkkeet")
    default_incomes = pd.DataFrame([
        {"Nimi": "Tulo1", "Määrä (€)": 3000, "Alku (kk)": 1, "Loppu (kk)": 0}
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

# --- NUMEERISET TULOKSET TAULUKOSSA ---
st.subheader(f"Lopputulokset {years} vuoden kohdalla")
summary_rows = []

# Inflaatiokerroin ostovoiman laskentaa varten
inflation_discount_factor = (1 + monthly_inflation_rate) ** months

for name, data in mc_results.items():
    if hide_euros:
        summary_rows.append({
            "Skenaario": name,
            "Mediana (%)": f"{data['median_pct'][-1]:+.1f} %",
            "10% Alalaita (%)": f"{data['p10_pct'][-1]:+.1f} %",
            "90% Ylälaita (%)": f"{data['p90_pct'][-1]:+.1f} %"
        })
    else:
        med_val = data['median_abs'][-1]
        p10_val = data['p10_abs'][-1]
        p90_val = data['p90_abs'][-1]

        # Jos käyttäjä valitsi nykyarvon, alennetaan eurot inflaatiolla
        if use_real_value:
            med_val /= inflation_discount_factor
            p10_val /= inflation_discount_factor
            p90_val /= inflation_discount_factor
            header_suffix = " (Nykyeuroina)"
        else:
            header_suffix = " (Tulevat €)"

        summary_rows.append({
            "Skenaario": name,
            f"Mediana{header_suffix}": f"{med_val:,.0f} € ({data['median_pct'][-1]:+.1f} %)",
            f"10% Alalaita{header_suffix}": f"{p10_val:,.0f} €",
            f"90% Ylälaita{header_suffix}": f"{p90_val:,.0f} €"
        })

st.table(pd.DataFrame(summary_rows))

# --- VISUALISOINTI (PLOTLY SUBPLOTS) ---
st.markdown("---")
st.subheader("Simulaation tulokset")

time_axis = np.arange(months + 1) / 12

n_scen = len(mc_results)
cols = 2
rows = (n_scen + 1) // 2

fig = make_subplots(rows=rows, cols=cols, subplot_titles=list(mc_results.keys()))

for idx, (name, data) in enumerate(mc_results.items()):
    r = (idx // cols) + 1
    c = (idx % cols) + 1

    # 10% ja 90% rajat
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


# --- INFLAATION VAIKUTUS TULOSTAULUKON JÄLKEEN ---
future_price = 100 * inflation_discount_factor
if use_real_value:
    st.caption(f"💡 **Ostovoimakorjaus päällä:** Taulukon euroarvot on diskontattu vastaamaan tämän päivän ostovoimaa. Simulaation alussa 100 € maksanut tuote maksaa simulaation päättyessä {future_price:.2f} €.")
else:
    st.caption(f"💡 **Tulevaisuuden eurot (ilman ostovoimakorjausta):** Simulaation alussa 100 € maksanut tuote maksaa simulaation päättyessä {future_price:.2f} €. Voit halutessasi vaihtaa arvot rahan nykyarvoon sivupalkista.")

# --- TEKOÄLYANALYYSI -OSIO ---
st.markdown("---")
st.subheader("🤖 Tekoälyneuvojan analyysi")

# Luodaan painike tekoälyanalyysin käynnistämiseksi
if st.button("Luo tekoälyanalyysi tuloksista", type="primary"):
    api_key = st.secrets.get("Salkku_KEY", None)

    if not HAS_OPENAI:
        st.error("Puuttuva kirjasto: Asenna `openai` komennolla: `pip install openai`")
    elif not api_key:
        st.warning("Syötä OpenAI API Key sivupalkkiin (sidebar) laittaaksesi tekoälykommentoinnin päälle.")
    else:
        with st.spinner("Tekoäly analysoi simulaation tuloksia..."):
            try:
                # Muotoillaan tulokset tekstiksi tekoälyä varten
                val_type_str = "reaaliarvoina / nykyeuroina" if use_real_value else "nimellisarvoina / tulevaisuuden euroina"
                summary_prompt_data = f"Monte Carlo -simulaation tulokset {years} vuoden ajalta ({val_type_str}):\n\n"

                for name, res in mc_results.items():
                    med_pct = res["median_pct"][-1]
                    p10_pct = res["p10_pct"][-1]
                    p90_pct = res["p90_pct"][-1]

                    if hide_euros:
                        summary_prompt_data += f"- {name}: Mediana {med_pct:+.1f}%, 10% alalaita {p10_pct:+.1f}%, 90% ylälaita {p90_pct:+.1f}%\n"
                    else:
                        med_val = res["median_abs"][-1]
                        if use_real_value:
                            med_val /= inflation_discount_factor
                        summary_prompt_data += f"- {name}: Mediana {med_val:,.0f} €, ({med_pct:+.1f}%), 10% alalaita {p10_pct:+.1f}%, 90% ylälaita {p90_pct:+.1f}%\n"

                # Sijoitetaan tulokset käyttäjän muokattavaan kehotteeseen
                if "{summary_prompt_data}" in ai_prompt_template:
                    final_prompt = ai_prompt_template.format(summary_prompt_data=summary_prompt_data)
                else:
                    final_prompt = f"{ai_prompt_template}\n\n{summary_prompt_data}"

                # Kutsutaan OpenAI API:a
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Olet asiantunteva, analyyttinen ja selkeä talousneuvoja."},
                        {"role": "user", "content": final_prompt}
                    ],
                    temperature=0.7
                )

                # Näytetään vastaus siistissä laatikossa
                st.info(response.choices[0].message.content)

            except Exception as e:
                st.error(f"Virhe tekoälyrajapinnassa: {e}")