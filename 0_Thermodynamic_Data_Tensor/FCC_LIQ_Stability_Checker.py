import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="CoCrFeNi Gibbs Free Energy Calculator",
    layout="centered"
)

st.title("CoCrFeNi Gibbs Free Energy Calculator")

csv_folder = "csv_files"

st.markdown(
    "Calculate **G_LIQ**, **G_FCC**, **ΔG = G_FCC − G_LIQ**, "
    "and determine the stable phase at the selected composition and temperature."
)


@st.cache_data
def load_temperature_file(T):
    file_path = os.path.join(csv_folder, f"Gibbs_{T}K.csv")

    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        st.stop()

    df = pd.read_csv(file_path)

    required_cols = ["Co", "Cr", "Fe", "Ni", "G_LIQ", "G_FCC"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"Missing columns in {file_path}: {missing}")
        st.stop()

    return df


with st.sidebar:
    st.header("Input Conditions")

    T = st.number_input(
        "Temperature [K]",
        min_value=300,
        max_value=3300,
        value=1500,
        step=100
    )

    Co = st.number_input(
        "Co composition",
        min_value=0.01,
        max_value=0.40,
        value=0.25,
        step=0.01,
        format="%.2f"
    )

    Cr = st.number_input(
        "Cr composition",
        min_value=0.01,
        max_value=0.40,
        value=0.25,
        step=0.01,
        format="%.2f"
    )

    Fe = st.number_input(
        "Fe composition",
        min_value=0.01,
        max_value=0.40,
        value=0.25,
        step=0.01,
        format="%.2f"
    )

    Ni = round(1.0 - Co - Cr - Fe, 2)

    st.number_input(
        "Ni composition = 1 − Co − Cr − Fe",
        value=Ni,
        disabled=True,
        format="%.2f"
    )

st.subheader("Selected Input")

input_df = pd.DataFrame({
    "T [K]": [T],
    "Co": [Co],
    "Cr": [Cr],
    "Fe": [Fe],
    "Ni = 1 - Co - Cr - Fe": [Ni]
})

st.dataframe(input_df, width='content', hide_index=True)

if Ni < 0.01 or Ni > 0.40:
    st.error(
        f"Invalid Ni = {Ni:.2f}. Adjust Co, Cr, and Fe so that Ni stays between 0.01 and 0.40."
    )
    st.stop()

if abs((Co + Cr + Fe + Ni) - 1.0) > 1e-8:
    st.error("Composition sum is not equal to 1. Please check the inputs.")
    st.stop()

df = load_temperature_file(T)

df_round = df.copy()
for col in ["Co", "Cr", "Fe", "Ni"]:
    df_round[col] = df_round[col].round(2)

matched = df_round[
    (df_round["Co"] == round(Co, 2)) &
    (df_round["Cr"] == round(Cr, 2)) &
    (df_round["Fe"] == round(Fe, 2)) &
    (df_round["Ni"] == round(Ni, 2))
]

if matched.empty:
    st.error(
        "No exact matching row found in the CSV file for this selected composition.\n\n"
        f"Required row: Co={Co:.2f}, Cr={Cr:.2f}, Fe={Fe:.2f}, Ni={Ni:.2f}, T={T} K"
    )
    st.stop()

row = matched.iloc[0]

G_LIQ = float(row["G_LIQ"])
G_FCC = float(row["G_FCC"])
DeltaG = G_FCC - G_LIQ

if G_LIQ < G_FCC:
    stable_phase = "LIQUID"
elif G_FCC < G_LIQ:
    stable_phase = "FCC"
else:
    stable_phase = "LIQUID and FCC have equal Gibbs free energy"

st.subheader("Calculated Result")

col1, col2 = st.columns(2)

with col1:
    st.metric("G_LIQ", f"{G_LIQ:,.2f} J/mol")
    st.metric("G_FCC", f"{G_FCC:,.2f} J/mol")

with col2:
    st.metric("ΔG = G_FCC − G_LIQ", f"{DeltaG:,.2f} J/mol")
    st.metric("Stable Phase", stable_phase)

result_df = pd.DataFrame({
    "T [K]": [T],
    "Co": [Co],
    "Cr": [Cr],
    "Fe": [Fe],
    "Ni": [Ni],
    "G_LIQ [J/mol]": [G_LIQ],
    "G_FCC [J/mol]": [G_FCC],
    "DeltaG = G_FCC - G_LIQ [J/mol]": [DeltaG],
    "Stable Phase": [stable_phase]
})

st.subheader("Result Table")
st.dataframe(result_df, width='content', hide_index=True)

csv = result_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Result as CSV",
    data=csv,
    file_name=f"Gibbs_Result_T{T}_Co{Co:.2f}_Cr{Cr:.2f}_Fe{Fe:.2f}_Ni{Ni:.2f}.csv",
    mime="text/csv"
)
