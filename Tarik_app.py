import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import datetime

# ======================
# Données ESG manuelles
# ======================
actifs = pd.DataFrame({
    "Nom": [
        "Schneider Electric", "Danone", "L'Oréal", "Microsoft", "Tesla",
        "iShares ESG Aware MSCI USA ETF", "Vanguard ESG U.S. Stock ETF",
        "iShares USD Green Bond ETF", "Toyota", "TSMC"
    ],
    "ISIN": ["SU.PA", "BN.PA", "OR.PA", "MSFT", "TSLA", "ESGU", "ESGV", "BGRN", "TM", "TSM"],
    "Type": ["Action", "Action", "Action", "Action", "Action", "ETF", "ETF", "ETF", "Action", "Action"],
    "Secteur": [
        "Gestion de l'énergie", "Agroalimentaire", "Cosmétique", "Technologie", "Automobile",
        "Large-mid caps", "All caps", "Obligations vertes", "Automobile", "Technologie"
    ],
    "Région": [
        "Europe", "Europe", "Europe", "Amérique", "Amérique",
        "Amérique", "Amérique", "Amérique", "Asie", "Asie"
    ],
    "ESG RISK RATING - Morningstar": [10.4, 18.01, 19.58, 17.34, 24.76, 19.73, 19.48, 18.96, 27.95, 15.19],
    "Certifications": [
        "ISO 14001, ISO 50001, ISO 45001, ISO 26000",
        "ISO 14001, ISO 50001, ISO 45001, ISO 26000, B Corp, Entreprise à mission",
        "ISO 14001, ISO 50001, ISO 45001, ISO 26000, CDP",
        "ISO 14001",
        "ISO 14001, ISO 50001, ISO 45001, ISO 26000",
        "Aucune", "Aucune", "Aucune",
        "ISO 14001",
        "ISO 14001, ISO 50001, ISO 14064"
    ],
    "ODD": [
        "7, 9, 11, 12, 13, 17", "2, 3, 6, 12, 13, 17", "3, 5, 6, 12, 13, 17",
        "7, 9, 13, 17", "7, 9, 13, 17", "7, 9, 12, 13",
        "7, 9, 12, 13", "7, 9, 12, 13", "7, 9, 12, 13", "7, 9, 12, 13, 17"
    ],
    "Critères environnementaux": [
        "Empreinte carbone réduite, 100 % énergies renouvelables, rapports ESG fréquents",
        "Neutralité carbone d'ici 2050, gestion de l'eau et des déchets, rapports ESG fréquents",
        "Réduction des émissions de CO2, gestion de l'eau et des déchets, rapports ESG fréquents",
        "Neutralité carbone, gestion de l'eau et des déchets, lutte contre le gaspillage, rapports ESG fréquents",
        "Neutralité carbone, transition écologique, rapports ESG fréquents",
        "Sélection d'entreprises ESG, rapports ESG fréquents",
        "Faible exposition aux combustibles fossiles, rapports ESG fréquents",
        "Financement de projets écologiques, rapports ESG fréquents",
        "Réduction des émissions de CO2, préservation de l'eau et de la biodiversité, rapports ESG fréquents",
        "Énergie renouvelable, gestion de l'eau, réduction des émissions de CO2"
    ],
    "Critères sociaux": [
        "Diversité et inclusion, conditions de travail responsables, éducation et formation",
        "Conditions de travail équitables, accès à la nutrition, engagement communautaire",
        "Diversité et inclusion, responsabilité produit, bien-être des employés",
        "Diversité et inclusion, protection des données et confidentialité",
        "Diversité et inclusion, conditions de travail",
        "Diversité et inclusion",
        "Égalité des genres",
        "Financement de projets à impact social positif",
        "Diversité et inclusion, conditions de travail, engagement communautaire",
        "Diversité et inclusion, bien-être des employés, engagement communautaire"
    ],
})

# ======================
# Configuration de la page
# ======================
st.set_page_config(page_title="Portefeuille Durable", layout="wide")
st.title("🌱 Portefeuille de Finance Durable")
st.markdown("Ce portefeuille est construit à partir de critères extra-financiers (ESG, ODD, labels, ISO 14001, etc.).")

# ======================
# Filtres utilisateur
# ======================
st.sidebar.header("🔍 Filtres")

# Renommage pour simplifier le code
actifs = actifs.rename(columns={"ESG RISK RATING - Morningstar": "Score ESG"})

Types = st.sidebar.multiselect("Type d’actif", options=actifs["Type"].unique(), default=actifs["Type"].unique())
Secteur = st.sidebar.multiselect("Secteur", options=actifs["Secteur"].unique(), default=actifs["Secteur"].unique())
Region = st.sidebar.multiselect("Région", options=actifs["Région"].unique(), default=actifs["Région"].unique())

# Flatten certifications (en séparant chaque certification individuelle)
certifications_uniques = sorted({c.strip() for sublist in actifs["Certifications"].str.split(",") for c in sublist})
Certifications = st.sidebar.multiselect("Certifications", options=certifications_uniques, default=[])

# Extraire les ODD en liste
odds_uniques = sorted({o.strip() for sublist in actifs["ODD"].str.split(",") for o in sublist})
odds = st.sidebar.multiselect("ODD (Objectifs de Développement Durable)", options=odds_uniques, default=[])

score_min = st.sidebar.slider("Score ESG maximum des actifs", min_value=0.0, max_value=50.0, value=25.0, step=0.1)

# Filtrage initial
actifs_filtres = actifs[(
    actifs["Type"].isin(Types)) &
    (actifs["Secteur"].isin(Secteur)) &
    (actifs["Région"].isin(Region)) &
    (actifs["Score ESG"] <= score_min)  # ESG RISK plus bas = meilleur score
]

# Filtrage ODD
if odds:
    actifs_filtres = actifs_filtres[actifs_filtres["ODD"].apply(lambda x: any(o in x.split(",") for o in odds))]

# Filtrage certifications
if Certifications:
    actifs_filtres = actifs_filtres[actifs_filtres["Certifications"].apply(lambda x: any(c.strip() in x for c in Certifications))]

# Calcul des poids inversés des actifs filtrés
if not actifs_filtres.empty:
    # Calcul des poids inversés
    actifs_filtres["Poids Inversés"] = 1 / actifs_filtres["Score ESG"]
    
    # Normalisation des poids (la somme des poids doit être égale à 1)
    total_poids = actifs_filtres["Poids Inversés"].sum()
    actifs_filtres["Poids"] = actifs_filtres["Poids Inversés"] / total_poids

# ======================
# Tableau des actifs sélectionnés
# ======================
st.header("📋 Actifs du portefeuille filtré")
st.dataframe(actifs_filtres.reset_index(drop=True), use_container_width=True)

# ======================
# Visualisation ESG
# ======================
st.header("📊 Visualisation ESG")

def draw_pie_chart(data, title, cmap):
    fig, ax = plt.subplots(figsize=(3, 3))
    couleurs = cmap(np.linspace(0, 1, len(data)))
    wedges, texts, autotexts = ax.pie(
        data,
        labels=None,  # Supprimer les labels du camembert
        autopct="%1.1f%%",
        startangle=90,
        colors=couleurs,
        textprops={'fontsize': 7}
    )
    ax.axis("equal")
    ax.set_title(title, fontsize=10)

    # Ajouter légende à côté pour ne pas tasser le camembert
    ax.legend(
        data.index,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=6,
        frameon=False
    )
    return fig

col1, col2, col3 = st.columns(3)

with col1:
    secteur_data = actifs_filtres.groupby("Secteur")["Poids"].sum().sort_values(ascending=False)
    fig1 = draw_pie_chart(secteur_data, "Secteurs", plt.cm.Paired)
    st.pyplot(fig1)

with col2:
    region_data = actifs_filtres.groupby("Région")["Poids"].sum().sort_values(ascending=False)
    fig2 = draw_pie_chart(region_data, "Régions", plt.cm.Set3)
    st.pyplot(fig2)

with col3:
    type_data = actifs_filtres.groupby("Type")["Poids"].sum().sort_values(ascending=False)
    fig3 = draw_pie_chart(type_data, "Types d’actifs", plt.cm.Accent)
    st.pyplot(fig3)

# ======================
# Performances financières
# ======================
st.header("📈 Performances du portefeuille")

end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=3*365)

tickers = actifs_filtres["ISIN"].tolist()

if not tickers:
    st.warning("Aucun actif sélectionné. Veuillez ajuster vos filtres.")
else:
    data = yf.download(tickers, start=start_date, end=end_date)["Close"].bfill()

    if data.empty:
        st.error("Les données financières n'ont pas pu être récupérées.")
    else:
        data.dropna(axis=1, how='any', inplace=True)

        if data.empty:
            st.error("Pas assez de données valides après nettoyage.")
        else:
            performance_par_actif = (data.iloc[-1] / data.iloc[0]) - 1
            performance_portefeuille = performance_par_actif.mean()
            st.metric(label="Performance sur 3 ans", value=f"{performance_portefeuille:.2%}")

            data_normalized = data 

            # Utilisation des poids des actifs filtrés
            poids_dict = actifs_filtres.set_index("ISIN")["Poids"].to_dict()
            data_weighted = data_normalized.multiply([poids_dict.get(t, 0) for t in data.columns], axis=1)

            valeur_actuelle = data_weighted.sum(axis=1).iloc[-1]
            st.metric(label="Valeure actuelle du portefeuille", value=f"{round(valeur_actuelle,2)}")

            # Évolution du portefeuille dans le temps
            st.write("### 📉 Évolution de la valeur du portefeuille depuis 3 ans")

            valeur_portefeuille = data_weighted.sum(axis=1)

            st.line_chart(valeur_portefeuille)

            # Détail par actif
            st.write("### 🔍 Détail par actif depuis 3 ans")
            st.dataframe(performance_par_actif.sort_values(ascending=False).map(lambda x: f"{x:.2%}"))