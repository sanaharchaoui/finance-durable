import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import datetime
import plotly.express as px

# ======================
# Données ESG manuelles
# ======================
actifs = pd.DataFrame({
    "Nom": [
        "Schneider Electric", "Danone", "L'Oréal", "Microsoft", "Tesla",
        "iShares ESG Aware MSCI USA ETF", "Vanguard ESG U.S. Stock ETF",
        "iShares USD Green Bond ETF", "Toyota", "TSMC",
        "Sony Group Corporation", "Tata Consultancy Services Limited"
    ],
    "ISIN": [
        "SU.PA", "BN.PA", "OR.PA", "MSFT", "TSLA", "ESGU", "ESGV", "BGRN", "TM", "TSM",
        "6758.T", "TCS.NS"
    ],
    "Type": [
        "Action", "Action", "Action", "Action", "Action", "ETF", "ETF", "ETF", "Action", "Action",
        "Action", "Action"
    ],
    "Secteur": [
        "Gestion de l'énergie", "Agroalimentaire", "Cosmétique", "Technologie", "Automobile",
        "Large-mid caps", "All caps", "Obligations vertes", "Automobile", "Technologie",
        "Technologie", "Technologie"
    ],
    "Région": [
        "Europe", "Europe", "Europe", "Amérique", "Amérique",
        "Amérique", "Amérique", "Amérique", "Asie", "Asie",
        "Asie", "Asie"
    ],
    "ESG RISK RATING - Morningstar": [
        10.4, 18.01, 19.58, 17.34, 24.76, 19.73, 19.48, 18.96, 27.95, 15.19,
        17.0, 11.4
    ],
    "Certifications": [
        "ISO 14001, ISO 50001, ISO 45001, ISO 26000",
        "ISO 14001, ISO 50001, ISO 45001, ISO 26000, B Corp, Entreprise à mission",
        "ISO 14001, ISO 50001, ISO 45001, ISO 26000, CDP",
        "ISO 14001",
        "ISO 14001, ISO 50001, ISO 45001, ISO 26000",
        "Aucune", "Aucune", "Aucune",
        "ISO 14001",
        "ISO 14001, ISO 50001, ISO 14064",
        "ISO 14001, ISO 50001",
        "ISO 14001, ISO 26000"
    ],
    "ODD": [
        "7, 9, 11, 12, 13, 17", "2, 3, 6, 12, 13, 17", "3, 5, 6, 12, 13, 17",
        "7, 9, 13, 17", "7, 9, 13, 17", "7, 9, 12, 13",
        "7, 9, 12, 13", "7, 9, 12, 13", "7, 9, 12, 13", "7, 9, 12, 13, 17",
        "9, 12, 13", "8, 9, 13"
    ],
    "MSCI IMPLIED TEMPERATURE RISE": [
        1.7, 2.4, 1.3, 1.4, 1.5, 2.7, "N.A", 2.7, 2.0, 2.5,
        1.6, 1.7
    ],
    "Critères environnementaux": [
        ["réduction des émissions de CO₂", "énergies renouvelables", "rapports ESG fréquents"],
        ["neutralité carbone", "gestion de l’eau", "gestion des déchets", "rapports ESG fréquents"],
        ["réduction des émissions de CO₂", "gestion de l’eau", "gestion des déchets", "rapports ESG fréquents"],
        ["neutralité carbone", "gestion de l’eau", "gestion des déchets", "réduction du gaspillage", "rapports ESG fréquents"],
        ["neutralité carbone", "transition écologique", "rapports ESG fréquents"],
        ["entreprises ESG sélectionnées", "rapports ESG fréquents"],
        ["faible exposition aux combustibles fossiles", "rapports ESG fréquents"],
        ["financement de projets écologiques", "rapports ESG fréquents"],
        ["réduction des émissions de CO₂", "gestion de l’eau", "biodiversité", "rapports ESG fréquents"],
        ["énergies renouvelables", "gestion de l’eau", "réduction des émissions de CO₂"],
        ["réduction des émissions de CO₂", "efficacité énergétique", "gestion des déchets électroniques"],
        ["neutralité carbone", "développement durable", "gestion de l’énergie"]
    ],

    "Critères sociaux": [
        ["diversité et inclusion", "conditions de travail", "éducation et formation"],
        ["conditions de travail", "accès à la nutrition", "engagement communautaire"],
        ["diversité et inclusion", "responsabilité produit", "bien-être des employés"],
        ["diversité et inclusion", "protection des données"],
        ["diversité et inclusion", "conditions de travail"],
        ["diversité et inclusion"],
        ["égalité des genres"],
        ["financement de projets à impact social positif"],
        ["diversité et inclusion", "conditions de travail", "engagement communautaire"],
        ["diversité et inclusion", "bien-être des employés", "engagement communautaire"],
        ["diversité et inclusion", "conditions de travail", "engagement communautaire"],
        ["éducation et formation", "inclusion numérique", "responsabilité sociétale"]
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

# Flatten certifications
certifications_uniques = sorted({c.strip() for sublist in actifs["Certifications"].str.split(",") for c in sublist})
Certifications = st.sidebar.multiselect("Certifications", options=certifications_uniques, default=[])

# ODD
odds_uniques = sorted({o.strip() for sublist in actifs["ODD"].str.split(",") for o in sublist})
odds = st.sidebar.multiselect("ODD (Objectifs de Développement Durable)", options=odds_uniques, default=[])

# Critères environnementaux et sociaux (placés ici AVANT le score ESG)
env_criteres_uniques = sorted({crit for sous_liste in actifs["Critères environnementaux"] for crit in sous_liste})
soc_criteres_uniques = sorted({crit for sous_liste in actifs["Critères sociaux"] for crit in sous_liste})

EnvCriteres = st.sidebar.multiselect("Critères environnementaux", options=env_criteres_uniques, default=[])
SocCriteres = st.sidebar.multiselect("Critères sociaux", options=soc_criteres_uniques, default=[])

# Score ESG
score_min = st.sidebar.slider("Score ESG maximum des actifs", min_value=0.0, max_value=50.0, value=25.0, step=0.1)

# ======================
# Filtrage des actifs
# ======================
actifs_filtres = actifs[(
    actifs["Type"].isin(Types)) &
    (actifs["Secteur"].isin(Secteur)) &
    (actifs["Région"].isin(Region)) &
    (actifs["Score ESG"] <= score_min)
]

# Filtrage ODD
if odds:
    actifs_filtres = actifs_filtres[actifs_filtres["ODD"].apply(lambda x: any(o in x.split(",") for o in odds))]

# Filtrage certifications
if Certifications:
    actifs_filtres = actifs_filtres[actifs_filtres["Certifications"].apply(lambda x: any(c.strip() in x for c in Certifications))]

# Filtrage environnemental
if EnvCriteres:
    actifs_filtres = actifs_filtres[
        actifs_filtres["Critères environnementaux"].apply(lambda liste: any(crit in liste for crit in EnvCriteres))
    ]

# Filtrage social
if SocCriteres:
    actifs_filtres = actifs_filtres[
        actifs_filtres["Critères sociaux"].apply(lambda liste: any(crit in liste for crit in SocCriteres))
    ]

# Calcul des poids inversés
if not actifs_filtres.empty:
    actifs_filtres["Poids Inversés"] = 1 / actifs_filtres["Score ESG"]
    total_poids = actifs_filtres["Poids Inversés"].sum()
    actifs_filtres["Poids"] = actifs_filtres["Poids Inversés"] / total_poids


# ======================
# Tableau des actifs sélectionnés
# ======================
st.header("Actifs du portefeuille filtré")
st.dataframe(actifs_filtres.reset_index(drop=True), use_container_width=True)

# ======================
# Notre choix : création d’un portefeuille personnalisé
# ======================
st.header("Notre approche : un portefeuille construit sur mesure")
st.markdown("""
Dans le cadre de ce projet, nous avons fait le choix de **ne pas sélectionner un fonds existant**, mais de construire **notre propre portefeuille durable** à partir d’une sélection d’actifs cotés et d’ETF labellisés.

Notre démarche repose sur une volonté de :
- **contrôler les critères ESG choisis** et les pondérations appliquées,
- **analyser activement l’alignement avec les ODD** (Objectifs de Développement Durable),
- et surtout de **tester une méthodologie personnelle** inspirée des meilleures pratiques en finance durable.

Nous avons donc défini nos propres critères de filtrage : type d’actif, score ESG, certifications ISO, ODD, etc., et nous avons pondéré les actifs sélectionnés en fonction de leur performance extra-financière.
""")

# ======================
# 📘 1. Explication de la stratégie ESG
# ======================
st.header("Stratégie ESG du portefeuille")
st.markdown("""
Ce portefeuille est construit selon une approche **best-in-class**, qui consiste à sélectionner les entreprises ayant les meilleurs scores ESG dans leur secteur.

La stratégie repose sur :
- l'exclusion d'actifs au-delà d’un seuil ESG donné (score > 25 éliminés)
- une pondération **inversement proportionnelle au risque ESG** (plus l’entreprise est vertueuse, plus elle est pondérée)
- une sélection diversifiée de secteurs, zones géographiques et types d’actifs

Les données ESG sont issues de la notation **Morningstar**, combinée à des certifications et alignements avec les **Objectifs de Développement Durable (ODD)**.
""")

# ======================
# 🌿 2. Analyse de l'impact ESG
# ======================
st.header("1. Analyse d'impact ESG")
if not actifs_filtres.empty:
    score_moyen = round(actifs_filtres["Score ESG"].mean(), 2)
    st.markdown(f"**Score ESG moyen du portefeuille :** `{score_moyen}`")

    st.markdown("""
    🔍 **Interprétation :** Un score ESG plus faible signifie un meilleur comportement extra-financier.
    - < 15 : très faible risque ESG 
    - 15 à 20 : risque modéré 
    - \\> 20 : à surveiller 

    De plus, les entreprises sont analysées en fonction de leurs contributions aux **ODD** :
    """)
    odd_series = actifs_filtres["ODD"].str.split(", ").explode().value_counts()
    st.bar_chart(odd_series)
else:
    st.info("Aucun actif sélectionné pour analyser l'impact ESG.")


# ======================
# 🏷️ 3. Labels et certifications
# ======================
st.header("2. Labels et certifications durables")
st.markdown("""
Certains actifs du portefeuille disposent de **labels ou certifications reconnus**, renforçant leur crédibilité :

- **ISO 14001, 50001, 26000** : gestion environnementale, performance énergétique, responsabilité sociétale.
- **B Corp**, **Entreprise à mission**, **CDP** : critères extra-financiers vérifiés par des entités indépendantes.
- **Label ISR (France)** : notre portefeuille suit des principes similaires :
    - exclusion des actifs avec un score ESG trop élevé,
    - intégration ESG systématique dans la sélection,
    - diversification sectorielle et géographique.
- **Label Greenfin (France)** : via l’exposition aux **obligations vertes (Green Bonds)**, le portefeuille contribue au **financement de projets écologiques**.

Ces éléments montrent une **volonté de conformité aux meilleures pratiques** de la finance durable française et européenne.
""")

# ======================
# 📊 4. Suivi combiné des performances financières et ESG
# ======================
st.header("3. Tableau de bord ESG & financier")
if not actifs_filtres.empty:
    comparaison = actifs_filtres[["Nom", "Score ESG", "Poids"]].copy()
    comparaison = comparaison.sort_values(by="Poids", ascending=False)
    st.dataframe(comparaison.style.format({"Score ESG": "{:.2f}", "Poids": "{:.2%}"}), use_container_width=True)
    st.markdown("""
    Ce tableau permet de croiser la performance ESG des entreprises avec leur importance dans le portefeuille.
    Il constitue un outil de **pilotage intégré** : durable et financier.
    """)
else:
    st.info("Aucun actif sélectionné pour afficher le tableau ESG/Finance.")

# ======================
# 📈 5. Visualisation croisée : Poids vs Score ESG
# ======================
st.header("4. Visualisation croisée : Poids vs Score ESG")

if not actifs_filtres.empty:
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(
        actifs_filtres["Score ESG"],
        actifs_filtres["Poids"],
        c=actifs_filtres["Poids"],
        cmap="viridis",
        s=100,
        edgecolors='black'
    )
    for i, row in actifs_filtres.iterrows():
        ax.annotate(row["Nom"], (row["Score ESG"], row["Poids"]), fontsize=8, alpha=0.7)

    ax.set_xlabel("Score ESG (plus faible = meilleur)")
    ax.set_ylabel("Poids dans le portefeuille")
    ax.set_title("Corrélation entre la qualité ESG et la pondération")
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("""
    Ce graphique permet de **visualiser l'approche Best-in-Class** :  
    Les entreprises avec un **meilleur score ESG (plus bas)** sont **plus fortement pondérées**,  
    ce qui maximise l’impact positif du portefeuille.
    """)
else:
    st.info("Aucun actif sélectionné pour afficher la corrélation Score ESG / Poids.")


# ======================
# Fin du rapport enrichi
# ======================


# ======================
# Visualisation ESG
# ======================
st.header("5. Visualisation ESG")

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

st.markdown("""
---

### **Analyse détaillée de la composition ESG**

#### **Répartition sectorielle**
Le portefeuille est largement exposé au **secteur technologique** (≈ 40 %), ce qui reflète le rôle clé des technologies dans la transition verte (efficacité énergétique, digitalisation responsable, neutralité carbone).  
La présence d’autres secteurs tels que l’**automobile**, l’**agroalimentaire** et les **énergies** (via la gestion de l’énergie et obligations vertes) assure une **diversification raisonnable**.

#### **Répartition géographique**
Le portefeuille est bien réparti entre **Amérique (38 %)**, **Asie (32 %)** et **Europe (30 %)**, ce qui reflète une **diversification régionale équilibrée**.  
L'ajout récent d’actifs asiatiques à faible température implicite (Sony, Infosys) a permis de **renforcer la couverture Asie** tout en **réduisant l’empreinte climatique globale** du portefeuille.

#### **Répartition par type d’actif**
La majorité des investissements sont concentrés sur des **actions individuelles (≈ 77 %)**, permettant un **contrôle fin des critères ESG** par actif.  
Les **ETF (≈ 23 %)** complètent la structure avec une approche plus large et diversifiée, tout en intégrant des filtres ESG dans leur construction.

---

Ces répartitions montrent que le portefeuille respecte les **principes de diversification** tout en **optimisant l’impact ESG** selon une approche best-in-class.
""")


# ======================
# 🔥 Température implicite (réelle MSCI si dispo, sinon estimation)
# ======================
st.subheader("6. Température implicite du portefeuille (source MSCI ou estimation)")

def complete_temperature(row):
    try:
        temp = float(row["MSCI IMPLIED TEMPERATURE RISE"])
        return temp
    except:
        score = row["Score ESG"]
        if score < 15:
            return 1.5
        elif score < 20:
            return 2.0
        elif score < 25:
            return 2.5
        else:
            return 3.0

if not actifs_filtres.empty:
    actifs_filtres["Température estimée"] = actifs_filtres.apply(complete_temperature, axis=1)
    temp_implicite = np.average(actifs_filtres["Température estimée"], weights=actifs_filtres["Poids"])
    st.metric(label="🌍 Température implicite du portefeuille", value=f"{temp_implicite:.2f}°C")
    st.markdown(f"""
    Cette température est calculée à partir de :
    - La **donnée MSCI** lorsqu'elle est disponible (colonne *MSCI IMPLIED TEMPERATURE RISE*),
    - Sinon, une **estimation pédagogique basée sur le score ESG** est utilisée.

    👉 **Température moyenne pondérée du portefeuille :** `{temp_implicite:.2f}°C`

    - Objectif de l'Accord de Paris : < **2°C**
    - Température < 2°C = portefeuille aligné
    """)
else:
    st.info("Aucun actif sélectionné pour estimer la température implicite.")


# ======================
# Performances financières
# ======================
st.header("7. Performances du portefeuille")

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
            st.write("### Évolution de la valeur du portefeuille depuis 3 ans")

            valeur_portefeuille = data_weighted.sum(axis=1)

            st.line_chart(valeur_portefeuille)

            # Détail par actif
            st.write("### Détail par actif depuis 3 ans")
            st.dataframe(performance_par_actif.sort_values(ascending=False).map(lambda x: f"{x:.2%}"))


# ======================
# 🌡️ Climate VaR : Scénario de stress climatique
# ======================
st.header("8. Climate VaR : Scénario de stress climatique")
st.markdown("""
La **Climate VaR (Value at Risk climatique)** est une estimation des pertes potentielles que subirait un portefeuille en cas de **choc climatique majeur**.

Nous avons simulé un **scénario de stress à court terme** :
- **-10%** de perte sur les actifs exposés aux risques physiques et de transition (technologie, automobile, énergie),
- **-5%** sur les autres secteurs.

Ce scénario est **inspiré de la logique de risque de transition** décrite dans les cadres comme le **SFDR**, ou les modèles de **stress test climatique** évoqués dans les rapports MSCI ou de la BCE.
""")

# Définir les secteurs sensibles au stress climatique
secteurs_sensibles = ["Technologie", "Automobile", "Énergie"]
stress_details = []

for i, row in actifs_filtres.iterrows():
    secteur = row["Secteur"]
    poids = row["Poids"]
    if secteur in secteurs_sensibles:
        perte_pct = -0.10
    else:
        perte_pct = -0.05
    perte_pondérée = perte_pct * poids
    stress_details.append({
        "Nom": row["Nom"],
        "Secteur": secteur,
        "Poids": poids,
        "Perte % secteur": f"{perte_pct:.0%}",
        "Impact portefeuille": perte_pondérée
    })

df_stress = pd.DataFrame(stress_details)
perte_totale = df_stress["Impact portefeuille"].sum()

import plotly.express as px
st.metric(label="Perte totale estimée en cas de choc climatique", value=f"{perte_totale:.2%}")

st.markdown("""
Ce tableau montre l’impact simulé de ce stress sur chaque actif du portefeuille :
""")
st.dataframe(df_stress[["Nom", "Secteur", "Poids", "Perte % secteur", "Impact portefeuille"]].sort_values(by="Impact portefeuille", ascending=True), use_container_width=True)

fig = px.bar(df_stress.sort_values(by="Impact portefeuille"),
             x="Nom", y="Impact portefeuille", color="Secteur",
             title="Impact du stress climatique par actif",
             labels={"Impact portefeuille": "Perte simulée"}, height=400)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Interprétation** :
- Cette estimation simplifiée montre comment les **secteurs sensibles aux régulations climatiques** pourraient amplifier les pertes.
- Elle donne un aperçu utile de la **vulnérabilité climatique du portefeuille**, bien qu’elle ne remplace pas un modèle climatique complet (type Climate Value-at-Risk MSCI ou scénarios Net-Zero).

Ce type de simulation peut être adapté à des **scénarios physiques (ouragan, sécheresse)** ou **de politique climatique (taxe carbone, réglementation stricte)**.
""")


