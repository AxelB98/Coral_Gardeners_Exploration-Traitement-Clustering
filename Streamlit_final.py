import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score)
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# MENU
st.sidebar.title("Sommaire")
pages = ["Introduction - Contexte","Exploration", "Data Vizualization", "Preprocessing", "Clustering", "Interprétation", "Recommandations et limites"]
page = st.sidebar.radio("Aller vers", pages)


# PAGE INTRODUCTION
if page=="Introduction - Contexte":
    st.title("Coral Gardeners")
    st.header("Projet de Machine Learning pour identifier des segments de clients propices au lancement d'un modèle d'abonnement mensuel.")

    with st.expander("Qu'est ce que Coral Gardeners ?"):
        st.image("https://cdn.shopify.com/s/files/1/0505/4663/9015/files/GIF.gif?v=1664990549", caption="Coral Gardeners")
        st.write("Coral Gardeners est une ONG fondée en 2017 en Polynésie française, dont la mission est de restaurer les récifs coralliens menacés par le changement climatique. Pour cela, l’organisation collecte des 'super coraux' ayant survécu aux hausses de températures, les fait grandir dans ses nurseries sous-marines, et les replante sur les récifs dégradés.")
        st.write(" L'organisation s'appuie sur trois leviers :")
        (" - La **sensibilisation** du plus grand nombre via une communication massive sur les réseaux sociaux (200 millions de personnes touchées).")
        (" - L'**innovation technologique** (ReefOS, un outil de suivi géospatial des récifs).")
        (" - La **restauration active des récifs** (déjà 30 000 coraux replantés). ")
        (" Leur programme d'adoption de coraux constitue leur principal modèle de **financement participatif** : moyennant un don, n'importe qui peut adopter un corail en ligne, lui donner un prénom, et suivre sa croissance en temps réel grâce à un QR code, jusqu'à sa replantation sur le récif. " \
        " L'enjeu business autour de ce programme est central : pour **financer ses opérations de terrain** (nurseries, plongeurs, laboratoires), Coral Gardeners doit constamment **recruter de nouveaux adoptants** et **fidéliser sa communauté**. ")
        st.image("CG_LIVESTREAM.jpg", caption="Coral Gardeners - Nurserie en live : Culture de coraux en mer")


    with st.expander("Problématique et objectifs du projet"):
        st.write("Problématique : Coral Gardeners mène une réflexion stratégique actuellement : doivent-ils passer ou non en modèle d’abonnement ? Et si oui, doivent-ils faire de l’abonnement le modèle unique pour l’adoption des coraux, ou doivent-ils le combiner au modèle d’adoption actuel, selon les géographies ou au choix du client ?")
        ("Objectif de ce projet :")
        ("Peut-on identifier, à partir des **données transactionnelles** (Shopify) et **CRM** (Klaviyo), les segments de clients les plus propices au lancement d’un modèle d’abonnement mensuel ?")
        st.image("Chris-Matt_noe.langronier-2-2.jpg", caption="Coral Gardeners - Un exemple de clients plutôt ... connus.")

    with st.expander("Etapes du projet"):
        st.write("A partir de ces données, nous avons :")
        ("- Dans un premier temps, analysé le comportement d’achat et d’engagement des clients de Coral Gardeners. (cf Data Vizualization)")
        ("- Ensuite, nous avons construit un modèle avec pour objectif de segmenter les clients en différentes catégories, selon leur comportement d’achat (achat, réachat, intervalle de temps entre 2 commandes, pays d’origine, etc…) et d’engagement (ouverture et clics des emails reçus).")
        ("- A partir de ces catégories, nous permettant d’identifier les clients (ou les pays) les plus compatibles avec un modèle de souscription mensuelle, nous avons réalisé une recommandation auprès de Coral Gardeners.") 

    with st.expander("Utilité pour Coral Gardeners"):
        st.write("Notre projet répond ainsi à 2 questions :")
        ("- **À quels clients envoyer des questionnaires pour obtenir des informations supplémentaires ?**")
        ("- **Comment rédiger l'email et quelles questions poser ?**")
        ("Avec ces données, Coral Gardeners peut ainsi :")
        ("- **Créer et réaliser les questionnaires multiples et ainsi récupérer des données complémentaires.**")
        ("- **Prendre une décision quant au lancement ou non d’un nouveau modèle de souscription, unique ou combiné avec le modèle d’adoption actuel.**")
        st.image("DataScienceProject_CG.JPG", caption="Coral Gardeners - Un enjeux de clustering")



# PAGE DATA EXPLORATION
if page=="Exploration":
    st.title("Coral Gardeners")
    st.header("Exploration des données")
    
    #Sources
    st.subheader("Données à disposition initialement")
    with st.expander("Voir les sources de données"):
        st.markdown("""
            Nous disposons de plusieurs sources de données :

            - **table_customers** : informations clients (Shopify - 33,8k lignes)  
            - **table_orders** : informations commandes (Shopify - 37,8k lignes)  
            - **table_orders_details** : détail des produits de chaque commande (Shopify - 64,9k lignes)
            - **table_sessions** : données de navigation (Shopify - 1,8M lignes)
            - **country_mapping** : clé d'harmonisation des pays  
            - **product_mapping** : clé d'harmonisation des produits  
            - **meta_account_monthly / meta_geo_monthly** : campagnes marketing META  
            - **klaviyo_2023_2025_final** : données CRM emailing (Klaviyo - 126k lignes)
            """)

    #Problématiques
    st.subheader("Problématiques identifiées")
    with st.expander("Voir les problèmes rencontrés"):
        st.markdown("""
            Lors de l’exploration des données Shopify, plusieurs incohérences ont été identifiées :
            - lignes avec `order = 0` mais quantités non nulles  
            - lignes avec `gross sales` incohérents  
            - présence de lignes liées aux **tips** et **shipping**  

            Cela provient du fait que :
            - 1 ligne = 1 produit  
            - la création de lignes supplémentaires pour les tips et le shipping, non cohérentes

            -> Le dataset était difficilement exploitable en l’état.
            """)

    #Premiers traitements
    st.subheader("Premiers traitements réalisés")
    with st.expander("1/ Construction d’un dataset Shopify consolidé au niveau produit"):
        st.markdown("""
            Création du fichier **Commandes_Shopify.csv** :
            - fusion des tables customers, orders et order_details  
            - nettoyage des noms via country_mapping et product_mapping  
            """)
        
    with st.expander("2/ Nouvel extract Shopify au niveau commande"):
        st.markdown("""
            Création d’un nouvel extract **shopify_orders_raw.csv** :
            - niveau commande (1 ligne = 1 commande)  
            - période : 01/01/2023 → 31/12/2025  
            - suppression des lignes inexploitables (tips, shipping)
            -> perte du détail produit, mais non bloquant pour l’analyse, car les ventes sont très majoritairement de l'adoption de coraux
            """)
        
    with st.expander("3/ Nettoyage et préparation du dataset Shopify au niveau commande"):
        st.markdown("""
            Création du dataset final **shopify_order_level_clean.csv** :
            - suppression des commandes remboursées
            - suppression des commandes avec `order = 0` (gift cards, geste commercial, anomalies)
            - suppression des colonnes inutiles  
            - gestion des valeurs manquantes (supression)
            - ajout d’une colonne Mois_Année  
            - conversion des types (dates, montants)  
            - harmonisation des pays  
            - nettoyage (supression des commandes employés et B2B) et harmonisation des emails (minuscule + trim)  
            """)

    with st.expander("4/ Données non retenues"):
        st.markdown("""
            Les données META n’ont pas été utilisées :
            - difficilement exploitables dans le cadre de cette analyse  
            - absence de lien direct avec les données CRM
            """)

    # Dataset finaux dataviz
    st.subheader("Datasets finaux à disposition")
    st.markdown("Voici les datasets utilisés pour la data vizualisation:")

    df_product = pd.read_csv("../Datas/Shopify_product_level.csv")
    df_order = pd.read_csv("../Datas/shopify_order_level_clean.csv")
    df_sessions = pd.read_csv("../Datas/table_sessions_20230101_ 20251231.csv")
    df_klaviyo = pd.read_csv("../Datas/klaviyo_2023_2025_final.csv")

    st.markdown("""
        **Dataset Shopify — niveau produit**  
        Chaque ligne correspond à un produit dans une commande.
        """) 
    if st.checkbox("Afficher un aperçu", key="produit"):
        st.dataframe(df_product.head(10))

    st.markdown("""
        **Dataset Shopify — niveau commande**  
        Chaque ligne correspond à une commande.
        """) 
    if st.checkbox("Afficher un aperçu", key="order"):
        st.dataframe(df_order.head(10))

    st.markdown("""
        **Dataset Shopify — sessions**  
        Données de navigation des utilisateurs sur le site.
        """) 
    if st.checkbox("Afficher un aperçu", key="sessions"):
        st.dataframe(df_sessions.head(10))

    st.markdown("""
        **Dataset Klaviyo (CRM)**  
        Données CRM et engagement email des clients.
        """) 
    if st.checkbox("Afficher un aperçu", key="klaviyo"):
       st.dataframe(df_klaviyo.head(10))


# PAGE DATA VISUALIZATION
if page == "Data Vizualization":
    st.title("Coral Gardeners")
    st.header("Exploration graphique des données")
    st.subheader("Performances globales (données Shopify)")
    
    with st.expander("Evolution mensuelle des sessions, du nombre de commandes et du chiffre d’affaires (CA) généré"):
        df_shopify = pd.read_csv('../Datas/shopify_order_level_clean.csv')
        df_sessions = pd.read_csv('../Datas/table_sessions_20230101_ 20251231.csv')

        #Création d'une colonne mois avec le mois et l'année (type str pour les graphiques):
        df_sessions["Day"] = pd.to_datetime(df_sessions["Day"])
        df_shopify["month_year"] = pd.to_datetime(df_shopify["month_year"], format='ISO8601')

        df_sessions['month_year'] = df_sessions['Day'].dt.to_period('M').astype(str)
        df_shopify['month_year'] = df_shopify['month_year'].dt.to_period('M').astype(str)

        #Obtenir le CA, le nombre de sessions et le nombre de commandes mensuels : 
        CA_mois = df_shopify.groupby(['month_year'])['net_sales'].sum().reset_index()
        sessions_mois = df_sessions.groupby(['month_year'])['Sessions'].sum().reset_index()
        commandes_mois = df_shopify.groupby(['month_year'])['Orders'].sum().reset_index()

        #Créer le graphique : 
        fig = plt.figure()
        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax2 = ax1.twinx() #création du 2e axe
        ax3 = ax1.twinx() #création du 3e axe
        ax3.spines["right"].set_position(("outward", 60))  # Décalage de l'axe pour lisibilité

        sns.lineplot(data=CA_mois, x='month_year', y='net_sales', label='CA', ax=ax1, color='black')
        sns.lineplot(data=sessions_mois, x='month_year', y='Sessions', label='Sessions', ax=ax2, color='grey')
        sns.lineplot(data=commandes_mois, x='month_year', y='Orders', label='Commandes', ax=ax3, color='#27BDF4')

        ax1.set_ylabel('CA', color='black')
        ax2.set_ylabel('Nombre Sessions', color='grey')
        ax3.set_ylabel('Nombre Commandes', color='#27BDF4')

        ax1.tick_params(axis='y', labelcolor='black')
        ax2.tick_params(axis='y', labelcolor='grey')
        ax3.tick_params(axis='y', labelcolor='#27BDF4')

        ax1.set_ylim(0, 400000)
        ax3.set_ylim(0, 20000)
        ax1.tick_params(axis='x', rotation=45)

        #avoir une légende commune :
        lines = ax1.get_lines() + ax2.get_lines() + ax3.get_lines()
        labels = ['CA', 'Sessions', 'Commandes']
        plt.legend(lines, labels, loc='upper left')

        plt.title('Évolution du CA vs nombre de Commandes vs nombre de Sessions par mois')
        plt.xlabel('Mois')
        st.pyplot(fig)

        st.write("Le nombre de sessions, le CA et le nombres de commandes sont très corrélés, ce qui est logique. " \
        "En revanche, on observe une 'anomalie' du CA en mai 2025 non-associé à un pic de visites sur le site." \
        "Ce pic est dû à un passage télévisé de Coral Gardeners dans l'émission 'Quotidien', ayant créé des leads très qualifiés sur une soirée résultant en un boost de commandes mais pas de boost de sessions dans le mois).")



    with st.expander("Evolution du CA sur la période des données"):
        #Créer un dataframe avec le CA généré par mois
        df_shopify["month_year"] = pd.to_datetime(df_shopify["month_year"])
        df_monthly = df_shopify.groupby('month_year')['net_sales'].sum()
        
        #décomposer la série pour avoir la vision de la tendance globale et de la saisonalité
        from statsmodels.tsa.seasonal import seasonal_decompose
        decompose = seasonal_decompose(df_monthly, model='multiplicative', period=12)
        fig = decompose.plot()
        st.pyplot(fig)
        st.write("- Une tendance globale à la hausse du chiffre d'affaires est observée, particulièrement en 2024. Cette hausse ralentit en 2025.")
        ("- Il y a une claire saisonnalité avec un pic à chaque période de Noël.")
        ("Cette analyse de la tendance, en linéarisant les résidus, serait à étoffer. Pour autant, nous n'en avons pas eu besoin pour le modèle de Clustering.")
    
    with st.expander("Répartition des ventes par catégorie de produit"):
        df_shopify_produit = pd.read_csv('../Datas/Shopify_product_level.csv')
        #Création d'une colonne mois_année avec le mois et l'année (type str pour les graphiques):
        df_shopify_produit["Day"] = pd.to_datetime(df_shopify_produit["Day"])
        df_shopify_produit['month_year'] = df_shopify_produit['Day'].dt.to_period('M').astype(str)

        # Obtenir la somme totale de ventes pour chaque catégorie de produit et les trier par ordre décroissant
        product_commandes = df_shopify_produit.groupby('product_category')["Quantity ordered"].sum().reset_index()
        product_commandes = product_commandes.sort_values(by='Quantity ordered', ascending=False)

        # Afficher la répartition du nombre de ventes pour chaque catégorie de produit 
        fig = plt.figure(figsize=(15, 15))

        plt.subplot(2, 1, 1)
        sns.barplot(data=product_commandes, x="Quantity ordered", y="product_category", color='#27BDF4')
        plt.title("Répartition des ventes par catégories de produit")
        plt.xlabel("Nombre de ventes")
        plt.ylabel("Catégories de produits")
        st.pyplot(fig)

        # Obtenir le total des ventes mensuelles du top 3 des catégories de produit
        coral_adoption = df_shopify_produit.loc[(df_shopify_produit['product_category']=='CORAL ADOPTION')]
        coral_adoption = coral_adoption.groupby('month_year')['Quantity ordered'].sum().reset_index()

        merch = df_shopify_produit.loc[(df_shopify_produit['product_category']=='MERCH')]
        merch = merch.groupby('month_year')['Quantity ordered'].sum().reset_index()

        xperience = df_shopify_produit.loc[(df_shopify_produit['product_category']=='XPERIENCE')]
        xperience = xperience.groupby('month_year')['Quantity ordered'].sum().reset_index()


        # Afficher l'évolution des ventes du top 3 des catégories de produit
        fig = plt.figure(figsize=(15, 15))
        plt.subplot(2, 1, 2)
        sns.lineplot(data=coral_adoption, x='month_year', y='Quantity ordered', label='Coral adoption', color='#27BDF4')
        sns.lineplot(data=merch, x='month_year', y='Quantity ordered', label='Merch', color='black')
        sns.lineplot(data=xperience, x='month_year', y='Quantity ordered', label='Xperience', color='grey')

        plt.title("Evolution des ventes pour les 3 catégories de produit les plus vendues")
        plt.xlabel("Mois")
        plt.ylabel("Nombre de ventes")
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)
        st.write("Coral Gardeners vend avant tout des Coraux à adopter. Le merch, les expériences et les produits B2B arrivent ensuite. Les ventes de cartes cadeaux et les donations sont marginales.")
        ("Nous observons que les ventes de coraux à adopter, de merch et d'expériences suivent la même tendance (il n'y a pas de saisonnalité pour un produit en particulier).")
        ("")
        ("Dans la suite du projet, nous n’allons pas nous intéresser au type de produit commandé, mais uniquement au nombre de commandes et à leur valeur associée.")
        ("En effet, le produit “Coral Adoption” représente l’écrasante majorité des ventes : une analyse à la maille produit n’est pas pertinente (et non intéressante pour Coral Gardeners).")

    
    st.subheader("Analyse à la maille client (données Shopify)")

    with st.expander("Nombre de commandes par client"):
         #Récupération des commandes par client
        commandes_par_client = df_shopify.groupby('customer_id')['order_id'].count().reset_index()
        commandes_par_client.columns = ['customer_id', 'nb_commandes']
        commandes_par_client = commandes_par_client.sort_values('nb_commandes', ascending=False)

        #Boxplot
        fig = plt.figure(figsize=(15, 4))
        sns.boxplot(data=commandes_par_client, x='nb_commandes', color='#27BDF4').set(title="Répartition du nombre de commandes par client", xlabel="Nombre de commandes")
        st.pyplot(fig)

        #Graphique
        commandes = []
        for i in range (1, 40):
            commande = commandes_par_client.loc[commandes_par_client["nb_commandes"] == i]['customer_id'].count()
            commandes.append(commande)

        df_commandes_par_nombre_articles = pd.DataFrame(commandes, columns=['Nombre de clients ayant réalisé x commandes'], index=range(1, 40))
        fig = plt.figure(figsize=(15, 10))
        plt.plot(df_commandes_par_nombre_articles, color='#27BDF4')
        plt.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xticks(range(0, 42, 2))
        plt.title('Nombre de clients en fonction du nombre de commandes réalisées')
        plt.xlabel('Nombre de commandes')
        plt.ylabel('Nombre de clients ayant réalisé x commandes')
        st.pyplot(fig)




        # --- Données clients
        clients = df_shopify.drop_duplicates(subset='customer_email', keep='last')
        client_counts = clients['repeat_flag'].value_counts().sort_index()

        # --- Données commandes
        order_counts = df_shopify['repeat_flag'].value_counts().sort_index()

        labels = ['One-shot', 'Repeat']
        colors = ['#000000', '#27BDF4']

        # --- Figure avec 2 subplots côte à côte
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # --- Graphique 1 : Clients
        client_pcts = [client_counts[0]/client_counts.sum()*100, client_counts[1]/client_counts.sum()*100]
        bars1 = ax1.bar(labels, client_pcts, color=colors, width=0.5)

        for bar, pct, val in zip(bars1, client_pcts, [client_counts[0], client_counts[1]]):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{pct:.2f}%\n({val:,} clients)',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

        ax1.set_title('Répartition des clients', fontsize=13, fontweight='bold')
        ax1.set_ylabel('% de clients', fontsize=11)
        ax1.set_ylim(0, 110)

        # --- Graphique 2 : Commandes
        order_pcts = [order_counts[0]/order_counts.sum()*100, order_counts[1]/order_counts.sum()*100]
        bars2 = ax2.bar(labels, order_pcts, color=colors, width=0.5)

        for bar, pct, val in zip(bars2, order_pcts, [order_counts[0], order_counts[1]]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{pct:.1f}%\n({val:,} commandes)',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

        ax2.set_title('Répartition des commandes', fontsize=13, fontweight='bold')
        ax2.set_ylabel('% de commandes', fontsize=11)
        ax2.set_ylim(0, 110)

        # --- Titre global
        fig.suptitle('Dynamique One-shot vs Repeat', fontsize=15, fontweight='bold', y=1.02)

        plt.tight_layout()
        st.pyplot(fig)
        st.write("La grande majorité des clients de Coral Gardeners ont réalisé une seule commande (plus de 90% des clients) représentant plus de 80% des commandes.")


    with st.expander("Intervalles entre les commandes (cycle naturel de réachat)"):
        ## Cycle naturel de réachat
        # Étape 1 — dataframe des commandes repeat
        repeat_orders = df_shopify[df_shopify['order_rank'] > 1].copy()

        # Étape 2 — filtrer same-day et outliers
        repeat_clean = repeat_orders[(repeat_orders['days_since_previous_order'] >= 1) & (repeat_orders['days_since_previous_order'] <= 365)].copy()

        # Étape 3 — extraire la Series pour le graphique
        data = repeat_clean['days_since_previous_order']

        # Etape 4 - graphique
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(data, bins=60, color='#27BDF4', edgecolor='white', linewidth=0.5)

        # Médiane et moyenne
        ax.axvline(data.median(), color='#000000', linewidth=2,
                linestyle='--', label=f'Médiane : {data.median():.0f}j')
        ax.axvline(data.mean(), color='gray', linewidth=2,
                linestyle='--', label=f'Moyenne : {data.mean():.0f}j')

        # Seuils abonnement
        ax.axvline(30, color='#000000', linewidth=1, linestyle=':', alpha=0.5, label='30j (mensuel)')
        ax.axvline(60, color='#000000', linewidth=1, linestyle=':', alpha=0.4, label='60j (bimensuel)')
        ax.axvline(90, color='#000000', linewidth=1, linestyle=':', alpha=0.3, label='90j (trimestriel)')

        # Zones colorées
        ax.axvspan(0,  30, alpha=0.05, color='#27BDF4', label='Zone < 30j (40%)')
        ax.axvspan(30, 60, alpha=0.05, color='black',   label='Zone 30-60j (14%)')

        ax.set_title('Cycle naturel de réachat\n(same-day exclus, outliers > 365j exclus)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Jours entre deux commandes', fontsize=12)
        ax.set_ylabel('Nombre de commandes', fontsize=12)
        ax.legend(fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        st.write("40% rachètent dans les 30 jours : traduisant une potentielle appétence pour une abonnement mensuel.")
        st.write("54% rachètent dans les 60 jours (et médiane à 48 jours): un abonnement bimestriel serait envisageable pour plus de la moitié des clients repeat.")
        st.write("Enfin, 63% rachètent dans les 90 jours (et moyenne à 65 jours) : un abonnement trimestriel serait envisageable pour près de 2 clients repeat sur 3.")
        st.write("Cependant, il est important de noter que ces chiffres sont à relativiser : les clients ayant un cycle de réachat plus long sont aussi ceux qui ont réalisé le plus de commandes (et donc qui sont les plus fidèles). ")
        st.write("Nous évaluerons ces éléments par notre modèle de clustering, qui nous permettra d'identifier les segments de clients les plus compatibles avec un modèle d'abonnement mensuel, le type d'abonnement nous intéressant ici.")
        st.write("")
        st.write("D'autres analyses ont également été effectuées. Brievement :")
        ("- Analyse de la rétention au cours du temps : Le volume grossit mais pas la rétention : les commandes augmentent au cours du temps et explosent à Noël, mais que la proportion des repeat reste globalement stable. Coral Gardeners a amélioré l’acquisition mais pas la rétention.")
        ("- Analyse du taux de rétention : Globalement, il ne s'améliore pas. Sur 3 ans, le repeat rate oscille entre 9% et 11% en moyenne.")
        ("- Analyse de l’intervalle moyen entre deux commandes : reste le même d’une année sur l’autre.")

    with st.expander("Analyse sur la récence des clients"):
         # Dernière commande par client uniquement
        last_orders = df_shopify.sort_values('order_date').groupby('customer_email').last().reset_index()

        # Appliquer les seuils
        def segment_recency(days):
            if days < 180: 
                return 'Actif'
            elif days < 365:
                return 'Dormant'
            elif days < 730:
                return 'À risque'
            else:
                return 'Perdu'

        last_orders['segment_recency'] = last_orders['recency_days'].apply(segment_recency)

        #Graphiques
        segments = ['Actif', 'Dormant', 'À risque', 'Perdu']
        colors   = ['#27BDF4', '#000000', '#555555', '#AAAAAA']
        counts   = [9780, 7747, 12329, 3745]
        pcts     = [c / sum(counts) * 100 for c in counts]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

        # --- Graphique 1 : Bar chart nb clients
        bars = ax1.bar(segments, counts, color=colors, width=0.6)
        for bar, val, pct in zip(bars, counts, pcts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
                    f'{val:,}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

        ax1.set_title('Segmentation clients par récence', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Nombre de clients', fontsize=11)
        ax1.set_ylim(0, 15000)

        # --- Graphique 2 : Donut
        wedges, texts, autotexts = ax2.pie(
            counts,
            labels=segments,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.75,
            wedgeprops=dict(width=0.5)
        )
        for text in autotexts:
            text.set_fontsize(10)
            text.set_fontweight('bold')
            text.set_color('white')

        ax2.set_title('Répartition des segments (%)', fontsize=13, fontweight='bold')

        fig.suptitle('Segmentation Recency clients',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
        st.write("Segment par segment :")
        st.write("- Actif — 29.1% — ~50j de recency : Ce sont les clients engagés récemment. C'est la cible prioritaire pour convertir en abonnés, ils sont dans une dynamique d'achat active.")
        st.write("- Dormant — 23.1% — ~280j de recency : Ont commandé il y a 6 mois à 1 an. Pour un business Noël, beaucoup de ces clients reviendront naturellement en décembre. Réactivables par email marketing.")
        st.write("- À risque — 36.7% — ~458j de recency : C'est le segment le plus large et le plus préoccupant. Ont commandé il y a 1 à 2 ans — ils s'éloignent mais ne sont pas encore perdus. C'est le levier principal — récupérer même 10% de ces 12 329 clients représente 1 233 abonnés potentiels.")
        st.write("- Perdu — 11.1% — ~857j de recency : Plus de 2 ans sans commande. Coût de réactivation élevé, ROI faible. À déprioriser.")
        st.write("L’analyse met en évidence que 70.9 % des clients n’ont pas commandé depuis plus de six mois, avec un segment \"À risque\" qui représente à lui seul 36.7% des clients. Ce constat renforce directement la nécessité d'un modèle d'abonnement: non seulement pour fidéliser les clients actifs mais aussi pour créer un mécanisme de rétention structurel qui empêche les clients de glisser vers les segments dormant et à risque.")



    st.subheader("Analyse des données CRM (données Klaviyo)")

    with st.expander("Analyse de la répartition de chaque variable"):
        klaviyo = pd.read_csv('../Datas/klaviyo_2023_2025_final.csv')
        fig = plt.figure(figsize=(10, 5))
        klaviyo.boxplot(column=['emails_received','opens', 'clicks', 'orders'])
        plt.title("Boxplot des variables CRM")
        st.pyplot(fig)
        st.write("En moyenne, les clients ont reçu 11 emails et en ont ouvert 9,7.")
        ("Nous pourrions conclure que les clients sont très engagés avec Coral Gardeners et lisent la majorité des emails.")
        ("Cependant, nous pouvons noter que certains clients ouvrent leurs emails plusieurs fois (opens > received), avec des valeurs très extrêmes dans le nombre d'emails ouverts, biaisant la moyenne")
        ("Par ailleurs, grâce à cette analyse, des emails corporate (@coral-gardeners) ont été détectés dans le dataframe parmi les valeurs extrèmes, à même titre que les emails des clients. Ils ont été retirés car faussaient l'analyse.")

#PAGE PREPROCESSING
if page == "Preprocessing":
    st.title("Coral Gardeners")
    st.header("Preprocessing des données pour la modélisation")

    #Création de variables pour la modélisation
    st.subheader("Création de variables pour la modélisation")
    with st.expander("Dataset Klaviyo"):
        st.markdown("""
            - Vérification et gestion des NaNs et doublons : il n’y en a aucun  
            - Gestion des valeurs extrêmes : suppression des emails corporate et B2B  
            - Harmonisation des emails : mise en minuscule et suppression des espaces   
            - Ajout de la variable open_rate : taux d’ouverture  = mails ouverts / mails reçus  
            - Ajout de la variable click_rate : taux de clic = nombre de clics / mails ouverts  
            - Remplacement des NaNs issues des divisions par 0 pour ces nouvelles variables par “0”
            """)
     
    with st.expander("Dataset Shopify niveau commandes"):
        st.markdown("""
            Ajout des variables :   
            - order_rank = rang chronologique de la commande pour un client donné  
            - is_first_order = indicateur de première commande (1 si oui, 0 si non)  
            - is_repeat_order = indicateur de commande répétée (1 si oui, 0 si non)  
            - previous_order_date = date de la commande précédente d’un même client  
            - days_since_previous_order = intervalle en jours entre deux commandes successives  
            - first_order_date = date de première commande du client  
            - recency_days = nombre de jours écoulés entre la commande et la date d’extraction des données.
            """)
    
    with st.expander("Dataset Shopify niveau client"):
        st.markdown("""
            Création du dataset avec les variables ci-dessous :    
            - nb_orders = nombre total de commandes  
            - longtime_value = chiffre d’affaires total généré par le client sur la période  
            - avg_order_value = panier moyen du client  
            - recency_days = nombre de jours depuis la dernière commande  
            - customer_lifetime_days = durée entre la première et la dernière commande  
            - avg_days_between_orders = intervalle moyen entre les commandes  
            - repeat_flag = indicateur de commande répétée (1 si oui, 0 si non)  
            - billing_country = pays associé au client 
            """)


    #Dataset final modélisation
    df_final = pd.read_csv("../Datas/coralgardeners_final_sans_preprocessing.csv")
    
    st.subheader("Dataset final")
    st.markdown("Voici un schéma récapitulatif des variables retenues dans le dataset final (utilisé pour la modélisation)")
    st.image("../Datas/schéma_données.png", caption="Schéma des données", use_container_width=True)
    if st.checkbox("Afficher un aperçu du dataset final", key="final"):
        st.dataframe(df_final.head(10))

    #Normalisation du dataset final
    df_final_encodé = pd.read_csv("../Datas/coralgardeners_final_top9paysABE.csv")
    st.subheader("Normalisation et encodage du dataset final")
   
    with st.expander("Traitement des commandes avant normalisation"):
        st.markdown("""
            Afin de pouvoir normaliser les colonnes customer_lifetime_days et avg_days_between_orders, 
            dans lesquelles nous avons remplacé les NaNs par -1, nous :  
            - créons des colonnes “existence d’évènements” = 0 si la colonne indique -1 et 1 sinon  
            - remplaçons les valeurs -1 par la médiane des colonnes
            """)
        
    with st.expander("Normalisation des variables quantitatives (et non binaires)"):
        st.markdown("""
            Nous vérifions la distribution de nos variables pour décider de la méthode de normalisation à appliquer :  
            - variables avec pic massif et une longue queue :  toutes nos variables sauf emails_received. 
            Nous appliquons une normalisation RobustScaler, moins sensible aux outliers. 
            Cependant, nous nous rendons compte que cela n’est pas suffisant (nous avons encore des maximums > 900) : 
            nous appliquons donc en amont une méthode Log1p, afin de compresser les longues pattes de répartition des variables.  
            -  variable avec plusieurs pics discernables : emails_received. Nous appliquons une normalisation MinMaxScaler.
            """)
        
    with st.expander("Encodage de la colonne billing_country"):
        st.markdown("""
            Nous réalisons 2 encodages différents, afin de tester lequel est le plus adapté pour notre modélisation :  
            - Méthode 1 : conservation uniquement du détail des pays top 9 en CA (comme vu dans l’analyse graphique, le CA des autres pays devient ensuite très faible), 
            les autres sont renommés “Autres”. Puis encodage via un get_dummies.  
            Taille du dataframe : 32 937 lignes, 22 colonnes  
            - Méthode 2 : création d’une colonne “billing_continent”, prenant pour valeur le continent de chaque pays, 
            sauf pour les Etats-Unis, qui conserve la valeur “United States”. En effet, étant donné l’importance de ce pays dans nos données, 
            nous le conservons à part. Suppression de la colonne “billing_country”. Encodage de la colonne “billing_continent” via un get_dummies.  
            Taille du dataframe : 32 937 lignes, 18 colonnes
            """)
    
    if st.checkbox("Afficher un aperçu du dataset final encodé", key="final_encodé"):
        st.dataframe(df_final_encodé.head(10))



# PAGE MODELISATION
if page == "Clustering":
    st.title("Coral Gardeners")
    st.header("Modélisations réalisées et leurs résultats")

    # Chargement des données et des modèles enregitrés
    df = pd.read_csv("../Datas/coralgardeners_final_top9paysABE.csv", index_col="customer_id")
    models = {
        "Modèle 7 - K-Means - PCA - retrait pays, 2x'existence...', repeat_flag": joblib.load("../models/model7.joblib"),
        "Modèle 5 - K-Means - PCA - retrait pays, 2x'existence...' ": joblib.load("../models/model5.joblib"),
        "Modèle 6 - K-Means - PCA - retrait pays, repeat_flag": joblib.load("../models/model6.joblib"),
        "Modèle 8A - K-Means - dataset one-shot": joblib.load("../models/model8A.joblib"),        
        "Modèle 8B - K-Means - dataset repeat": joblib.load("../models/model8B.joblib"),
        "Modèle 2 - CAH - retrait pays": joblib.load("../models/model2.joblib"),
        "Modèle 11 - GMM": joblib.load("../models/model11.joblib")
    }

    # Preprocessing de chaque modèle
    def preprocess_data(df, model_name):

        if model_name == "Modèle 2 - CAH - retrait pays":
            X = df.drop(columns=['billing_country_Autre', 'billing_country_Australia', 
                                'billing_country_Canada', 'billing_country_France', 
                                'billing_country_Germany', 'billing_country_Italy', 
                                'billing_country_Netherlands', 'billing_country_Switzerland', 
                                'billing_country_United Kingdom', 'billing_country_United States'])
            features = X.columns.tolist()

        elif model_name == "Modèle 5 - K-Means - PCA - retrait pays, 2x'existence...' ":
            features = ['nb_orders', 'longtime_value', 'avg_order_value', 'recency_days',
                        'customer_lifetime_days', 'avg_days_between_orders', 'repeat_flag',
                        'emails_received', 'open_rate', 'click_rate']
            X = PCA(n_components=0.9).fit_transform(df[features])

        elif model_name == "Modèle 6 - K-Means - PCA - retrait pays, repeat_flag":
            features = ['nb_orders', 'longtime_value', 'avg_order_value', 'recency_days',
                        'customer_lifetime_days', 'existence_customer_lifetime_days',
                        'avg_days_between_orders', 'existence_avg_days_between_orders',
                        'emails_received', 'open_rate', 'click_rate']
            X = PCA(n_components=0.9).fit_transform(df[features])

        elif model_name == "Modèle 7 - K-Means - PCA - retrait pays, 2x'existence...', repeat_flag":
            features = ['nb_orders', 'longtime_value', 'avg_order_value', 'recency_days',
                        'customer_lifetime_days', 'avg_days_between_orders',
                        'emails_received', 'open_rate', 'click_rate']
            X = PCA(n_components=0.9).fit_transform(df[features])

        elif model_name == "Modèle 8A - K-Means - dataset one-shot":
            df_filtered = df[df["repeat_flag"] == 0].drop(
                columns=['billing_country_Autre', 'billing_country_Australia', 
                        'billing_country_Canada', 'billing_country_France', 
                        'billing_country_Germany', 'billing_country_Italy', 
                        'billing_country_Netherlands', 'billing_country_Switzerland', 
                        'billing_country_United Kingdom', 'billing_country_United States',
                        'existence_customer_lifetime_days', 'customer_lifetime_days',
                        'existence_avg_days_between_orders', 'avg_days_between_orders',
                        'repeat_flag', 'nb_orders', 'avg_order_value'])
            features = df_filtered.columns.tolist()
            X = PCA(n_components=0.9).fit_transform(df_filtered)

        elif model_name == "Modèle 8B - K-Means - dataset repeat":
            df_filtered = df[df["repeat_flag"] == 1].drop(
                columns=['billing_country_Autre', 'billing_country_Australia', 
                        'billing_country_Canada', 'billing_country_France', 
                        'billing_country_Germany', 'billing_country_Italy', 
                        'billing_country_Netherlands', 'billing_country_Switzerland', 
                        'billing_country_United Kingdom', 'billing_country_United States',
                        'existence_customer_lifetime_days',
                        'existence_avg_days_between_orders', 'repeat_flag'])
            features = df_filtered.columns.tolist()
            X = PCA(n_components=0.9).fit_transform(df_filtered)

        elif model_name == "Modèle 11 - GMM":
            features = ['nb_orders', 'longtime_value', 'avg_order_value', 'recency_days',
                        'customer_lifetime_days', 'existence_customer_lifetime_days',
                        'avg_days_between_orders', 'existence_avg_days_between_orders',
                        'emails_received', 'open_rate', 'click_rate']
            X = PCA(n_components=0.9).fit_transform(df[features])

        else:
            raise ValueError(f"Modèle inconnu : {model_name}")

        return X, features

    # Choix du modèle
    option = st.selectbox("Choix du modèle", list(models.keys()))

    # Modélisation
    X, features = preprocess_data(df, option)
    model = models[option]
    try:
        labels = model.predict(X)
    except:
        labels = model.fit_predict(X)

    # Affichage des features du modèle sélectionné
    st.write("### Features du modèle")
    st.dataframe(pd.DataFrame(features, columns=["Features"]))

    # Choix de la métrique silhouette selon le modèle
    if "CAH" in option:
        silhouette = silhouette_score(X, labels, metric="sqeuclidean")
        silhouette_name = "Silhouette (sqeuclidean)"
    else:
        silhouette = silhouette_score(X, labels)
        silhouette_name = "Silhouette (euclidean)"

    # Calcul des métriques
    results = {
        silhouette_name: silhouette,
        "Davies-Bouldin": davies_bouldin_score(X, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels)
    }

    # Mise des résultats dans un tableau
    df_results = pd.DataFrame([results])
    st.write("### Résultats du modèle")
    st.dataframe(df_results)


    # Camembert des clusters
    st.write("### Répartition des clusters")
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    labels_text = [
        f"Cluster {i}\n(n={count})"
        for i, count in zip(cluster_counts.index, cluster_counts.values)
    ]

    fig, ax = plt.subplots()
    ax.pie(
        cluster_counts,
        labels=labels_text,
        autopct='%1.1f%%',
        startangle=90
    )
    ax.set_title("Répartition des clusters")
    st.pyplot(fig)

    st.subheader("Modèle retenu")
    st.markdown("""
            Nous retenons le **modèle 7**, qui regroupe à la fois :  
            - les métriques parmi les meilleures  
            - une répartition équilibrée de la population entre les clusters  
            - un nombre de cluster opérationnellement interprétable            
            """)



# PAGE INTERPRETATION
if page == "Interprétation":
    st.title("Coral Gardeners")
    st.header("Interprétation du modèle sélectionné")
    
    # Chargement des données et des modèles
    df = pd.read_csv("../Datas/coralgardeners_final_top9paysABE.csv", index_col="customer_id")
    model = joblib.load("../models/model7.joblib")
    rf = joblib.load("../models/random_forest.joblib")
    
    # I/ Récupération des clusters
    # Preprocessing
    df = df[['nb_orders', 'longtime_value', 'avg_order_value', 'recency_days',
            'customer_lifetime_days', 'avg_days_between_orders',
            'emails_received', 'open_rate', 'click_rate']]
    df_pca = PCA(n_components=0.9).fit_transform(df)
    
    # Clustering
    try:
        labels = model.predict(df_pca)
    except:
        labels = model.fit_predict(df_pca)
    df['cluster'] = labels

    # II / Random Forest
    st.subheader("Interprétation avec un Random Forest : importance des variables")
    X = df.drop(columns='cluster')
    y = df['cluster']
    rf.fit(X, y)

    # Importance des variables
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    feature_importances.plot(kind='bar', ax=ax)
    ax.set_title("Importance des variables — Random Forest")
    ax.set_ylabel("Importance")
    plt.xticks(rotation=45)

    st.pyplot(fig)

    # III/ PROFIL DES CLUSTERS
    st.subheader("Interprétation des clusterings")

    df_group = df.groupby('cluster').mean(numeric_only=True)
    categories = df_group.columns
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for i, row in df_group.iterrows():
        values = row.tolist()
        values += values[:1]

        ax.plot(angles, values, label=f'Cluster {i}')
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    plt.title("Profil moyen par cluster (Radar)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1))

    st.pyplot(fig)

    st.image("Moyenne_clusters.png", caption="Moyenne des variables par cluster")
    st.subheader("Interprétation des clusters")
    
    # Cluster 0
    with st.expander("Clients VIP à forte valeur (2,9k clients) - Cluster #0"):
        st.markdown("""
            - valeur: +++ (panier moyen et longtime_value) 
            - récence : ++ (~8 mois)
            - fidélité : ++ (nb commandes 1,3 et intervalle de repeat élevé) 
            - engagement : + (open rate = 104%, click rate = 32%).
            """)

    # Cluster 5
    with st.expander("Clients nouveaux ou réactivés très récemment (4,8k clients) - Cluster #5"):
        st.markdown("""
            - valeur: - (panier moyen et longtime_value) 
            - récence : +++ (~17 jours)
            - fidélité : - (repeat_flag faible) 
            - engagement : + (open rate = 99%, click rate = 39%).
            """)

    # Cluster 3
    with st.expander("Clients fidèles à réactiver (685 clients) - Cluster #3"):
        st.markdown("""
            - valeur: + (panier moyen et longtime_value) 
            - récence : + (<1 an)
            - fidélité : ++ (repeat_flag = 1, mais intervalle très faible entre 2 commandes) 
            - engagement : +++ (open rate = 110%, click rate = 53%).
            """)

    # Cluster 2
    with st.expander("Clients moyens (9,7k clients) - Cluster #2"):
        st.markdown("""
            - valeur: + (panier moyen et longtime_value) 
            - récence : - (>12 mois)
            - fidélité : - (repeat_flag faible) 
            - engagement : -
            """)

    # Cluster 4
    with st.expander("Clients engagés réalisant peu d’achats (2,3k clients) - Cluster #4"):
        st.markdown("""
            - valeur: - (panier moyen et longtime_value) 
            - récence : + (~9 mois)
            - fidélité : - (repeat_flag faible) 
            - engagement : + (Engagement atypique : le plus fiable open rate (25%) mais un click rate très élevé (224%))
            """)

    # Cluster 1
    with st.expander("Clients one-shot perdus (12,4k clients) - Cluster #1"):
        st.markdown("""
            - valeur : --- (panier moyen et longtime_value) 
            - récence : -- (>14 mois)
            - fidélité : -- (repeat_flag quasi nul) 
            - engagement : -- (engagement le plus faible, même si correct).
            """)


    
# PAGE RECOMMANDATIONS ET LIMITES
if page == "Recommandations et limites":
    st.title("Coral Gardeners - Clustering clients")
    st.header("Recommandations et Limites")
    st.subheader("Recommandations à Coral Gardeners")
    st.write("-> comment adresser les différents clusters de clients afin de sonder leur intérêt pour une formule d'abonnement")

    # Cluster 0
    with st.expander("Clients VIP à forte valeur (2,9k clients) - Cluster #0"):
        st.markdown("""
            **Objectif:** comprendre comment convertir ces clients à forte valeur en soutiens réguliers  
            **Ton de l’email :** “Continuez d’agir avec Coral Gardeners”  

            **Exemple de questions :**
            - Qu’est-ce qui vous plaît le plus dans votre soutien aujourd’hui ?
            - Quel niveau d’impact aimeriez-vous voir concrètement (ex : nombre de coraux restaurés) ?
            - Seriez-vous intéressé par une formule qui simplifie votre engagement régulier ?
            - Qu’est-ce qui vous ferait passer à un abonnement (prix, visibilité de l’action, avantages) ?
            - Quel serait pour vous un prix ‘juste’ pour un soutien mensuel ?

            **Bonus :** proposer une offre d’abonnement exclusive à l’essai  
            """)

    # Cluster 5
    with st.expander("Clients nouveaux ou réactivés très récemment (4,8k clients) - Cluster #5"):
        st.markdown("""
            **Objectif:** comprendre comment capitaliser sur ces clients via un abonnement  
            **Ton de l’email :** “Merci pour votre soutien récent, souhaitez-vous agir à plus long terme pour l’océan ?”  

            **Exemple de questions :**
            - Qu’est-ce qui vous a donné envie d’adopter un corail récemment ?
            - Avez-vous envisagé de soutenir Coral Gardeners sur la durée ?
            - Qu’est-ce qui rendrait un engagement mensuel attractif pour vous ? (prix, visibilité de l’action, avantages) ?

            **Bonus :** proposer une offre d’abonnement exclusive à l’essai  
            """)

    # Cluster 3
    with st.expander("Clients fidèles à réactiver (685 clients) - Cluster #3"):
        st.markdown("""
            **Objectif:** comprendre si une formule d’abonnement permettrait de réactiver ces clients très engagés  
            **Ton de l’email :** “L’océan a toujours besoin de vous !”  

            **Exemple de questions :**
            - Qu’est-ce qui vous a motivé à adopter plusieurs coraux en peu de temps ?
            - Est-ce qu’une formule d’abonnement serait intéressante pour faciliter votre soutien?
            - Qu’est-ce qui rendrait un abonnement vraiment utile pour vous ?
            - Quel type de retour ou avantage vous ferait plaisir ?

            **Bonus :** conserver une offre one-shot en repli pour les clients qui souhaitent soutenir “de temps en temps”, et leur proposer de soutenir Coral Gardeners à nouveau  
            """)

    # Cluster 2
    with st.expander("Clients moyens (9,7k clients) - Cluster #2"):
        st.markdown("""
            **Objectif:** comprendre à quel tarif ces clients seraient prêts à soutenir Coral Gardeners de façon régulière  
            **Ton de l’email :** “Merci pour votre soutien, Coral Gardeners a besoin de vous”  

            **Exemple de questions :**
            - Avez-vous besoin de plus de visibilité sur l’impact de votre contribution ?
            - Qu’est-ce qui vous empêcherait de soutenir Coral Gardeners tous les mois ?
            - Seriez-vous intéressé pour nous soutenir de façon régulière via un abonnement ? Si oui, qu’est-ce qui vous ferait passer à un abonnement (prix, visibilité de l’action, avantages) ?
            """)

    # Cluster 4
    with st.expander("Clients engagés réalisant peu d’achats (2,3k clients) - Cluster #4"):
        st.markdown("""
            **Objectif:** comprendre pourquoi ces clients qui semblent suivre Coral Gardeners ne convertissent pas  
            **Ton de l’email :** “Le devenir de l’océan vous touche, quelles actions êtes-vous prêts à réaliser ?”  

            **Exemple de questions :**
            - Suivez-vous régulièrement l’activité de Coral Gardeners ?
            - Avez-vous déjà hésité à ré-adopter un corail ? Pourquoi ?
            - Qu’est-ce qui vous freine le plus aujourd’hui : prix, informations, compréhension, autre ?
            - Souhaiteriez-vous nous soutenir plus activement ? Si oui, comment ?
            - Préférez-vous soutenir ponctuellement ou régulièrement ? Pourquoi ?
            """)

    # Cluster 1
    with st.expander("Clients one-shot perdus (12,4k clients) - Cluster #1"):
        st.markdown("""
            **Objectif:** le questionnaire auprès de ces clients n’aura pas pour objectif de sonder leur intérêt pour une formule d’abonnement mais bien de comprendre les raisons de leur désengagement.  
            **Ton de l’email :** “Nous sommes tristes de ne plus vous compter parmi nos soutiens …”  

            **Exemple de questions :**
            - Pourquoi n’avez vous pas réitéré l’adoption d’un corail ? Qu’est-ce qui vous a manqué après votre adoption ?
            - Un soutien régulier aurait-il changé quelque chose ?
            - Qu’est-ce qui vous ferait adopter à nouveau un corail ou agir à nouveau avec Coral Gardeners ?
            """)

    st.subheader("Limites à avoir en visibilité")
    st.write("-> il existe plusieurs limites à connaître avant d'opérationnaliser ce projet")
    st.write("")
    
    st.write("1/ Limites liées au modèle")
    with st.expander("Choix du modèle"):
      st.markdown("""
        Plusieurs modèles testés donnent des segmentations différentes (notamment sur le nombre de clusters).
        Cela suggère qu’une autre segmentation, non testée dans le cadre de ce projet, pourrait être plus pertinente.
        """)
      
    with st.expander("Clients outsiders"):
      st.markdown("""
        Le clustering repose principalement sur quelques variables clés.  
        Certains clients peuvent donc être mal classés si une seule variable est dominante (ex : taux de clic élevé).
        Recommandation : affiner la segmentation avec des règles métier pour éviter les biais liés aux moyennes.
        """)

    st.write("2/ Limites pour opérationnaliser le clustering")
    with st.expander("Cohérence métier de certains clusters"):
      st.markdown("""
        Certains clusters (notamment le cluster 4) sont difficiles à interpréter.
        Peut être serait il pertninent de regrouper ces clients dans d'autres clusters existants.
        """)
    
    with st.expander("Taille de certains clusters"):
      st.markdown("""
        Certains clusters (ex : cluster 3) sont de petite taille.
        Cela pose la question de leur pertinence opérationnelle.
        """)

    with st.expander("Nombre de clusters"):
      st.markdown("""
        6 clusters peuvent être complexes à adresser opérationnellement.
        Il peut être intéressant de regrouper certains segments afin de les adresser :
        - Clients à potentiel : #0 et #5  
        - Clients à réactiver : #2, #3 et #4  
        - Clients perdus : #1  
        """)

    st.write("3/ Variables non prises à en compte, à creuser")
    with st.expander("Non prise en compte du pays"):
      st.markdown("""
        Le pays n’a pas été retenu dans le modèle.
        Pourtant, comme vu dans la partie data visualization, des différences existent selon les pays (ex : États-Unis).  
        Cela pourrait être approfondi.
        """)

    with st.expander("Influence de la publicité (META)"):
      st.markdown("""
        Les comportements d’achat semblent fortement influencés par les campagnes META.
        Cette variable n’a pas été intégrée car nous ne savions pas la lier au CRM.  
        Elle pourrait cependant améliorer l’analyse.
        """)  

