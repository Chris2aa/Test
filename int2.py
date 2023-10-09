# Importing necessary libraries
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime
# Importation des bibliothèques nécessaires pour le modèle GRU
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import seaborn as sns
from keras.models import Sequential
from keras.layers import GRU, Dropout, Dense, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model
import streamlit as st
'''Exemple d'affichage'''
# Fonction pour convertir et nettoyer les colonnes de date
def convert_and_clean_date(df, col_name):
    df[col_name] = pd.to_datetime(df[col_name], errors='coerce', utc=True).dt.tz_convert(None)
    df.dropna(subset=[col_name], inplace=True)

# 1. Fonction pour l'importation des données
def load_files():
    # Chargement du fichier principal via Streamlit
    uploaded_file = st.file_uploader("Choisir le fichier principal (test2.xlsx)", type="xlsx", key="main_file_uploader")

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write("Aperçu des données principales", df.head())
    else:
        st.warning('...')
        return None
    
    # Chargement des fichiers supplémentaires
    excel_paths = {
        'vacances': 'fr-en-calendrier-scolaire.csv',
        'feries': 'Jours_feries.xlsx',
        'inflation': 'Inflation.xlsx'
    }
    
    df_vacances = pd.read_csv(excel_paths['vacances'], sep=';').query('Zones == "Corse"')
    df_feries = pd.read_excel(excel_paths['feries'])
    df_inflation = pd.read_excel(excel_paths['inflation'])

        # Conversion et nettoyage des colonnes de date
    for df_temp, col in zip([df_vacances, df_feries], ['Date de début', 'Date']):
        convert_and_clean_date(df_temp, col)

    # Normalisation des noms de colonnes et types de données
    df_inflation.rename(columns={'Année': 'Annee'}, inplace=True)
    df_inflation['Annee'] = df_inflation['Annee'].astype(int)

    return df, df_vacances, df_feries, df_inflation

def create_training_dataframe(df, df_vacances, df_feries, df_inflation):
    # Initialisation du dictionnaire pour stocker les DataFrame par catégorie
    dfs_by_category = {}
    tire_categories = df['Catégorie de Pneu'].unique()

    for category in tire_categories:
        sub_df = df[df['Catégorie de Pneu'] == category]
        sub_df = sub_df.groupby('Date de la réception')['Poids Net Collecté'].sum().reset_index()
        sub_df['Date de la réception'] = pd.to_datetime(sub_df['Date de la réception'])
        sub_df.set_index('Date de la réception', inplace=True)
        sub_df = sub_df.resample('D').asfreq().fillna(0).reset_index()

        # Ajouter des colonnes temporelles et d'autres métadonnées
        sub_df['Jours_Sem'] = sub_df['Date de la réception'].dt.dayofweek
        sub_df['Jours_Mois'] = sub_df['Date de la réception'].dt.day
        sub_df['Mois'] = sub_df['Date de la réception'].dt.month
        sub_df['Annee'] = sub_df['Date de la réception'].dt.year

        # Fusion avec df_inflation pour chaque année décalée
        for shift_year in range(6):
            df_temp = df_inflation.copy()
            column_name = f'Inflation_Annee_N-{shift_year}'

            # Ici, je crée une nouvelle colonne qui représente les années décalées
            sub_df[f'Annee_N-{shift_year}'] = sub_df['Annee'] - shift_year

            # Fusion avec le DataFrame principal
            sub_df = pd.merge(sub_df, df_temp, left_on=f'Annee_N-{shift_year}', right_on='Annee', how='left')

            # Renommer la colonne nouvellement ajoutée
            sub_df.rename(columns={'Taux inflation': column_name}, inplace=True)

            # Supprimer les colonnes temporaires et inutiles
            sub_df.drop(columns=[f'Annee_N-{shift_year}', 'Annee_y'], inplace=True)
            sub_df.rename(columns={'Annee_x': 'Annee'}, inplace=True)
            # Marquage des jours fériés et vacances
        sub_df['Vacances'] = 0
        sub_df['Feries'] = 0
        sub_df['Description_Vacances'] = 'NC'

        # Conversion des dates en datetime.date pour une comparaison valide
        for i, row in df_vacances.iterrows():
            date_debut = pd.Timestamp(row['Date de début']).date()
            date_fin = pd.Timestamp(row['Date de fin']).date()
            mask = (sub_df['Date de la réception'].dt.date >= date_debut) & (sub_df['Date de la réception'].dt.date <= date_fin)
            sub_df.loc[mask, 'Vacances'] = 1
            sub_df.loc[mask, 'Description_Vacances'] = row['Description']

        for i, row in df_feries.iterrows():
            mask = sub_df['Date de la réception'].dt.date == pd.Timestamp(row['Date']).date()

            sub_df.loc[mask, 'Feries'] = 1

        # Ordre des colonnes
        col_order = ['Date de la réception', 'Jours_Sem', 'Jours_Mois', 'Mois', 'Annee', 'Poids Net Collecté'] + \
                    [f'Inflation_Annee_N-{i}' for i in range(6)] + ['Vacances', 'Feries', 'Description_Vacances']
        sub_df = sub_df[col_order]

        # Ajouter à la collection de DataFrames
        dfs_by_category[category] = sub_df
    #print(dfs_by_category[tire_categories[0]].head())
    return dfs_by_category, tire_categories


    
def create_prediction_dataframe(df, df_vacances, df_feries, df_inflation, future_dates_range):
    # 1. Plage de Dates
    future_dates = pd.date_range(start=future_dates_range[0], end=future_dates_range[1], freq='D')
    future_df = pd.DataFrame({'Date de la réception': future_dates})

    # 2. Colonnes Temporelles
    future_df['Jours_Sem'] = future_df['Date de la réception'].dt.dayofweek
    future_df['Jours_Mois'] = future_df['Date de la réception'].dt.day
    future_df['Mois'] = future_df['Date de la réception'].dt.month
    future_df['Annee'] = future_df['Date de la réception'].dt.year

    # 3. Données Externes (Inflation)
    for i in range(0, 6):
        temp_df = df_inflation.copy()
        temp_df['Annee'] = temp_df['Annee'] + i
        temp_df.rename(columns={'Taux inflation': f'Inflation_Annee_N-{i}'}, inplace=True)
        future_df = pd.merge(future_df, temp_df, how='left', on='Annee', suffixes=('', f'_N-{i}'))

    # Marquage des jours fériés et vacances
    future_df['Vacances'] = 0
    future_df['Feries'] = 0
    future_df['Description_Vacances'] = 'NC'

    for i, row in df_vacances.iterrows():
        date_debut = pd.Timestamp(row['Date de début']).date()
        date_fin = pd.Timestamp(row['Date de fin']).date()
        mask = (future_df['Date de la réception'].dt.date >= date_debut) & (future_df['Date de la réception'].dt.date <= date_fin)
        future_df.loc[mask, 'Vacances'] = 1
        future_df.loc[mask, 'Description_Vacances'] = row['Description']

    for i, row in df_feries.iterrows():
        mask = future_df['Date de la réception'].dt.date == pd.Timestamp(row['Date']).date()
        future_df.loc[mask, 'Feries'] = 1

    # Ordre des colonnes
    col_order = ['Date de la réception', 'Jours_Sem', 'Jours_Mois', 'Mois', 'Annee'] + \
                [f'Inflation_Annee_N-{i}' for i in range(6)] + ['Vacances', 'Feries', 'Description_Vacances']

    future_df = future_df[col_order]
    #print(future_df)
    return future_df

# Fonction pour préparer les données pour le modèle LSTM
def prepare_data_for_lstm(df, feature_cols, target_col=None, sequence_length=14):
    data = df[feature_cols].values
    
    # Si target_col est fourni, utilisez-le pour créer y
    if target_col is not None:
        target = df[target_col].values
    else:
        target = np.zeros(len(data))  # des zéros ou n'importe quelle valeur de remplissage
    
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(target[i+sequence_length] if target_col else 0)
    
    X = np.array(X)
    y = np.array(y)

    return X, y


# Fonction pour visualiser les résultats
def plot_predictions(category, y_test, y_pred):
    plt.figure(figsize=(15, 6))
    plt.plot(y_test, label='Valeurs réelles', color='black')
    plt.plot(y_pred, label='Valeurs prédites', color='orange')
    plt.title(f'Prédictions de tonnage pour la catégorie {category}')
    plt.xlabel('Temps')
    plt.ylabel('Tonnage')
    plt.legend()
    st.pyplot(plt)

def group_data_by_month(df):
    df_monthly = df.resample('M').sum()
    return df_monthly

def group_data_by_week(df):
    df_weekly = df.resample('W').sum()
    return df_weekly

#def plot_predictions_mens(category, y_test, y_pred):
def plot_predictions_mens(category, y_test, y_pred, dfs_by_category):
    # 1. Convertir les données en DataFrame
    df_real = pd.DataFrame({'Real': y_test.flatten()})
    df_pred = pd.DataFrame({'Pred': y_pred.flatten()})
    
    # 2. Ajouter 'Month' et 'Year'
    df_real['Month'] = dfs_by_category[category]['Mois'].values[-len(df_real):]
    df_real['Year'] = dfs_by_category[category]['Annee'].values[-len(df_real):]
    df_pred['Month'] = dfs_by_category[category]['Mois'].values[-len(df_pred):]
    df_pred['Year'] = dfs_by_category[category]['Annee'].values[-len(df_pred):]

    # 3. Grouper par mois et année
    df_real_monthly = df_real.groupby(['Year', 'Month']).sum().reset_index()
    df_pred_monthly = df_pred.groupby(['Year', 'Month']).sum().reset_index()

    # 4. Trier par date
    df_real_monthly['Date'] = pd.to_datetime(df_real_monthly[['Year', 'Month']].assign(DAY=1))
    df_pred_monthly['Date'] = pd.to_datetime(df_pred_monthly[['Year', 'Month']].assign(DAY=1))
    df_real_monthly.sort_values('Date', inplace=True)
    df_pred_monthly.sort_values('Date', inplace=True)

    # 5. Générer le graphique
    plt.figure(figsize=(15, 6))
    plt.plot(df_real_monthly['Date'], df_real_monthly['Real'], label='Valeurs réelles', linestyle='-')
    plt.plot(df_pred_monthly['Date'], df_pred_monthly['Pred'], label='Valeurs prédites', linestyle='--')
    plt.title(f'Prédictions de tonnages mensuelles pour la catégorie {category}')
    plt.xlabel('Date')
    plt.ylabel('Tonnage')
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)

   # 6. Créer un tableau récapitulatif
    recap_table = pd.concat([df_real_monthly.set_index('Date')['Real'], df_pred_monthly.set_index('Date')['Pred']], axis=1)
    recap_table.columns = ['Valeurs réelles', 'Valeurs prédites']

    # 6.1 Arrondir les valeurs à un chiffre après la virgule
    # Convertir en float si nécessaire
    recap_table['Valeurs réelles'] = recap_table['Valeurs réelles'].astype(float)
    recap_table['Valeurs prédites'] = recap_table['Valeurs prédites'].astype(float)

    # Appliquer l'arrondi
    recap_table = recap_table.round(1)

    
    # 6.2 Formater la colonne Date
    recap_table.index = recap_table.index.strftime('%m/%Y')

    # 7. Afficher le tableau récapitulatif sur Streamlit
    st.table(recap_table)

        # 7.1 Calculer et afficher les totaux
    total_real = round(recap_table['Valeurs réelles'].sum(), 1)
    total_pred = round(recap_table['Valeurs prédites'].sum(), 1)
    st.write(f'Total réel : {total_real:.1f}')
    st.write(f'Total prédit : {total_pred:.1f}')

    # 7.2 Calculer et afficher le taux d'erreur global
    taux_erreur_global = abs(total_real - total_pred) / total_real * 100
    st.write(f'Taux d\'erreur global entre les totaux : {round(taux_erreur_global, 2):.1f}%')

def plot_predictions_mens_futur(y_test, y_pred, time_labels, title):
    # 1. Conversion en DataFrame
    df_real = pd.DataFrame({'Real': y_test.flatten()})
    df_pred = pd.DataFrame({'Pred': y_pred.flatten()})
    
    # 2. Ajout des labels temporels
    df_real['Time'] = pd.to_datetime(time_labels[-len(df_real):])
    df_pred['Time'] = pd.to_datetime(time_labels[-len(df_pred):])

    # 3. Grouper par mois
    df_real['Month'] = df_real['Time'].dt.to_period('M')
    df_pred['Month'] = df_pred['Time'].dt.to_period('M')

    df_real_monthly = df_real.groupby('Month')['Real'].sum().reset_index()
    df_pred_monthly = df_pred.groupby('Month')['Pred'].sum().reset_index()

    # Exclure le premier mois
    df_real_monthly = df_real_monthly.iloc[1:]
    df_pred_monthly = df_pred_monthly.iloc[1:]

    # Générer les histogrammes
    plt.figure(figsize=(15, 6))
    plt.bar(df_real_monthly['Month'].astype(str), df_real_monthly['Real'], alpha=0.6, label='Valeurs réelles')
    plt.bar(df_pred_monthly['Month'].astype(str), df_pred_monthly['Pred'], alpha=0.6, color='blue', label='Valeurs prédites')

    # Ajouter les valeurs sur les barres
    for i, value in enumerate(df_real_monthly['Real']):
        plt.text(i, value, str(round(value, 1)), ha='center')
    for i, value in enumerate(df_pred_monthly['Pred']):
        plt.text(i, value, str(round(value, 1)), ha='center')

   

    z = np.polyfit(range(len(df_real_monthly['Real'])), df_real_monthly['Real'], 1)
    p = np.poly1d(z)
    plt.plot(df_real_monthly['Month'].astype(str), p(range(len(df_real_monthly['Real']))), "r--")


    plt.title(title)
    plt.xlabel('Mois')
    plt.ylabel('Tonnage')
    plt.legend().remove()
    plt.tight_layout()
    st.pyplot(plt)

@st.cache_resource
def load_my_model():
    model = load_model('modele_AE_V1.h5')
    return model

result = load_files()
if result:
    df, df_vacances, df_feries, df_inflation = result
    dfs_by_category, tire_categories = create_training_dataframe(df, df_vacances, df_feries, df_inflation)
    
    # Charger le modèle sauvegardé
    #model_charge = load_model("modele_AE_V1.h5")
    model_charge = load_my_model()

    feature_cols = ['Jours_Sem', 'Jours_Mois', 'Mois', 'Annee', 'Inflation_Annee_N-0', 'Inflation_Annee_N-1', 'Inflation_Annee_N-2', 'Inflation_Annee_N-3', 'Inflation_Annee_N-4', 'Inflation_Annee_N-5', 'Vacances', 'Feries']
    target_col = 'Poids Net Collecté'
    sequence_length = 14
    n_features = len(feature_cols)

    df = dfs_by_category[tire_categories[0]]

    X, y = prepare_data_for_lstm(df, feature_cols, target_col, sequence_length)

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Faire des prédictions
    y_pred = model_charge.predict(X_test)
    # Visualisation des prédictions
    #plot_predictions(tire_categories[0], y_test, y_pred)
    plot_predictions_mens(tire_categories[0], y_test, y_pred,dfs_by_category)
    # Afficher le tableau récapitulatif dans Streamlit
    #display_summary_table(tire_categories[0], y_test, y_pred)
    # Ensuite, vous pouvez continuer avec le reste de votre code.
    future_dates_range = ('2023-08-01', '2023-12-31')
    future_df = create_prediction_dataframe(df, df_vacances, df_feries, df_inflation, future_dates_range)
    # Préparez future_df pour la prédiction
    X_future, _ = prepare_data_for_lstm(future_df, feature_cols, target_col=None, sequence_length=sequence_length)

    # Utiliser le modèle pour faire des prédictions
    y_future_pred = model_charge.predict(X_future)
    # Visualisation des prédictions
    #plot_predictions_mens(tire_categories[0],y_future_pred, y_future_pred, future_df)

    # Utilisation de la fonction modifiée avec future_df
    time_labels = pd.date_range(start='2023-08-01', end='2023-12-31', freq='D')
    y_future_pred = model_charge.predict(X_future)  # Supposé que X_future a été bien préparé
    plot_predictions_mens_futur(y_future_pred, y_future_pred, time_labels, 'Prédictions futures de tonnage')
    # Afficher le tableau récapitulatif dans Streamlit
    #display_summary_table(tire_categories[0], y_future_pred, future_df)

else:
    st.warning("Aucun fichier n'a été chargé, impossible de continuer.")

