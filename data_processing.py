import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import re
from model import clf
from flask import request, jsonify
from model import clf

#clf = load('transaction_category_modele.pkl')
# Fonction pour nettoyer les noms des commerçants
df = pd.read_csv("newData.csv", low_memory=False)
#print(df)

# Fonction pour nettoyer les noms des commerçants
def clean_and_extract_merchant_name(raw_name):
    special_cases = {
        'FACEBK': 'FACEBOOK',
        'LIGNE': 'LWS',
        'EUR.SHEIN.COM': 'SHEIN',
        'CRD*SMMFOLLOWS': 'SMMFOLLOWS',
        'NAME-CHEAP.COM': 'NAMECHEAP'
    }
    
    # Vérifier les cas spéciaux
    upper_name = raw_name.upper()
    for key, value in special_cases.items():
        if key in upper_name:
            return value
    
    # Première tentative : Vérifier le format 'domaine.com', 'www.domaine.com', ou 'domaine.com/bill'
    domain_match = re.search(r'^(?:www\.)?([^\.]+)\.(com|org|io)(?:/bill)?$', raw_name, re.IGNORECASE)
    if domain_match:
        return domain_match.group(1).upper()

    # Deuxième tentative : Extraire jusqu'au premier espace ou caractère spécial
    simple_match = re.match(r'^[^.\s:*]+', raw_name)
    if simple_match:
        return simple_match.group().upper()
    
    # Si aucun motif n'est trouvé, retourner la chaîne originale
    return raw_name.upper()


# Appliquer la fonction de nettoyage aux noms des commerçants
df['cleaned_name'] = df['merchantName'].apply(clean_and_extract_merchant_name)

# Assurez-vous que toutes les valeurs dans 'cleaned_name' sont des chaînes de caractères
df['cleaned_name'] = df['cleaned_name'].astype(str)

# Ajouter les colonnes 'transactionType' et 'transactionTime' à notre sélection
selected_features = ['transactionAmount', 'cardId', 'transactionType', 'transactionTime', 'cleaned_name', 'DescriptionMcc']

# Créer un DataFrame avec les colonnes sélectionnées
df_selected = df[selected_features]

print(df_selected)
print(df_selected['transactionAmount'].dtype)

# Conversion de la colonne 'transactionTime' en format datetime
# Conversion de la colonne 'transactionTime' en format datetime
dates=pd.to_datetime(df_selected['transactionTime']) 
print(dates.dtype)

# Extraction de caractéristiques pertinentes de la date et de l'heure
df_selected['hour_of_day'] = dates.dt.hour
df_selected['day_of_week'] = dates.dt.dayofweek
df_selected['month'] = dates.dt.month


# Suppression de la colonne 'transactionTime' d'origine
df_selected.drop(columns=['transactionTime'], inplace=True)
print(df_selected)

# Utiliser LabelEncoder pour encoder la colonne 'transactionType' en valeurs numériques
le_transactionType = LabelEncoder()
df_selected.loc[:, 'transactionType'] = le_transactionType.fit_transform((df_selected['transactionType']))

# Utiliser OneHotEncoder pour encoder la colonne 'cleaned_name' en variables binaires
ohe = OneHotEncoder()
cleaned_name_encoded = ohe.fit_transform(df_selected[['cleaned_name']])

# Utiliser LabelEncoder pour encoder la colonne 'cardId' en valeurs numériques
le_cardId = LabelEncoder()
df_selected.loc[:, 'cardId'] = le_cardId.fit_transform(df_selected['cardId'])

# Obtenir les noms de caractéristiques après l'encodage
feature_names = ohe.get_feature_names_out(['cleaned_name'])
print(cleaned_name_encoded.shape)
# Afficher les formes des données avant et après l'encodage
print("Shape before encoding:", df_selected[['cleaned_name']].shape)
print("Shape after encoding:", cleaned_name_encoded.shape)

# Obtenez les noms de colonnes pour les nouvelles fonctionnalités
cleaned_name_encoded_df = pd.DataFrame(cleaned_name_encoded.toarray(), columns=feature_names)


def predict_category():
    transaction_data = request.json
    print('request.json',request.json)
    
    # Convertir les données de transaction en DataFrame
    transaction_df = pd.DataFrame(transaction_data, index=[0])
    print(transaction_df)
    
    # Appliquer les mêmes transformations que celles effectuées sur les données d'entraînement
    transaction_df['cleaned_name'] = transaction_df['cleaned_name'].apply(clean_and_extract_merchant_name)
    print(transaction_df['cleaned_name'])
    transaction_df['transactionTime'] = pd.to_datetime(transaction_df['transactionTime'])
    transaction_df['hour_of_day'] = transaction_df['transactionTime'].dt.hour
    transaction_df['day_of_week'] = transaction_df['transactionTime'].dt.dayofweek
    transaction_df['month'] = transaction_df['transactionTime'].dt.month
    
    # Utiliser LabelEncoder pour encoder la colonne 'transactionType' en valeurs numériques
    le_transactionType = LabelEncoder()
    transaction_df['transactionType'] = le_transactionType.fit_transform(transaction_df['transactionType'])

    # Utiliser LabelEncoder pour encoder la colonne 'cardId' en valeurs numériques
    le_cardId = LabelEncoder()
    transaction_df['cardId'] = le_cardId.fit_transform(transaction_df['cardId'])

    # Utiliser OneHotEncoder pour encoder la colonne 'cleaned_name' en variables binaires
    ohe = OneHotEncoder()
     
    
     
    cleaned_name_encoded_df = pd.DataFrame(cleaned_name_encoded.toarray(), columns=feature_names)
     
    # Concaténer toutes les caractéristiques nécessaires pour la prédiction
    transaction_final = pd.concat([transaction_df[['transactionAmount', 'cardId', 'transactionType', 'hour_of_day', 'day_of_week', 'month']], cleaned_name_encoded_df], axis=1)

    # Faire la prédiction avec le modèle
    predicted_category = clf.predict(transaction_final)
    # Retourner la catégorie prédite dans la réponse JSON
    return predicted_category

 
