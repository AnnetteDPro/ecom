import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

title = "Projet Estimation de ventes e-commerce"
sidebar_name = "DATASET"

import tabs
from tabs.DATASET import DATASET
from tabs.DataViz import DataViz
from tabs.RFM import RFM
from tabs.Kmeans import Kmeans
from tabs.Prediction import Prediction


def run():
   

    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    st.image("https://pirscapital.com/wp-content/themes/pirscapital-v1/assets/images/gif-1-Revised.gif", width = 600)
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")

    st.title(title)
   
    st.markdown("---")

    st.markdown(
        """
       Dans le cadre de la formation de Data Analyste, notre équipe , composé des personnes suivantes :**TO Thi Phuong Thao, Annette DUBUS, Camille LARA**, a choisi un sujet très actuel, le e-commerce. 
       Avec l'explosion des ventes online, nous avons moins de contacts avec le client et donc, nous avons d'autant plus besoin d'étudier les habitudes d'achat, les tendances et le comportement des clients. Nous travaillerons en tant que Data Analyst dans l'une des plus grandes plateformes de commerce électronique au Pakistan. Notre objectif est de sortir l'information importante pour le conseil d'administration de l'entreprise et les aider à prendre les décisions pertinentes.
        """
    )
    
    st.header("***Présentation des données***")
    
    st.markdown(
    "Le dataframe proposé est assez complexe, il y a beaucoup d'informations très utiles dans le contexte business et marketing. Par contre, les données sont certainement issue d'une extraction brute d'un ERP. Elle nécessite beaucoup de travail, de compréhension et nettoyage. Par exemple, 44% des valeurs sont manquentes, (les dates, grand total, discount etc.)")
    
    st.dataframe(df1.head())
    
    st.markdown(
    "Nous avons vu beaucoup de valeurs manquantes: 44% des 464 051 lignes  et 5 dernières colonnes sont vides. Nous avons décidé de les supprimer grâce à la méthode dropna. Après la suppression des lignes et des colonnes vides, nous avons le Data Frame suivant: 15 lignes manquantes dans status, 20 dans sku, 164 dans category_name, 137 175 dans sales commissions codes, 11 dans Customer Since et Customer ID. Pour chaque variable, la méthode de suppression suivante a été choisie :  ") 
    
    code = '''
#suppression des lignes et des colonnes vides
df = df.dropna(axis = 1, how = 'all')                                               #Suppression des colonnes avec des valeurs manquantes
df = df.dropna(axis = 0, how = 'all')                                               #Suppression des lignes avec des valeurs manquantes

#traitement de la colonne "status" 
df['status'] =  df['status'].fillna(df['status'].mode()[0]) 
df.status = df.status.replace({"\\N" : "complete"})

#traitement de la colonne "category_name_1"
df.category_name_1= df.category_name_1.replace({"\\N" : "Others"})                 #Remplacement dans la colonne category les /N par category Others
df['category_name_1'].fillna(value = "Others", inplace = True)                      #Remplacement dans la colonne category les NaN par category Others
df['sku'].fillna(value = "Others", inplace = True)                                  #Remplacement dans la colonne sku les NaN par category Others

#Remplacement dans la colonne sales_commission_code les NaN par No_code
                                 
df.sales_commission_code = df.sales_commission_code.replace({"\\N" : "No_code"})    #Remplacement dans la colonne sales_commission_code les /N par No_code
df.sales_commission_code.fillna(('No_code'), inplace = True)


#traitement de la colonne "Customer Since" et'Customer ID'
df['Customer Since'].fillna(df['Year']- df['Month'], inplace = True)                #Remplacement dans la colonne Customer Since les NaN par Year - Month
df['Customer ID']=df['Customer ID'].fillna(0)                                       #Remplacement dans la colonne Customer ID les NaN par 0

#renommage des colonnes
new_col_names={" MV ":"mv",
              "category_name_1": "category_name",
              "Customer Since":"customer_since",
              "Customer ID":"customer_id",
              "BI Status":"bi_status"}
df=df.rename(new_col_names,axis=1)
# changement de types conformes
new_col_types={"item_id":"object",
              "customer_id":"object",
              "increment_id":"str",
               'qty_ordered' : 'float'}
df=df.astype(new_col_types)
'''

    if st.checkbox('Afficher mon code'):    
        st.code(code, language='python')
        
    
   