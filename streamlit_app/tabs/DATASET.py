import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io

import matplotlib.pyplot as plt

title = "Projet Estimation de ventes e-commerce"
sidebar_name = "DATASET"

import zipfile
with zipfile.ZipFile("file.zip","python") as zip_ref:
    zip_ref.extractall("df")


def run():
    @st.cache
    def net_data ():
           
     #suppression des lignes et des colonnes vides
        df = df.dropna(axis = 1, how = 'all')                                               #Suppression des colonnes avec des valeurs manquantes
        df = df.dropna(axis = 0, how = 'all')                                               #Suppression des lignes avec des valeurs manquantes

    #traitement de la colonne "status" 
        df['status'] =  df['status'].fillna(df['status'].mode()[0]) 
        df.status = df.status.replace({"\\N" : "complete"})

    #traitement de la colonne "category_name_1"
        df.category_name_1= df.category_name_1.replace({"\\N" : "Others"})                 #Remplacement dans la colonne category les /N par     category Others
        df['category_name_1'].fillna(value = "Others", inplace = True)                      #Remplacement dans la colonne category les NaN par category Others
        df['sku'].fillna(value = "Others", inplace = True)                                  #Remplacement dans la colonne sku les NaN par category Others

#Remplacement dans la colonne sales_commission_code les NaN par No_code
                                 
        df.sales_commission_code = df.sales_commission_code.replace({"\\N" : "No_code"})    #Remplacement dans la colonne sales_commission_code les /N par No_code
        df.sales_commission_code.fillna(('No_code'), inplace = True)


#traitement de la colonne "Customer Since" et'Customer ID'
        df['Customer Since'].fillna(df['Year']- df['Month'], inplace = True)                #Remplacement dans la colonne Customer Since les NaN par Year - Month
        df['Customer ID']=df['Customer ID'].fillna(0)                                      #Remplacement dans la colonne Customer ID les NaN par 0

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
        
        return df


    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    st.image("https://pirscapital.com/wp-content/themes/pirscapital-v1/assets/images/gif-1-Revised.gif", width = 600)
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")

    st.title(title)
   
    st.markdown("---")

    st.markdown(
        """
       Dans le cadre de la formation de Data Analyste notre équipe **TO Thi Phuong Thao, Annette DUBUS, Camille LARA** a chosi le sujet très   très actuel de nos jours E-commerce. 
       Avec l'explosion des ventes offline nous avons moins de contacte avec le client c'est-à-dire nous avons autant plus besoin creuser plus pour connaître nos clients, leurs habitudes d'achat, les tendances etc.  Nous travaillerons en tant que Data Analyst dans l'une des plus grandes plateformes de commerce électronique au Pakistan. Notre objectif sortir l'information importante pour le conseil d'administration de l'entreprise et les aider à prendre les décisions pértinantes pour l'évolution de l'entreprise.
        """
    )
    
    st.header("***Présentation des données***")
    
    st.markdown(
    "Le data frame proposé est assez complexe, il y a beaucoup d'informations très utiles dans le contexte business et marketing. Par contre les données sont certainement une extraction d’une ERP brute nécessite beaucoup de travail de compréhension et nettoyage, par exemple 44% des valeurs manquentes, les dates, grand total, discount etc.")
    
    st.dataframe(df.head())
    
    st.markdown(
    "Nous avons aperçu beaucoup de valeurs manquantes - 44% : 464 051 lignes  et 5 dernières colonnes sont vides . Nous avons décidé de les supprimer grâce à la méthode dropna. Après la suppression, les lignes et les colonnes vides, nous avons le Data Frame suivant 15 ligne manquantes dans le status, 20 dans sku, 164 dans category_name, 137 175 dans sales commissions codes, 11 dans Customer Since et Customer ID. Pour chaque variable la methode de suppression appropriée a été choisie :  ") 
    
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

    
    st.code(code, language='python')
    

    st.header("""
    
    Les variables analisé:

        """
    ) 

    st.subheader("""
    
    CATEGORY NAME

        """
    ) 
    
    
    st.markdown(
        """
L'entreprise propose les produits des catégories très variés : 
 
         """
    )
            
    fig, ax = plt.subplots()

    plt.title(label='Catégorie par quantité des commandes ', fontsize=24)
    sns.countplot(x=df["category_name_1"], data=df, palette="cividis", order = df.category_name_1.value_counts().index)
    plt.xticks(rotation=90)
    plt.xlabel('Catégory', fontsize=16)
    plt.ylabel('Quantité des commandes', fontsize=16)
    st.pyplot(fig)
    
    st.markdown(
        """    
    
    Au niveau des ventes, nous observons la dominance des certaines catégories par CA Mobiles&Tabletes, Computing, Entraînement et Appliance ou par quantité de ventes les produits de mode pour hommes et femme, Mobiles&Tabletes, produits de beauté et Appliance. Ce constat peut être la base de notre stratégie marketing avec le choix de positionnement sur le marché et les actions des ventes plus particulières par catégorie. Ici, l'analyse de coût et la marge peuvent nous donner plus d'informations.

         """
    )
    
    st.subheader("""
    
    STATUS

        """
    ) 
    
    
    fig, ax = plt.subplots()

    plt.title(label='Status par quantité des commandes ', fontsize=24)
    sns.countplot(x="status", data=df, palette="cividis", order = df.status.value_counts().index)
    plt.xticks(rotation=90)
    plt.xlabel('status', fontsize=16)
    plt.ylabel('Quantité des commandes', fontsize=16)

    st.pyplot(fig)
    
    
    
    st.markdown(
        """
 Nous avons plus de 10 statuts similaire la groupage va faciliter notre analyse :
 
         """
    )
    st.markdown(
        """
**Complété** = Completed (complete, received, cod, paid, closed, exchange) -54%
     """
    )
    st.markdown(
        """
**Annulé** = Canceled (canceled, order_refunded, refund, fraud)  -46%    """ 
    )
    st.markdown(
        """
**En cours** = Pending (payement_review, pending, processing, holded, pending_paypal) -0,03%

        """
    )


        
    st.subheader("""
    
    PAIEMENT METHOD

        """
    ) 
    
    st.subheader("""
    
    DATES

        """
    ) 
    st.subheader("""
    
    CHIFFRES D’AFFAIRES

        """
    ) 
    

   
