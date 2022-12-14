import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io

import matplotlib.pyplot as plt

title = "Projet Estimation de ventes e-commerce"
sidebar_name = "DATASET"

import ZipFile

with zipfile.ZipFile("Pakistan Largest Ecommerce Dataset.csv.zip", mode="r") as archive:
     df = archive.getinfo("Pakistan Largest Ecommerce Dataset.csv")


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
       Dans le cadre de la formation de Data Analyste notre ??quipe **TO Thi Phuong Thao, Annette DUBUS, Camille LARA** a chosi le sujet tr??s   tr??s actuel de nos jours E-commerce. 
       Avec l'explosion des ventes offline nous avons moins de contacte avec le client c'est-??-dire nous avons autant plus besoin creuser plus pour conna??tre nos clients, leurs habitudes d'achat, les tendances etc.  Nous travaillerons en tant que Data Analyst dans l'une des plus grandes plateformes de commerce ??lectronique au Pakistan. Notre objectif sortir l'information importante pour le conseil d'administration de l'entreprise et les aider ?? prendre les d??cisions p??rtinantes pour l'??volution de l'entreprise.
        """
    )
    
    st.header("***Pr??sentation des donn??es***")
    
    st.markdown(
    "Le data frame propos?? est assez complexe, il y a beaucoup d'informations tr??s utiles dans le contexte business et marketing. Par contre les donn??es sont certainement une extraction d???une ERP brute n??cessite beaucoup de travail de compr??hension et nettoyage, par exemple 44% des valeurs manquentes, les dates, grand total, discount etc.")
    
    st.dataframe(df.head())
    
    st.markdown(
    "Nous avons aper??u beaucoup de valeurs manquantes - 44% : 464 051 lignes  et 5 derni??res colonnes sont vides . Nous avons d??cid?? de les supprimer gr??ce ?? la m??thode dropna. Apr??s la suppression, les lignes et les colonnes vides, nous avons le Data Frame suivant 15 ligne manquantes dans le status, 20 dans sku, 164 dans category_name, 137 175 dans sales commissions codes, 11 dans Customer Since et Customer ID. Pour chaque variable la methode de suppression appropri??e a ??t?? choisie :  ") 
    
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
    
    Les variables analis??:

        """
    ) 

    st.subheader("""
    
    CATEGORY NAME

        """
    ) 
    
    
    st.markdown(
        """
L'entreprise propose les produits des cat??gories tr??s vari??s : 
 
         """
    )
            
    fig, ax = plt.subplots()

    plt.title(label='Cat??gorie par quantit?? des commandes ', fontsize=24)
    sns.countplot(x=df["category_name_1"], data=df, palette="cividis", order = df.category_name_1.value_counts().index)
    plt.xticks(rotation=90)
    plt.xlabel('Cat??gory', fontsize=16)
    plt.ylabel('Quantit?? des commandes', fontsize=16)
    st.pyplot(fig)
    
    st.markdown(
        """    
    
    Au niveau des ventes, nous observons la dominance des certaines cat??gories par CA Mobiles&Tabletes, Computing, Entra??nement et Appliance ou par quantit?? de ventes les produits de mode pour hommes et femme, Mobiles&Tabletes, produits de beaut?? et Appliance. Ce constat peut ??tre la base de notre strat??gie marketing avec le choix de positionnement sur le march?? et les actions des ventes plus particuli??res par cat??gorie. Ici, l'analyse de co??t et la marge peuvent nous donner plus d'informations.

         """
    )
    
    st.subheader("""
    
    STATUS

        """
    ) 
    
    
    fig, ax = plt.subplots()

    plt.title(label='Status par quantit?? des commandes ', fontsize=24)
    sns.countplot(x="status", data=df, palette="cividis", order = df.status.value_counts().index)
    plt.xticks(rotation=90)
    plt.xlabel('status', fontsize=16)
    plt.ylabel('Quantit?? des commandes', fontsize=16)

    st.pyplot(fig)
    
    
    
    st.markdown(
        """
 Nous avons plus de 10 statuts similaire la groupage va faciliter notre analyse :
 
         """
    )
    st.markdown(
        """
**Compl??t??** = Completed (complete, received, cod, paid, closed, exchange) -54%
     """
    )
    st.markdown(
        """
**Annul??** = Canceled (canceled, order_refunded, refund, fraud)  -46%    """ 
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
    
    CHIFFRES D???AFFAIRES

        """
    ) 
    

   
