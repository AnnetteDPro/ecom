import streamlit as st
import pandas as pd
import numpy as np


title = "RFM"
sidebar_name = "RFM"

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt


def run():
    
    st.title(title)
    
    st.markdown(
        """
       RFM permet de segmenter sa base clients selon l’intention d’achat et de les cibler efficacement. En appliquant cette méthode, vous allez pouvoir :""")
    
    st.markdown(
        """    
 - Analyser le comportement d’achat""")

    st.markdown(
        """
- éterminer la valeur des clients"""

)
    st.markdown(
        """

 - Prédire des résultats raisonnables
""")

    
    
        
    @st.cache
    def fin_data ():
        df = pd.read_csv("Pakistan Largest Ecommerce Dataset.csv")


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


#suppression des colonnes dates redondantes

        df_2=df.drop(["Working Date"],1)
        


#nouvelles colonnes prix pour calculer le cout total de la ligne avec et sans discount

        df_2["total_price_wo_disc"]=df_2.qty_ordered*df_2.price

        df_2["total_price_wt_disc"]=df_2.qty_ordered*df_2.price-df_2.discount_amount

#harmonisation des status des commandes

        df_2['status'] = df_2.status.replace({'complete': 'completed',

                               'received': 'completed',

                               'cod': 'completed',

                               'paid': 'completed',

                               'closed': 'completed',

                               'exchange': 'canceled',

                               'canceled': 'canceled',

                               'order_refunded': 'canceled',

                               'refund': 'canceled',

                               'fraud': 'canceled',

                               'payment_review': 'canceled',

                               'pending': 'canceled',

                               'processing': 'canceled',

                               'holded': 'canceled',

                               'pending_paypal': 'canceled'})



# selection des commandes uniquement completed

        df_2=df_2[df_2.status=='completed']

        df_2.shape #Nous avons 315 506 lignes des commandes en status completed 

    ## création du dataframe order pour reconstituer les factures
    #fonction pour compter le nombre de références différences
        nb_products = lambda sku: len(np.unique(sku))
    # création des différentes colonnes
        functions_to_apply = {'qty_ordered':'sum',                  #nb de produits total dans la factures
                      'sku': nb_products,                   #nb de référence différentes
                     'discount_amount':['min','max',"sum"], #différent calcul de discount
                     'grand_total':"max",                   # pour vérification
                     'total_price_wo_disc':'sum',           # prix total de la facture sans discount
                     'total_price_wt_disc':'sum',
                     "status":"unique"}           # prix total de la facture avec discount
        order=df_2.groupby("increment_id").agg(functions_to_apply).reset_index()
#renommage des colonnes pour supprimer le double index
        order.columns=["increment_id","total_qty","nb_sku","disc_min","disc_max","disc_sum","gd_total",
              'total_order_wo_disc','total_order_wt_disc',"status"]

#Nous avons 206827 factures unique 

# recuperation des increment id dont le disc min= discount max 
#c.à.d. le discount est commun à toute la facture et non individualisé par ligne
        ls=order[(order.disc_sum!=0)&(order.nb_sku>1)&(order.disc_max==order.disc_min)].increment_id.values
        ls
        df_2=df_2[~df_2.increment_id.isin(ls)] #suppretion des ligne ou le discount est sur la totalité de la facture 
    #suppression de la colonne grand total
        df_2=df_2.drop("grand_total",1)
        df_2.shape
#après la suppretion des lignes avec discount sur la totalité de la factures nous avons le Data frame de 300 810 lignes

        df_2.total_price_wt_disc = df_2.total_price_wt_disc/100 #convertion en USD
        df_2["created_at"]=pd.to_datetime(df_2["created_at"])
        return df_2

    df_2 = fin_data()

    st.markdown(
        """
Nous allons ainsi identifier plusieurs typologies de clients (les champions, les clients fidèles, les clients Prometteur, les clients perdus…) et les séparer complètement des clients inactifs. Cela est possible grâce à la détermination quatre quartiles pour la récence, la fréquence et le montant. La segmentation :""")

    st.markdown(
        """    
 - **Champion** : Acheté récemment, achète souvent et dépense le plus""")

    st.markdown(
        """
 - **Clients fidèles** : Dépense beaucoup d'argent. Réactif aux promotionsDéterminer la valeur des clients"""

)
    st.markdown(
        """

 - **Client Prometteur** : Acheteurs récents, mais qui n'ont pas dépensé beaucoup
""")
    
    st.markdown(
        """
 - **Client à risque** : Dépensé beaucoup d'argent, acheté souvent, mais il y a longtemps"""

)
    st.markdown(
        """

 - **Client perdu** : Récence, fréquence et scores monétaires les plus bas
""")
    st.subheader("""
        Récence  
        """)
        
      
    st.markdown("""
La date du dernier achat. Notez bien que l’on part du principe qu’une personne qui a acheté récemment chez vous a plus de chances de revenir commander chez nous.: 
            """
    )

    
    
    st.subheader("""
        Fréquence 
        """)
        
      
    st.markdown("""
Le nombre d'achats réalisés sur une période donnée. Plus un client achète régulièrement chez vous, plus il y a de chances pour qu’il achète à nouveau. Vous l’avez compris, on analyse ici son niveau de fidélité. 
            """
    )
    
    
    st.subheader("""
        Montant
        """)
        
      
    st.markdown("""
La somme des achats cumulés sur une période donnée. Les gros acheteurs sont le meilleur générateur de revenus pour la plate-forme de commerce électronique. On mesure ici la valeur client. Aux moyennes, les gens dépensent sur la plateforme 151$, mais nous avons des gros (plus de 10 000$)et des petits acheteurs (moins de 100 $):
            """
    )