# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 22:49:58 2022

@author: dubus
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

from sklearn.metrics import mean_absolute_error, mean_squared_error

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import plotly.express as px


title = "Arima"
sidebar_name = "Arima"


def run():
    st.title(title)
    
    st.markdown(
        """
        Dans le commerce les valeurs les plus intéressantes sont les ventes et les chiffres d’affaires. Le marché actuel rencontre beaucoup de problématiques d'approvisionnement de stock dans cette optique nous avons décidé d'essayer de prédire les ventes pour assurer le meilleur approvisionnement de  notre stock.
Nous avons créé le DataFrame des ventes, qui représente  la série temporelle sur 26 mois  :
.
        """
    )
    @st.cache (allow_output_mutation=True)
    def prev_data ():
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
        df_2=df_2.drop(df_2[df_2.total_price_wt_disc >= 10000].index,axis=0)
        df_gr = df_2['qty_ordered'].groupby(df_2['created_at']).agg('sum')
        df_prev = pd.DataFrame(list(df_gr.items()), columns=['created_at', 'qty_ordered'])
        df_prev['created_at'] = pd.to_datetime(df_prev['created_at'])
        
       
        return df_prev

        df_prev = prev_data()
        
    from statsmodels.tsa.arima.model import ARIMA
    ar_model = ARIMA(df_prev.qty_ordered, order=(3,0,3))
    ar_model_fit = ar_model.fit()
    print(ar_model_fit.summary())
        
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    fig = plt.figure(figsize=(30, 20))
    plt.rcParams["figure.figsize"] = [16,9]
    plt.plot(df_prev.qty_ordered, color='b')
    plt.plot(ar_model_fit.predict(dynamic=False),color='orange')
    plt.legend(['Actual', 'Predicted'])
    plt.xticks(rotation=90);
    st.pyplot(fig)

    fig = plt.figure(figsize=(30, 20))
    ar_model_fit.plot_diagnostics(figsize=(15, 12))
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    ax = df_prev.qty_ordered.plot(label='observed', figsize=(20, 15))

    ax = test_data.plot(ax=ax) 
    ax = df_prev['ar_pred'].plot(ax=ax) 

    forecast = ar_model_fit.predict(start=end, end=end+157, dynamic=True)
    forecast.plot(ax=ax)
    st.pyplot(fig)         
