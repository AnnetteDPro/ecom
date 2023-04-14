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
import plotly.express as px


def run():
    df_2 = pd.read_csv("https://www.dropbox.com/s/vtlr8jubvopw4v9/ecom_df_2.csv?dl=1")    
    st.title(title)
    
    st.markdown(
        """
       RFM permet de segmenter sa base clients selon l’intention d’achat et de les cibler efficacement. En appliquant cette méthode, vous allez pouvoir :""")
    
    st.markdown(
        """    
 - Analyser le comportement d’achat""")

    st.markdown(
        """
- Déterminer la valeur des clients"""

)
    st.markdown(
        """

 - Prédire des résultats raisonnables
""")
 
    
  
    st.markdown(
        """
Nous allons ainsi identifier plusieurs typologies de clients (les champions, les clients fidèles, les clients Prometteur, les clients perdus…) et les séparer complètement des clients inactifs. 
Cela est possible grâce à la détermination de quatre quartiles pour la récence, la fréquence et le montant. La segmentation :""")

    st.markdown(
        """    
 - **Champion** : Acheté récemment, achète souvent et dépense le plus"""
      )

    st.markdown(
        """
 - **Clients fidèles** : Dépense beaucoup d'argent. Réactif aux promotions."""
      )
    st.markdown(
        """

 - **Client Prometteur** : Acheteurs récents, mais qui n'ont pas dépensé beaucoup.
    """)
    
    st.markdown(
        """
 - **Client à risque** : n'a pas dépensé beaucoup d'argent, achete pas sovent """

)
    st.markdown(
        """

 - **Client perdu** : Récence, fréquence et scores monétaires les plus bas
""")
    st.subheader("""
        Récence  
        """)
        
      
    st.markdown("""
La date du dernier achat. Notez bien que l’on part du principe qu’une personne qui a acheté récemment chez vous a plus de chances de revenir commander chez nous: 
            """
    )
    
      
    @st.cache(allow_output_mutation=True)
    def rfm ():
    
        from datetime import datetime
 
        df_2["created_at"]= pd.to_datetime(df_2["created_at"]) #transformation en type date
        reference_date= pd.to_datetime('30/8/2018')            #on va partir sur analyse à la fin de périod
        RFM_recence = df_2.groupby(by = 'customer_id', as_index=False)["created_at"].max() #groupage par customer et max date
        RFM_recence.columns = ['customer_id', 'max_Date'] #creation DataFrame par ID avec la date du dernier achat
        RFM_recence['Recence'] = RFM_recence['max_Date'].apply(lambda row: (reference_date - row).days)  #calcule de recence 
        RFM_recence.drop('max_Date', inplace =True, axis = 1)
        qty_cde = df_2.groupby('customer_id')['increment_id'].agg('count').sort_values(ascending=False)  #calcule de nomre d'achat par client

        RFM_freq = pd.DataFrame(list(qty_cde.items()), columns=['customer_id', 'Frequance'])
        RFM_mont = df_2.groupby(by = 'customer_id', as_index=False)["total_price_wt_disc"].sum()
        RFM_mont = RFM_mont.rename(columns = {'total_price_wt_disc':'Montant'})
        RFM_fin = RFM_recence.merge(RFM_freq,  on  = 'customer_id')
        RFM = RFM_fin.merge(RFM_mont,  on  = 'customer_id')
        RFM['F_score'] = pd.cut(RFM['Frequance'], [0, 100, 250, 600,  np.inf], labels=['1', '2','3', '4'])
        RFM['M_score'] = pd.qcut(RFM['Montant'], q=4,  labels=['1', '2', '3', '4'])
        RFM['R_score'] = pd.qcut(RFM['Recence'], q=4,  labels=['4', '3', '2','1'])
        RFM['RFM_score'] = RFM['R_score'].astype('int') + RFM['F_score'].astype('int') + RFM['M_score'].astype('int')
        RFM.loc[(RFM['RFM_score'] == 12.0), 'Segment'] = 'Champions'
        RFM.loc[(RFM['RFM_score'] >= 9) & (RFM['RFM_score'] <= 11), 'Segment'] = 'Clients fidèles'
        RFM.loc[(RFM['RFM_score'] >= 6) & (RFM['RFM_score'] <= 8), 'Segment'] = 'Clients prometteurs'
        RFM.loc[(RFM['RFM_score'] >= 4) & (RFM['RFM_score'] <= 5), 'Segment'] = 'Clients à risque'
        RFM.loc[(RFM['RFM_score'] == 3), 'Segment'] = 'Clients perdus'
        
        return RFM
        
    RFM = rfm()
    seg = RFM['Segment'].value_counts()
