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
La date du dernier achat. Notez bien que l’on part du principe qu’une personne qui a acheté récemment chez vous a plus de chances de revenir commander chez nous.: 
            """
    )
    
