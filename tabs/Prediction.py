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


title = "Prédiction/ Timeseries"
sidebar_name = "Prediction"


def run():
    st.title(title)
    
    st.markdown(
        """
        Dans le commerce, les valeurs les plus intéressantes sont les ventes et les chiffres d’affaires. Le marché actuel rencontre beaucoup de problématiques d'approvisionnement de stock. Dans cette optique, nous avons décidé d'essayer de prédire les ventes pour assurer le meilleur approvisionnement de notre stock.
Nous avons créé le DataFrame des ventes, qui représente la série temporelle sur 26 mois.
        """
    )
    @st.cache (allow_output_mutation=True)
    def prev_data ():
        df_2 = pd.read_csv("https://www.dropbox.com/s/vtlr8jubvopw4v9/ecom_df_2.csv?dl=1")   
        df_2["created_at"] = pd.to_datetime(df_2["created_at"], format='ISO8601')
        df_gr = df_2['qty_ordered'].groupby(df_2['created_at']).agg('sum')
        df_prev = pd.DataFrame(list(df_gr.items()), columns=['created_at', 'qty_ordered'])
        df_prev['created_at'] = pd.to_datetime(df_prev['created_at'], format='ISO8601')
        
       
        return df_prev

    df_prev = prev_data()
    
    if st.checkbox('Afficher mon DF'):
        st.dataframe(df_prev)
    
    
    fig = plt.figure(figsize=(30, 20))
    sertim = px.line(df_prev, x = "created_at", y = "qty_ordered" )
    plt.xticks(rotation=90)
    st.plotly_chart(sertim)
    
    st.markdown(
        """    
    Nous pouvons observer les pics de ventes ainsi que la légère tendance à l'augmentation dans le temps. Mais, du fait des variations saisonnières de la série, cette augmentation est difficile à quantifier.
Pour analyse des séries temporelles, on va s'intéresser aux notions trend, no trend, saisonnalité  :""")
    
    from statsmodels.tsa.tsatools import detrend
    notrend = detrend(df_prev['qty_ordered'])#La fonction detrend retourne la tendance. On l’obtient en réalisant une régression linéaire de Y sur le temps t.
    df_prev["notrend"] = notrend
    df_prev["trend"] = df_prev['qty_ordered'] - notrend
    df_prev.tail()
    
    fig = plt.figure(figsize=(30, 20))
    
    tr= px.line(df_prev, x= "created_at", y=["qty_ordered", "notrend", "trend"]);
    st.plotly_chart(tr)

     
    st.markdown(
        """    
  Pour analyse des séries temporelles, on va s'intéresser aux notions trend, no trend, saisonnalité  :""")
    
        
    
    from statsmodels.tsa.seasonal import seasonal_decompose

    #si erreur changer period par freq:
    res = seasonal_decompose(df_prev['qty_ordered'].values.ravel(), period=30, two_sided=False)  #extra la composante saisonnière
    df_prev["season"] = res.seasonal
    df_prev["trendsea"] = res.trend
    
    fig = plt.figure(figsize=(30, 20))
    saison = px.line(df_prev, x= "created_at", y=['qty_ordered', "season", "trendsea"]);
    
    st.plotly_chart(saison)
    
     
    st.markdown(
        """    
   Pour s'assurer de la stationnarité de la série différenciée, il faut faire un test statistique, nous utilisons le plus commun le test augmenté de Dickey-Fuller : """)
    
    df_prev2=df_prev
    df_prev2['qty_ordered'] = np.log(df_prev2.qty_ordered)
    df_prev2['qty_ordered']=df_prev2['qty_ordered'].ewm(alpha=0.5).mean()
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(df_prev2.qty_ordered)
    st.write('ADF Statistic: %f' % result[0])
    st.write('p-value: %f' % result[1])
    st.write('Critical Values:')
    for key, value in result[4].items():
        st.write('\t%s: %.3f' % (key, value))
        
    st.markdown(
        """    
ADF statistics est -4.18 , p -value 0.000697 < 0.05 on peut donc légitimement rejeter l'hypothèse H0. Nous avons stationnarité notre série. """)

    
    df_prev.index = pd.date_range(start='2016-07-01',periods=788, freq='D')
    
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig = plt.figure(figsize=(20,15))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(df_prev2['qty_ordered'], lags=32, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(df_prev2['qty_ordered'], lags=32, ax=ax2);
    st.pyplot(fig)
    
    st.header("""Prédiction avec ARIMA""")
    from statsmodels.tsa.arima.model import ARIMA
    
    st.markdown(
        """    
Afin de choisir les meilleurs paramètres, on utilise les autocorrélogrammes simples et partiel ainsi que fonction autorima. """)
    
    ar_model = ARIMA(df_prev2.qty_ordered, order=(3,0,3))
    ar_model_fit = ar_model.fit()
    st.write(ar_model_fit.summary())
    
    
    train_data = df_prev2['qty_ordered'].iloc[:630]
    test_data = df_prev2['qty_ordered'].iloc[630:]
    start = len(train_data)  #predictions
    end = len(train_data) + len(test_data) - 1
  
    df_prev2['ar_pred'] = ar_model_fit.predict(start, end,
                             typ = 'levels').rename("Predictions")
    
    fig = plt.figure(figsize=(30, 20))
    sertim = px.line(df_prev2, x = "created_at", y = ['qty_ordered', "ar_pred"] )
    plt.legend(['Actual', 'Predicted'])
    plt.xticks(rotation=90)
    st.plotly_chart(sertim)
    
    st.markdown(
        """    
Les résultats semblent corrects. C'est aussi confirmé par le plot diagnostic test : """)


    fig = plt.figure(figsize=(20,15))
    fig = ar_model_fit.plot_diagnostics(figsize=(15, 12))
    st.pyplot(fig)

    st.markdown(
        """    
Essayons de faire les prédiction pour les mois à venir :  """)

    fig, ax = plt.subplots()
    ax = df_prev2.qty_ordered.plot(label='observed', figsize=(20, 15))

    ax = test_data.plot(ax=ax) 
    ax = df_prev2['ar_pred'].plot(ax=ax) 

    forecast = ar_model_fit.predict(start=end, end=end+30, dynamic=True)
    forecast.plot(ax=ax)   
    st.pyplot(fig, ax)
    
  
    
    st.header("""Prediction avec SARIMAX""")
    
    from PIL import Image
    
    image = Image.open("sarimax.png")
    st.image(image, caption='SARIMAX')
