

title = "K-means"
sidebar_name = "K-means"


import streamlit as st
import pandas as pd
import numpy as np


from PIL import Image
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def run():
    
    st.title(title)
    
    st.markdown("La méthode RFM n’est pas standardisée. "
                "Nous devons donc appliquer un apprentissage non-supervisé. Nous avons sélectionné le modèle de clustering K-means.")
    
   

    st.title("Jeu de données initiales")
    rfm=pd.read_csv('rfm_kmeans.csv',index_col="CustomerID")
    rfm_num=rfm[["Recency","Frequency","Monetary"]]
    scaler=MinMaxScaler()
    Z=pd.DataFrame(scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]]),columns=["Recency","Frequency","Monetary"])
    Zi=pd.DataFrame(scaler.inverse_transform(Z),columns=["Recency","Frequency","Monetary"])
    
    
    st.markdown("Tout d'abord, voici le résultat de classification via la méthode RFM de Guillaume MARTIN ainsi que la représentation en bar plot")
        
    st.dataframe(rfm.head())
    fig1 =plt.figure(figsize=(15,7))
    for index, column in enumerate(rfm.select_dtypes("number").columns):
        plt.subplot(1,7,(index+1))
        plt.boxplot(rfm[column]) 
        plt.xlabel(column)
    st.pyplot(fig1)
        
    st.write("Afin de choisir le meilleur nombre de clusters, nous pouvons utiliser la méthode du coude.")
    
    coude=st.button('Calcul du coude')
    
    if coude:
        st.write("Le coude de la courbe de distorsion indique le nombre de cluster optimal.")
        # Importation de la fonction cdist du package scipy.spatial.distance
        from scipy.spatial.distance import cdist
        
        # Liste des nombre de clusters
        range_n_clusters = [1,2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,19,20]  
        
        # Initialisation de la liste de distorsions
        distorsions = []
        
        # Calcul des distorsions pour les différents modèles
        for n_clusters in range_n_clusters:
            
            # Initialisation d'un cluster ayant un pour nombre de clusters n_clusters
            cluster = KMeans(n_clusters = n_clusters)
            
            # Apprentissage des données suivant le cluster construit ci-dessus
            cluster.fit(Z)
            
            # Ajout de la nouvelle distorsion à la liste des données
            distorsions.append(sum(np.min(cdist(Z, cluster.cluster_centers_, 'euclidean'), axis=1)) / np.size(Z, axis = 0))
        
        
        # Visualisation des distorsions en fonction du nombre de clusters
        fig3=plt.figure()
        plt.plot(range_n_clusters, distorsions, 'gx-')
        plt.xlim([0,20])
        plt.xlabel('Nombre de Clusters K')
        plt.ylabel('Distorsion SSW/(SSW+SSB)')
        plt.title('Méthode du coude affichant le nombre de clusters optimal')
        plt.show()
        st.pyplot(fig3)
    
    
    
    
        
        
    
    st.title("Comparaison K-means, et méthode RFM")
    st.write("Sélection d'un nombre de cluster pour Kmeans")
    nb_clusters=st.slider('Pick a number', 2, 25)
    
    
    
    kmeans = KMeans(n_clusters = nb_clusters)
    kmeans.fit(Z)
    
    # Centroids and labels
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    Centroids_i=pd.DataFrame(scaler.inverse_transform(centroids),columns=["Recency","Frequency","Monetary"]).sort_values(["Recency","Frequency"])
    
    st.title("Comparaison du nombre de clients par catégories")
    
    fig2=plt.figure(constrained_layout=True)
    ax1=fig2.add_subplot(121)
    sns.countplot(y=labels,ax=ax1)
    ax2=fig2.add_subplot(122)
    sns.countplot(y=rfm.Segment,ax=ax2)
    st.pyplot(fig2)
    
    st.title("Comparaison des méthodes selon les différents axes")
    # Nous comparons les résultats de K-means et celui de la méthode
    # Nous comparons les résultats de K-means et celui de la méthode
    #création d'une fig4ure avec la méthode classe objet
    fig4=plt.figure(constrained_layout=True)
    #Analyse par Recence et fréquence
    ax1=fig4.add_subplot(321)
    sns.scatterplot(x=Zi.Recency, y=Zi.Frequency, hue=labels, palette="deep", legend = False,ax=ax1)
    sns.scatterplot(x=Centroids_i.Recency, y=Centroids_i.Frequency, marker="o", legend = False, color="blue", s=50,ax=ax1)
    ax1.set_title("K_means")
    ax2=fig4.add_subplot(322)
    sns.scatterplot(x=rfm.Recency,y=rfm.Frequency, hue=rfm.Segment, palette="deep", legend = False,ax=ax2)
    ax2.set_title("Méthode Guillaume Martin")
    #Analyse par Fréquence et Monnaie
    ax3=fig4.add_subplot(323)
    sns.scatterplot(x=Zi.Frequency,y=Zi.Monetary, hue=labels, palette="deep", legend = False,ax=ax3)
    sns.scatterplot(x=Centroids_i.Frequency,y=Centroids_i.Monetary, marker="o", legend = False, color="blue", s=50,ax=ax3)
    ax3.set_title("K_means")
    ax4=fig4.add_subplot(324)
    sns.scatterplot(x=rfm.Frequency,y=rfm.Monetary, hue=rfm.Segment, palette="deep", legend = False,ax=ax4)
    ax4.set_title("Méthode Guillaume Martin")
    #Analyse par  Monnaie et Recence;
    ax5=fig4.add_subplot(325)
    sns.scatterplot(x=Zi.Recency,y=Zi.Monetary, hue=labels, palette="deep", legend = False,ax=ax5)
    sns.scatterplot(x=Centroids_i.Recency,y=Centroids_i.Monetary, marker="o", color="blue", legend = False, s=50,ax=ax5)
    ax5.set_title("K_means")
    ax6=fig4.add_subplot(326)
    sns.scatterplot(x=rfm.Recency,y=rfm.Monetary, hue=rfm.Segment, palette="deep", legend = False,ax=ax6)
    ax6.set_title("Méthode Guillaume Martin")
    st.pyplot(fig4)
    
    st.title("Comparaison des centres des clusters")
    st.markdown(" Pour informations, dans la méthode Guillaume Martin , les coordonnées des centroids Monetary n'ont pas d'intérêts puisque seules la récence et la fréquence sont prises en compte dans la segmentation. ")

    fct_to_apply={"Recency":"mean",
            "Frequency":"mean",
            "Monetary":"mean"}
    guillaume_martin_centroids=rfm.groupby('Segment').agg(fct_to_apply).sort_values(["Recency","Frequency"])
   
    st.dataframe(guillaume_martin_centroids)
    
    st.dataframe(Centroids_i)
    