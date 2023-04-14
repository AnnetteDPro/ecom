import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

title = "DataViz"
sidebar_name = "DataViz"

def run():
    
    st.title(title)
    
    st.markdown(
        """
        La première étape nous a permis de mieux comprendre les données.
Pour le moment, cela reste des chiffres ou des informations brutes. La deuxième étape va nous permettre d'approfondir et communiquer plus facilement les résultats en les transformant en objets visuels : points, barres, courbes, cartographies…"""

)

    st.subheader("""
        L'analyse des ventes  
        """)
    

    df_2 = pd.read_csv("https://www.dropbox.com/s/vtlr8jubvopw4v9/ecom_df_2.csv?dl=1")
  
    df_2["created_at"]=pd.to_datetime(df_2["created_at"])

        
      
    st.markdown("""
      Ce DataFrame représente les données de vente d'une plateforme E-commerce. Dans le contexte de ventes, les variables qui peuvent nous être utiles sont : 
            """
    )
    st.markdown("""  
 - La quantité de commandes, 
     """
    )
    st.markdown("""  
 - Le chiffre d’affaires - nous avons effectué la conversion en USD,
"""
    )
    st.markdown("""      
 - La quantité de produits achetés . 
   """
    ) 
    df_2["Year"].value_counts()
    
    year = df_2['total_price_wt_disc'].groupby(df_2["created_at"].dt.year).agg('sum').sort_values(ascending=False) #analyse de CA par l'année
    df_2["Year"].value_counts()
    year2 = df_2['qty_ordered'].groupby(df_2["created_at"].dt.year).agg('sum').sort_values(ascending=False)
    if st.button("Chiffre d'affaires"):
        
        fig, ax = plt.subplots()
        sns.barplot(x=year.index, y=year.values, data= year, palette='Blues_d')
        plt.title(label='CA par année  ', fontsize=20);
        plt.xlabel('Year', fontsize=12)
        plt.ylabel("Chiffre d'affaires", fontsize=12)
        labels = [2016, 2018, 2017]
        ax.set_xticklabels(labels);
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        sns.barplot(x=year2.index, y=year2.values,  palette='Blues_d')
        plt.title(label='QTY par année  ', fontsize=20)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel("QTY", fontsize=12);
    
        st.pyplot(fig)
        

    
    st.markdown("""
        On remarque l'évolution des ventes en 2017 qui continue en 2018 car en 8 mois les chiffres d'affaires en 2018 ont déjà presque le même niveau que sur 12 mois en 2017. La comparaison des chiffre d'affaire et quantité de ventes en 2018 nous illustrent qu'un travail sur la marge ou repositionnement a été effectué cette année. 

        """
    )
    
    st.subheader("""
    
    L'analyse des ventes par catégories:

        """
    ) 
        
    st.markdown("""
        L'entreprise propose les produits très variés qui sont regroupés par catégories, il est très utile d'analyser les ventes par catègories pour se projeter par exemple sur le repositionement.   

        """
    )
        
    cat =df_2.groupby('category_name')
    hpc= cat['price'].agg(np.sum)
    bsc= cat['qty_ordered'].agg(np.sum)
    bca = cat['total_price_wt_disc'].agg(np.sum)
    top_qty = bsc.sort_values(ascending=False).head(6)
    top_ca = bca.sort_values(ascending=False).head(6)
    
    fig, ax = plt.subplots()

    plt.pie(x=top_qty.values, labels = top_qty.index, 
           explode = [0.2, 0, 0, 0, 0, 0],
           autopct = lambda x: str(round(x, 2)) + '%',
           pctdistance = 0.7, labeldistance = 1.1,
           shadow = True)
    plt.title(label='Top 6 des ventes par catégorie ', fontsize=20)
           
    plt.legend(loc="upper left", bbox_to_anchor=(0.95, 0.8));
    
    st.pyplot(fig)
    
    fig, ax = plt.subplots()

    plt.pie(x=top_ca.values, labels = top_ca.index, 
        explode = [0.2, 0, 0, 0, 0, 0],
           autopct = lambda x: str(round(x, 2)) + '%',
           pctdistance = 0.7, labeldistance = 1.05,
           shadow = True)
    plt.title(label="Top 6 catégorie par chiffres d'affaires", fontsize=20)
           
    plt.legend(loc="upper left", bbox_to_anchor=(0.95, 0.8));
    
    st.pyplot(fig)
    
    st.markdown("""
        La catégorie Mobile et Tablets est nettement en tête de vente par la quantité des commandes et par les chiffres d'affaires. Ensuite, on peux voir le décalage par exemple de la catégorie Men's fashion que est en 2nde place par quantité et en 7ème par chiffre d'affaire. 

        """
    )
    
    
    if st.checkbox('Afficher mon graphique plotly'):
        hist_cat = px.histogram(df_2, x = "category_name", color =  "Year", barmode = "group")
        st.plotly_chart(hist_cat)
    st.subheader("""
La distribution des ventes par catégorie est suivante:
    
         """
    )
    
    df_2['M-Y'] = pd.to_datetime(df_2['M-Y'])
    df_nombre_cde = pd.crosstab(df_2['M-Y'], df_2['category_name'])
    
    import datetime
    

    from bokeh.plotting import figure, show, output_notebook
    from bokeh.io import output_file
 n  from bokeh.models import  LabelSet, ColumnDataSource

    from datetime import datetime
    from bokeh.models import BoxAnnotation
    from bokeh.plotting import figure
    
     
    source = ColumnDataSource(df_nombre_cde)


    p = figure(width=1100,height=400, x_axis_type='datetime',
           title = "Evolution des ventes par mois")

            
     
    p.line(x = df_nombre_cde.index, y=df_nombre_cde['Mobiles & Tablets'], color='blue', line_width=2, legend_label='Mobiles & Tablets')
    p.line(x = df_nombre_cde.index, y=df_nombre_cde['Appliances'], line_width=2, color='red', legend_label='Appliances')
    p.line(x = df_nombre_cde.index, y=df_nombre_cde['Computing'], color='steelblue', line_width=2, legend_label='Computing')
    p.line(x = df_nombre_cde.index, y=df_nombre_cde['Superstore'], line_width=2, color='yellow', legend_label='Superstore')
    p.line(x = df_nombre_cde.index, y=df_nombre_cde['Entertainment'], color='orange', line_width=2, legend_label='Entertainment')
    p.line(x = df_nombre_cde.index, y=df_nombre_cde["Men's Fashion"], line_width=2, color='purple', legend_label="Men's Fashion")
    p.line(x = df_nombre_cde.index, y=df_nombre_cde["Women's Fashion"], color='rosybrown', line_width=2, legend_label="Women's Fashion")
    p.line(x = df_nombre_cde.index, y=df_nombre_cde['Others'], line_width=2, color='lawngreen', legend_label='Others')

  
    p.legend.click_policy = 'hide'

    box_left = pd.to_datetime('2017-10-01-')
    box_right = pd.to_datetime('2017-12-01')


    box = BoxAnnotation(left=box_left, right=box_right,
                    line_width=1, line_color='black', line_dash='dashed',
                    fill_alpha=0.2, fill_color='orange')

    p.add_layout(box)

    st.bokeh_chart(p, use_container_width=True)
    
    
    st.markdown("""
        Nous pouvons voir le pic en Novembre pour presque toutes les catégories et une petite augmentation en Mars, Avril, Mai pour certaines catégories. Regardons les ventes par jour et mois :  

        """
    )
        
    if st.button("Ventes par jour"): 
        fig, ax = plt.subplots()
        sns.countplot(x=df_2["created_at"].dt.day); 
        plt.title("Nombre de ventes par jour", fontsize=20)
        plt.xlabel('Ventes', fontsize=14)
        plt.ylabel('Jour', fontsize=10);
        st.pyplot(fig)

    else:
        
        fig, ax = plt.subplots()
        sns.countplot(x=df_2["created_at"].dt.month); 
        plt.title("Nombre de ventes par mois", fontsize=20)
        plt.xlabel('Ventes', fontsize=14)
        plt.ylabel('Mois', fontsize=14);
        st.pyplot(fig)

    
    
    st.markdown("""
    Le pic des ventes est bien en Novembre, ce qui peut correspondre aux achats du black friday du 25 novembre. 
    Dans le mois, les ventes augmentent à la fin du mois, ça peut aider à l'organisation de travail des magasin.
    Par exemple, on peut embaucher des intérimaire dans la période plus intense (très pratiqué par AMAZON) ou mettre en place des promotions au début de mois ou les mois creux. 
L’analyse peut être approfondi avec la projection des ventes sur les catégories et les produits concrets:
    """)
    
    most_sold = pd.DataFrame(df_2['sku'].value_counts())
    top_10_most_sold = most_sold[0:10]

    st.dataframe(top_10_most_sold)
    

    
    

        
        

    
