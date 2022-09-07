import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

title = "DataViz"
sidebar_name = "DataViz"

def run():
    
    st.title(title)
    
    st.markdown(
        """
        La première étape nous a permis de mieux comprendre les données .
Pour le moment, cela reste des chiffres ou des informations brutes. La deuxième étape va nous permettre d'approfondir et communiquer plus facilement les résultats en les transformant en objets visuels : points, barres, courbes, cartographies…"""

)

    st.subheader("""
        L'analyse des ventes  
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


        
      
    st.markdown("""
      Ce DataFrame représente les données de vente d'une plateforme E-commerce. Dans le contexte de ventes les variableS qui peuvent nous être utiles se sont : 
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
    st.markdown("""  
Tout d’abord, nous avons aperçu la présence de quelques valeurs aberrantes,  3 ventes plus de 10 000$ :
 
        """
    )

    st.dataframe(df_2[df_2.total_price_wt_disc>=10000])

    df_2 = df_2.drop(df_2[df_2.total_price_wt_disc >= 10000].index,axis=0) #suppretion des valeurs aberantes
    
          
    st.markdown("""
     Les données pour nous ne sont pas cohérentes (prix 49 499, achat 1000 pcs 2 fois dans la journée  ), donc on décide de les supprimer.
 

        """
    )
    
    st.markdown("""
 Les données proposées dans ce DF s'étalent de juillet 2016 à Août 2018. Il n’y a donc qu'une année calendaire complète 2017 et  2 années fiscales 2017 et 2018 qui sont  complètes. L’année calendaire 2018  est intéressante pour l'analyse car on peut observer les tendances. 


        """
    )
    
    
    
    year = df_2['total_price_wt_disc'].groupby(df_2["created_at"].dt.year).agg('sum').sort_values(ascending=False)
    
    st.dataframe(year)
    
    fig, ax = plt.subplots()
    sns.barplot(x=year.index, y=year.values, data= year, palette='Blues_d')
    plt.title(label='CA par année  ', fontsize=20);
    plt.xlabel('Year', fontsize=12)
    plt.ylabel("Chiffre d'affaires", fontsize=12)
    labels = [2016, 2018, 2017]
    ax.set_xticklabels(labels);
    st.pyplot(fig)

    
    year2 = df_2['qty_ordered'].groupby(df_2["created_at"].dt.year).agg('sum').sort_values(ascending=False) #analyse de CA par l'année
    fig, ax = plt.subplots()
    sns.barplot(x=year2.index, y=year2.values,  palette='Blues_d')
    plt.title(label='QTY par année  ', fontsize=20)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel("QTY", fontsize=12);
    
    st.pyplot(fig)
    
    st.markdown("""
        On remarque l'évolution des ventes en 2017 qui continue en 2018 car en 8 mois les chiffres d'affaires en 2018 ont déjà presque le même niveau que sur 12 mois en 2017. La comparaison des CA et quantité de ventes en 2018 nous illustre  le travail sur la marge ou repositionnement a été effectué cette année. 

        """
    )
    
    st.subheader("""
La distribution des ventes par catégorie est suivante:
    
         """
    )
    
    df_2['M-Y'] = pd.to_datetime(df_2['M-Y'])
    df_nombre_cde = pd.crosstab(df_2['M-Y'], df_2['category_name'])
    
    import datetime
    from bokeh.plotting import figure, show, output_notebook
    output_notebook()
    from bokeh.models import  LabelSet, ColumnDataSource
    from bokeh.models import Range1d, OpenURL, TapTool, HoverTool, DatetimeTickFormatter
    from bokeh.transform import linear_cmap
    from datetime import datetime
    from bokeh.models import BoxAnnotation
    from bokeh.plotting import figure
    
     
    source = ColumnDataSource(df_nombre_cde)


    p = figure(width=1100,height=400, x_axis_type='datetime',
           title = "Evolution des ventes par mois")
    hover = HoverTool(
        tooltips=[
            ("date", '@x{%Y-%m-%d}'),            
            ("Qty", "@y")],
            formatters={'@x' : 'datetime'})
            
     
    p.line(x = df_nombre_cde.index, y=df_nombre_cde['Mobiles & Tablets'], color='blue', line_width=2, legend_label='Mobiles & Tablets')
    p.line(x = df_nombre_cde.index, y=df_nombre_cde['Appliances'], line_width=2, color='red', legend_label='Appliances')
    p.line(x = df_nombre_cde.index, y=df_nombre_cde['Computing'], color='steelblue', line_width=2, legend_label='Computing')
    p.line(x = df_nombre_cde.index, y=df_nombre_cde['Superstore'], line_width=2, color='yellow', legend_label='Superstore')
    p.line(x = df_nombre_cde.index, y=df_nombre_cde['Entertainment'], color='orange', line_width=2, legend_label='Entertainment')
    p.line(x = df_nombre_cde.index, y=df_nombre_cde["Men's Fashion"], line_width=2, color='purple', legend_label="Men's Fashion")
    p.line(x = df_nombre_cde.index, y=df_nombre_cde["Women's Fashion"], color='rosybrown', line_width=2, legend_label="Women's Fashion")
    p.line(x = df_nombre_cde.index, y=df_nombre_cde['Others'], line_width=2, color='lawngreen', legend_label='Others')

    p.add_tools(hover)
    p.legend.click_policy = 'hide'

    box_left = pd.to_datetime('2017-10-01-')
    box_right = pd.to_datetime('2017-12-01')


    box = BoxAnnotation(left=box_left, right=box_right,
                    line_width=1, line_color='black', line_dash='dashed',
                    fill_alpha=0.2, fill_color='orange')

    p.add_layout(box)

    st.bokeh_chart(p, use_container_width=True)
    
    
    st.markdown("""
        Nous pouvons voir le pic en Novembre pour presque toutes les catégories et une petite augmentation en Mars, Avril, Mai que pour certaines catégories. Regardons les ventes par jour et mois :  

        """
    )
    fig, ax = plt.subplots()
    sns.countplot(x=df_2["created_at"].dt.month); 
    plt.title("Nombre de ventes par mois", fontsize=20)
    plt.xlabel('Ventes', fontsize=14)
    plt.ylabel('Mois', fontsize=14);
    st.pyplot(fig)
    
    
    fig, ax = plt.subplots()
    sns.countplot(x=df_2["created_at"].dt.day); 
    plt.title("Nombre de ventes par jour", fontsize=20)
    plt.xlabel('Ventes', fontsize=14)
    plt.ylabel('Jour', fontsize=10);
    st.pyplot(fig)

    
    
    st.markdown("""
    Le pic des ventes est bien en Novembre, ce qui peut correspondre aux achats du black friday du 25 novembre. Dans la mois les ventes augmentent  à la fin du mois, ça peut aider à l'organisation de travail des magasin p.ex le intérimaire dans  le période plus intense (très pratiqué par AMAZON)  ou les promotions début de mois ou le mois creuses. 
L’analyse peut être approfondi avec la projection des ventes sur les catégories et les produits concrets:
    """)
    
    most_sold = pd.DataFrame(df_2['sku'].value_counts())
    top_10_most_sold = most_sold[0:10]

    st.dataframe(top_10_most_sold)
    

    
    
    st.subheader("""
    
    L'analyse des vetes par catègorie

        """
    ) 
        
    st.markdown("""
        L'entreprise propose les produits très variés qui sont regrouper par categories, il est très utile d'analyser les ventes par catègories pour se projeter par example sur le repositionement.   

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
    plt.title(label='Top 6 catégorie par quantité de ventes ', fontsize=20)
           
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
        La catégorie Mobile et Tablets est nettement en tête de vente par la quantité des commandes et par les chiffres d'affaires. Par contre après on peux voir le décalage par exemple la catégorie Men's fashion est en 2 place par quantité et en 7 par CA. 

        """
    )
        


    