import streamlit as st
from PIL import Image
import pandas as pd
import seaborn as sns
sns.set_theme(style="darkgrid")
import matplotlib.pyplot as plt
from random import choice

def intro():

    st.title('Détecteur de COVID - Version 1')
    st.write('## Analyse de radiographies pulmonaires Covid-19')
    st.image(Image.open('illustrations/goUp200_cVRT.jpg'))
    st.write('''Afin de faire le diagnostic des patients au Covid-19, l’analyse de radiographies 
    pulmonaires est une possibilité à explorer pour détecter plus facilement les cas positifs. 
    Si la classification par le biais du deep learning de telles données se révèle efficace pour 
    détecter les cas positifs, alors cette méthode peut être utilisée dans les hôpitaux et cliniques 
    quand on ne peut pas faire de test classique.''')
    
    df = pd.read_csv('COVID-19 Radiography Database/img_metadata.csv', index_col = 0)

    st.write('### Dataset')
    st.write('''Le set de donnée contient des images radiographiques pulmonaires pour des cas 
    positifs au covid-19 mais aussi des images radiographiques de pneumonies virales et de patients sains :
    https://www.kaggle.com/tawsifurrahman/covid19-radiography-database''')
    st.write('\n')
    st.write('**Taille des données :** 1.15 Gb')
    st.write('''Il y a 3886 images radiographiques, réparties suivant 3 classes :''')
    
    fig = plt.figure()
    sns.countplot(x="Class", data=df)
    st.pyplot(fig)
    
    st.write('''Aperçu des images :''')
    
    st.button(label="Changer les images")
    # url = 'https://3aptiste.s3.eu-west-3.amazonaws.com/COVID-19+Radiography+Database/'
    fig, axes = plt.subplots(1, 3, figsize=(14,9))
    axes = axes.ravel()
    for ii, label in enumerate(('covid', 'normal', 'pneumo')):
        image_name = choice(df.loc[df.Class==label].index)
        axes[ii].set_title(image_name)
        img = plt.imread('COVID-19 Radiography Database/' + image_name)
        axes[ii].imshow(img, cmap = 'gray')
        axes[ii].grid(False);
    st.pyplot(fig)
    st.write('### Formats des images')
    st.write('''Les images n'ont pas toutes la même résolution. Le tableau suivant
             présente le nombre d'images du dataset avec le format hauteur/largeur :''')
    st.dataframe(df[['height', 'width']].value_counts().to_frame("Nombre d'images"))
    
    st.write('### Luminosité des images')
    st.write('''Les images n'ont pas toutes la même luminosité. L'histogramme suivant
         présente les luminosités des images du dataset :''')
    fig = sns.displot(data = df, x="luminosite", hue="Class", multiple="stack")
    st.pyplot(fig)
    