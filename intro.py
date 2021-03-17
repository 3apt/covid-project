import streamlit as st
from PIL import Image

def intro():

    st.title('Détecteur de COVID - Version 1')
    st.write('## Analyse de radiographies pulmonaires Covid-19')
    st.image(Image.open('illustrations/goUp200_cVRT.jpg'))
    st.write('''Afin de faire le diagnostic des patients au Covid-19, l’analyse de radiographies 
    pulmonaires est une possibilité à explorer pour détecter plus facilement les cas positifs. 
    Si la classification par le biais du deep learning de telles données se révèle efficace pour 
    détecter les cas positifs, alors cette méthode peut être utilisée dans les hôpitaux et cliniques 
    quand on ne peut pas faire de test classique.''')
    
    st.write('### Dataset')
    st.write('''Le set de donnée contient des images radiographiques pulmonaires pour des cas 
    positifs au covid-19 mais aussi des images radiographiques de pneumonies virales et de patients sains :
    https://www.kaggle.com/tawsifurrahman/covid19-radiography-database''')
    st.write('\n')
    st.write('**Taille des données :** 1.15 Gb')
    st.write('''Il y a 3886 images radiographiques, répartient suivant 3 classes :''')
    st.image(Image.open('illustrations/distrib_dataset.png'))
    st.write('''Aperçu des images :''')
    st.image(Image.open('illustrations/dataset_visu.png'))
    st.write('''Les images n'ont pas toutes la même résolution. Le tableau suivant
             présente le nombre d'images du dataset avec le format "height/width":''')
    st.image(Image.open('illustrations/formats.png'))
    