import streamlit as st
from PIL import Image

def conclusion():
    st.title('Conclusion')
    st.write('''Lors de ce projet, nous avons concentré nos efforts sur l'analyse des biais 
             du dataset. En effet, est-ce normal d'avoir une précision de presque 97% avec un
             modèle CNN simple à 2 couches de convolution pour détecter une maladie thoracique ?
             La réponse est non. C'est pour cette raison que nous avons tenté d'en savoir plus
             sur les biais du dataset et de les corriger avant d'appliquer une méthode de Transfer
             Learning.
             ''')
    st.write('## Bilan')
    st.write('- Mise en évidence des biais du dataset par réduction de dimension avec les méthodes t-SNE et pca')    
    st.write("- Développement d'une pipeline pour égaliser les histogrammes et supprimer les bords des images en segmentant les poumons")
    st.write("- Création de deux modèles CNN avec Transfer Learning (Densenet121 et VGG16)")
    st.write("- Utilisation de la récente méthode de Grad-CAM pour visualiser les cartes d'activation des classes du réseau de neurone.")
    
    st.write('## Perspectives')
    st.write("- Corriger les biais toujours présents malgré le preprocessing en utilisant de nouvelles images avec différentes provenances pour chaque classes")
    st.write("- Continuer à optimiser les performances du réseau en entraînant sur plus d'epochs et en dégelant certaines couches du réseau pré-entraîné.")
    st.write("- Travailler avec des radiologues pour évaluer la pertinence des cartes grad-CAM sur des images de patients")
    st.write("- Ajouter au modèle des informations sur les patients en plus des images : symptômes, antécédents médicaux, âge, sexe, localisation, date, etc ; afin de rendre le modèle plus robuste et fidèle au diagnostique médical.")
    st.image(Image.open('illustrations/radio.jpeg'))