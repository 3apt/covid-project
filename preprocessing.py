import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from random import choice

def preprocessing():
    st.title('Preprocessing des images')
    st.write('''Suite à l'analyse des biais du dataset,
             nous avons observé que les algorithmes
             de classification se basaient principalement sur les pixels en bordure d'image
             pour prendre leur décision.
             
             Notre idée est donc de supprimer ces bords, 
             et de ne garder que les poumons qui concentrent l’information 
             nécessaire à la classification.''')
    st.write('''Pour cela, nous avons utilisé un réseau de neurones U-net* pré-entrainé [1] et 
             développé pour la segmentation des poumons sur des radiographies de cages thoraciques''')
    st.write('''    La séquence ci-dessous illustre la transformation des images à l’issue 
             de l’ensemble des méthodes de pré-processing : ''')
    st.image(Image.open('illustrations/segment_process.png'))
    st.write('''Après segmentation, les images sont croppées autour des poumons 
             avec une marge de 10 pixels. Certaines images ont été retirées du dataset 
             du fait que la segmentation n’a pas fonctionné. La répartition des images 
             retirées du dataset est la suivante :''')
    st.image(Image.open('illustrations/images_non_segmentees.png'))
    st.write('''Les images homogénéisées et croppées sur les poumons sont ensuite 
             utilisées pour constituer notre nouveau dataset d'entraînement et de 
             test de modèles de deep learning.''')
    
    st.write('### Aperçu des images avant/après le preprocessing')
    st.button(label="Changer l'image")
    url = 'https://3aptiste.s3.eu-west-3.amazonaws.com/'
    df = pd.read_csv('COVID-19 Radiography Database/img_metadata.csv', index_col = 0)
    df = df[df['keep'] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(14,9))
    axes = axes.ravel()
    image_name = choice(df.index)
    for ii, dossier in enumerate(('COVID-19+Radiography+Database/', 'cropped_dataset/')):
        if ii == 0:
            axes[ii].set_title('Avant crop \n' + image_name)
        else:
            axes[ii].set_title('Après crop \n' + image_name)
        img = plt.imread(url + dossier + image_name.replace(' ', '+'))
        axes[ii].imshow(img, cmap = 'gray', aspect = 'equal')
        axes[ii].grid(False);
    st.pyplot(fig)
    st.write('''En appliquant à nouveau la méthode du t-SNE sur les images croppées,
             nous observons qu'il reste des biais dans le dataset, puisque l'algorithme
             non-supervisé arrive encore à séparer les classes en projetant sur 2 dimensions :''')
    st.image(Image.open('illustrations/t-SNE_visu_cropped.png'))
    if st.checkbox("Plus d'info sur U-net"):
            
        st.write('''*Réseau entièrement convolutionnel U-net :''')
        st.image(Image.open('illustrations/unet_lung.png'))
        st.write('''[1] Haghanifar, A., Majdabadi, M. M., & Ko, S. (2020). 
                                            Covid-cxnet: Detecting covid-19 in frontal chest x-ray 
                                            images using deep learning. arXiv preprint arXiv:2006.13807''')

