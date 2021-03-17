import streamlit as st
from PIL import Image

def biais():
    st.title('Biais du dataset')
    st.write('''Nous avons identifié lors de l’analyse exploratoire une forte hétérogénéité 
             entre les clichés du dataset COVID et les datasets ‘VIRAL PNEUMONIA’ et ‘NORMAL’ 
             de l’autre côté, probablement due à la provenance des radiographies qui différe.
             Nous faisons alors l’hypothèse que ces disparités dans la qualité 
             des clichés peuvent constituer des biais et ainsi être préjudiciables à la pertinence 
             de notre prototype de classification. 
             Nous avons dans un premier temps cherché à objectiver et caractériser les 
             biais susceptibles de fausser notre intuition et dans un second temps tenter 
             d’apporter une réponse permettant de contourner les écueils rencontrés.''')
    
    st.write('## Visualisation des biais du dataset')
    st.write('''Pour visualiser les biais du dataset, l’idée est de projeter les images dans un 
             espace de dimension inférieure pour observer des tendances dans les images des 3 classes.
             Pour cela, nous redimensionnons les images en 28x28 :''')
    st.image(Image.open('illustrations/image_28x28.png'))
    st.write('''En théorie, il est impossible de détecter le covid ou une pneumonie 
             à cause du manque de détail.''')
    st.write('''Nous appliquons ensuite un t-SNE avec 2 composantes et 
             nous visualisons sur ces deux axes les images 28x28 du dataset 
             (points bleus : normal, points rouges : covid et points verts : pneumonie) :''')
    st.image(Image.open('illustrations/t-SNE_visu.png'))
    st.write('''Nous remarquons avec étonnement qu’il est possible de distinguer les 3 classes 
             sur une projection 2D, avec une méthode de clustering non-supervisée et  
             des images sans détail !''')
    st.write('''En effet, en appliquant un random forest avec uniquement les deux variables 
             correspondant aux deux axes t-SNE, nous obtenons une val_accuracy de presque 93%.''')
    st.write('''Il est même possible d'observer sur l’image suivante qu’une classification SVM sur ces 
             deux composantes semble bien fonctionner :''')
    st.image(Image.open('illustrations/t-SNE_classif.png'))
    st.write('''L’algorithme t-SNE semble extraire sur les images sans détail 28x28 une information 
             importante qui lui permet de faire une bonne classification. Pour faciliter 
             l'interprétation, faisons la même chose avec une PCA :''')
    st.image(Image.open('illustrations/pca_visu.png'))
    st.write('''La projection est moins efficace qu’avec la méthode t-SNE,
             mais nous remarquons que le premier mode (axe des abscisses) permet de séparer correctement 
             les covid des non-covid. Suite à cette observation, nous pouvons afficher 
             le vecteur pca.components_[0], en prenant seulement les valeurs absolues des coefficients, 
             et en affichant la heatmap 28x28 :''')
    st.image(Image.open('illustrations/pca_premier_mode.png'))
    st.write('''Nous observons alors très bien que les pixels qui ont de l’importance 
             dans la projection suivant le premier mode pca sont les pixels aux bords droit et 
             gauche de l’image. La luminosité des bords permet donc de séparer correctement 
             les covids des non-covid dans notre dataset, ce qui représente un réel biais 
             dans notre analyse !''')
    st.write('''Suite à ce constat, nous sommes passés à la seconde phase du projet. 
             Ayant identifié les biais probables, il convenait de corriger ces biais avant 
             de poursuivre la conception d’un réseau de neurones.''')

    
    

    




    