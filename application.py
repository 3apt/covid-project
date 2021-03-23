import streamlit as st
import numpy as np
import functions_streamlit
from PIL import Image

def application():
    # App Title
    st.title("Détecteur de COVID")
    
    option = 'VGG16'
    option = st.selectbox('Quel modèle voulez-vous utiliser ?',('densenet121', 'VGG16', 'CNN simple (biaisé)'))
    
    if option == 'VGG16':
        
        # Introduction text
        st.markdown(unsafe_allow_html=True, body="<p>Bienvenue sur le détecteur de COVID-19 et pneumonie.</p>"
                                                 "Avec cette app, vous pouvez uploader une radiographie thoracique et prédire si le patient "
                                                 "est atteint du COVID, d'une pneumonie ou sain.</p>"
                                                 "<p>Le modèle est un réseau de neurone de convolution basé sur VGG16, "
                                                 "il a pour le moment une précision test de "
                                                 "<strong>85.6%.</strong></p>")
        if st.checkbox('Afficher la description du modèle'):
            st.image(Image.open('illustrations/vgg16.png'))
    if option == 'densenet121':
        # Introduction text
        st.markdown(unsafe_allow_html=True, body="<p>Bienvenue sur le détecteur de COVID-19 et pneumonie.</p>"
                                                 "Avec cette app, vous pouvez uploader une radio de la poitrine et prédire si le patient "
                                                 "est atteint du COVID, d'une pneumonie ou sain.</p>"
                                                 "<p>Le modèle est un réseau de neurone de convolution basé sur densenet121, "
                                                 "il a pour le moment une précision test de "
                                                 "<strong>91.4%.</strong></p>")
        if st.checkbox('Afficher la description du modèle'):
            st.image(Image.open('illustrations/densenet121.png'))
    if option == 'CNN simple (biaisé)':
        # Introduction text
        st.markdown(unsafe_allow_html=True, body="<p>Bienvenue sur le détecteur de COVID-19 et pneumonie.</p>"
                                                 "Avec cette app, vous pouvez uploader une radio de la poitrine et prédire si le patient "
                                                 "est atteint du COVID, d'une pneumonie ou sain.</p>"
                                                 "<p>Le modèle est un réseau de neurone à 2 couches de convolution, "
                                                 "il a pour le moment une précision test de "
                                                 "<strong>97.3%.</strong></p>"
                                                 "<p style='color:red'>Attention : ce modèle a été entrainé sur le dataset biaisé.</p>")
        if st.checkbox('Afficher la description du modèle'):
            st.image(Image.open('illustrations/cnn_simple.png'))
    st.write('### Grad-CAM')
    
    st.write('''Pour améliorer l'interprétabilité du modèle, nous avons adopté la méthode de 
             Mapping Grad-CAM [1] pour visualiser les régions importantes menant à la 
             décision du modèle de deep learning. 
             Elle permet d'obtenir une carte d'activation de la classe déterminée par le réseau
             en utilisant la dernière couche de convolution.
             Cette carte est affichée en dessous de la décision lorsque vous chargez une image.
             ''')
    st.write("### Commencez par charger une image radio des poumons")
    
    # uploader une image
    image_name = st.file_uploader(label="Charger l'image", type=['jpeg', 'jpg', 'png'], key="xray")
    
    
    if image_name is None:
        image_name = 'COVID-19 Radiography Database/COVID (1).png'
    
    # chargement de l'image
    img = np.array(Image.open(image_name))
    
    im_shape = (226, 226)
    if option == 'CNN simple (biaisé)':
        im_shape = (160, 160)
    # preprocess de l'image
    img_pp = functions_streamlit.preprocess_image(img, im_shape)
    
    
    if st.checkbox('Avec preprocessing'):
        st.image((img_pp[0]*255).astype(np.uint8), use_column_width=True)

    else:
        st.image(img)

        
    # chargement du modèle
    if option == 'VGG16':
        MODEL = "troisieme_prototype.h5"
    if option == 'densenet121':
        MODEL = "deuxieme_prototype.h5"
    if option == 'CNN simple (biaisé)':
        MODEL = "premier_prototype.h5"
    loading_msg = st.empty()
    loading_msg.text("En cours de prédiction..")
    model = functions_streamlit.load_model(MODEL)

    # Prédiction
    prob, prediction = functions_streamlit.predict(model, img_pp)

    loading_msg.text('')

    if prediction == 'normal':
        st.markdown(unsafe_allow_html=True, body="<span style='color:green; font-size: 50px'><strong><h3>Normal! :smile:</h3></strong></span>")
    elif prediction == 'covid':
        st.markdown(unsafe_allow_html=True, body="<span style='color:red; font-size: 50px'><strong><h3>COVID! :slightly_frowning_face: </h3></strong></span>")
    elif prediction == 'pneumo':
        st.markdown(unsafe_allow_html=True, body="<span style='color:red; font-size: 50px'><strong><h3>Pneumonie! :slightly_frowning_face: </h3></strong></span>")

    st.text(f"*Probabilité associée à la prédiction : {round(prob * 100, 2)}%")
    
    if option == 'VGG16':
        last_conv_layer_name = "block5_conv3"
        classifier_layer_names = [
            'block5_pool',
            'global_average_pooling2d_3',
            'dense_9',
            'dropout_7',
            'dense_10',
            'dropout_8',
            'dense_11',
            'dropout_9',
            'dense_12',
            'dropout_10',
            'dense_13',
            'dropout_11',
            'dense_14']
    if option == 'densenet121':
        last_conv_layer_name = "bn"
        classifier_layer_names = [
            "relu",
            "avg_pool",
            "dense_3",
            "dropout_2",
            "dense_4",
            "dropout_3",
            "dense_5"]
    if option == 'CNN simple (biaisé)':
        last_conv_layer_name = "conv2d_1"
        classifier_layer_names = [
            "max_pooling2d_1",
            "dropout",
            "flatten",
            "dense",
            "dropout_1",
            "dense_1"]
    
    heatmap = functions_streamlit.make_gradcam_heatmap(
        img_pp, model, last_conv_layer_name, classifier_layer_names)
    st.markdown(unsafe_allow_html=True, body="<h3>Zone d'intérêt du réseau de neurone dans l'image pour prendre sa décision (Grad-CAM)</h3>")
    st.image(heatmap, use_column_width=True)
    
    if option != 'CNN simple (biaisé)':
        st.markdown(unsafe_allow_html=True, body="<h3>Image coloriée par le réseau en amont du transfer learning</h3>")
        st.image(functions_streamlit.colorize(model, img_pp), use_column_width=True)
    
    st.write('''[1] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). 
             Grad-cam: Visual explanations from deep networks via gradient-based localization. 
             In Proceedings of the IEEE international conference on computer vision (pp. 618-626).''')
    
