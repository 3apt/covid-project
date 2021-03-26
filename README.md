# Détecteur de COVID

Final project for Datascientest bootcamp jan21

## Analyse de radiographies pulmonaires Covid-19

Afin de faire le diagnostic des patients au Covid-19, l’analyse de radiographies pulmonaires est une possibilité à explorer pour détecter plus facilement les cas positifs. Si la classification par le biais du deep learning de telles données se révèle efficace pour détecter les cas positifs, alors cette méthode peut être utilisée dans les hôpitaux et cliniques quand on ne peut pas faire de test classique.

## Dataset

Le set de donnée contient des images radiographiques pulmonaires pour des cas positifs au covid-19 mais aussi des images radiographiques de pneumonies virales et de patients sains : [Kaggle dataset](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database/)

### Démo streamlit

[Application Streamlit](https://share.streamlit.io/3apt/covid-project/main/app.py)

## Fichiers github

- Codes
	- 0_Projet_COVID_explo.ipynb
	- 0_Projet_COVID_segmentation.ipynb
	- 1_Projet_COVID_premier_prototype.ipynb (LeNet)
	- 2_Dataset_Visualization.ipynb (t-SNE / PCA)
	- 2_Segmentation_v1.ipynb
	- 2_Segmentation_v2.ipynb (U-Net segmentation)
	- 2_deuxieme_prototype.ipynb (Densenet121)
	- 3_troisieme_prototype.ipynb (VGG16)
	
- Fonctions
	- functions_covid.py
	
- Streamlit
	- app.py (main)
	- intro.py
	- biais.py
	- preprocessing.py
	- application.py
	- conclusion.py
	- functions_streamlit.py
	- requirements.txt
	- packages.txt

- Dataset
	- COVID-19 Radiography Database/
	- COVID-19 Radiography Database/img_metadata.csv (metadata)
	- cropped_dataset/ (preprocessed)
	
- Modèles
	- COVID-19 Radiography Database/premier_prototype.h5 (LeNet)
	- cropped_dataset/deuxieme_prototype.h5 (Densenet121)
	- cropped_dataset/troisieme_prototype.h5 (VGG16)

- 
