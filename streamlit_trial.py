import streamlit as st
import pickle
import numpy as np


st.title("Seminararbeit Praktische Einführung in Machine Learning")
st.header("Automatisierte Erkennung und Dokumentation von Werkzeugverschleiß zur Abschätzung der Standzeiten von Werkzeugen")

    # Abschnitt für SelectSlider-Elemente

st.subheader("Klassifikation von Daten nach dem MobilNetV2-Modell mit und ohne Data Augmentation")
option = st.selectbox( ' ',['With Augmentation','Without Augmentation'])

if option == 'With Augmentation':
    Plot_Accuracy_MobilNetV2_Mit_Augmentation ='Plot_Accuracy_MobilNetV2_Mit_Augmentation.png'
    Plot_des_Modellverlustes_MobilNetV2_Mit_Augmentation = 'Plot_des_Modellverlustes_MobilNetV2_Mit_Augmentation.png'
    Konfusion_Matrix_MobilNetV2_Mit_Augmentation = 'Konfusion_Matrix_MobilNetV2_Mit_Augmentation.png'

    st.image(Plot_Accuracy_MobilNetV2_Mit_Augmentation, caption='Plot_Accuracy_MobilNetV2_Mit_Augmentation')  
    st.image(Plot_des_Modellverlustes_MobilNetV2_Mit_Augmentation, caption = 'Plot_des_Modellverlustes_MobilNetV2_Mit_Augmentation')
    st.image(Konfusion_Matrix_MobilNetV2_Mit_Augmentation, caption = 'Konfusion_Matrix_MobilNetV2_Mit_Augmentation')
else:

    Plot_Accuracy_MobilNetV2_Ohne_Augmentation = "Plot_Accuracy_MobilNetV2_Ohne_Augmentation.png"
    Plot_des_Modellverlustes_MobilNetV2_Ohne_Augmentation = "Plot_des_Modellverlustes_MobilNetV2_Ohne_Augmentation.png"
    Modellvorhersage_und_Label_Vergleich_MobilNetV2_Ohne_Augmentation = "Modellvorhersage und Label-Vergleich_MobilNetV2_Ohne_Augmentation.png"
    Konfusion_Matrix_MobilNetV2_Ohne_Augmentation = "Konfusion_Matrix_MobilNetV2_Ohne_Augmentation.png"


    st.image(Plot_Accuracy_MobilNetV2_Ohne_Augmentation, caption='Plot_Accuracy_MobilNetV2_Ohne_Augmentation')
    st.image(Plot_des_Modellverlustes_MobilNetV2_Ohne_Augmentation, caption = 'Plot_des_Modellverlustes_MobilNetV2_Ohne_Augmentation')
    #st.image(Modellvorhersage_und_Label_Vergleich_MobilNetV2_Ohne_Augmentation,caption= 'Modellvorhersage und Label-Vergleich_MobilNetV2_Ohne_Augmentation')
    st.image(Konfusion_Matrix_MobilNetV2_Ohne_Augmentation,caption = 'Konfusion_Matrix_MobilNetV2_Ohne_Augmentation')
    

st.subheader("Klassifikation von Daten nach dem ResNet50-Modell mit und ohne Data Augmentation")
option = st.selectbox( 'Do you want augmentation',['With Augmentation','Without Augmentation'],key = 'select2')
if option == 'With Augmentation':
    Plot_Accuracy_RestNet50_Mit_Augmentation = 'Plot_Accuracy_RestNet50_Mit_Augmentation.png'
    Plot_des_Modellverlustes_RestNet50_Mit_Augmentation = 'Plot_des_Modellverlustes_RestNet50_Mit_Augmentation.png'
    Konfusion_Matrix_RestNet50_Mit_Augmentation = 'Konfusion_Matrix_RestNet50_Mit_Augmentation.png'

    st.image(Plot_Accuracy_RestNet50_Mit_Augmentation, caption= 'Plot_Accuracy_ResNet50_Mit_Augmentation')
    st.image(Plot_des_Modellverlustes_RestNet50_Mit_Augmentation, caption='Plot_des_Modellverlustes_ResNet50_Mit_Augmentation')
    st.image(Konfusion_Matrix_RestNet50_Mit_Augmentation, caption='Konfusion_Matrix_ResNet50_Mit_Augmentation')

else:
    Plot_Accuracy_RestNet50_Ohne_Augmentation = 'Plot_Accuracy_RestNet50_Ohne_Augmentation.png'
    Plot_des_Modellverlustes_RestNet50_Ohne_Augmentation = 'Plot_des_Modellverlustes_RestNet50_Ohne_Augmentation.png'
    Konfusion_Matrix_RestNet50_Ohne_Augmentation = 'Konfusion_Matrix_RestNet50_Ohne_Augmentation.png'

    st.image(Plot_Accuracy_RestNet50_Ohne_Augmentation, caption = 'Plot_Accuracy_ResNet50_Ohne_Augmentation')
    st.image(Plot_des_Modellverlustes_RestNet50_Ohne_Augmentation, caption = 'Plot_des_Modellverlustes_RestNe50_Ohne_Augmentation')
    st.image(Konfusion_Matrix_RestNet50_Ohne_Augmentation, caption = 'Konfusion_Matrix_ResNet50_Ohne_Augmentation')

