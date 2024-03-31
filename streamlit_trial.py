import streamlit as st
import pickle
import numpy as np


st.title("Seminararbeit Praktische Einführung in Machine Learning")
st.header("Automatisierte Erkennung und Dokumentation von Werkzeugverschleiß zur Abschätzung der Standzeiten von Werkzeugen")

    # Abschnitt für SelectSlider-Elemente

st.subheader("Klassifikation von Daten nach dem MobilNetV2-Modell mit und ohne Data Augmentation")
st.button("No Data Augmentation", type="primary")
if st.button('Data Augmentation'):
    st.write('Why hello there')
    filename = "afro-guitar-colorful-music-sunglasses-digital-art-4k-wallpaper-uhdpaper.com-222@1@n.jpg"
    st.image(filename, caption='Sunrise by the mountains')  
else:

    Plot_Accuracy_MobilNetV2_Ohne_Augmentation = "Plot_Accuracy_MobilNetV2_Ohne_Augmentation.png"
    Plot_des_Modellverlustes_MobilNetV2_Ohne_Augmentation = "Plot_des_Modellverlustes_MobilNetV2_Ohne_Augmentation.png"
    Modellvorhersage_und_Label_Vergleich_MobilNetV2_Ohne_Augmentation = "Modellvorhersage und Label-Vergleich_MobilNetV2_Ohne_Augmentation.png"
    Konfusion_Matrix_MobilNetV2_Ohne_Augmentation = "Konfusion_Matrix_MobilNetV2_Ohne_Augmentation.png"
    st.image(Plot_Accuracy_MobilNetV2_Ohne_Augmentation, caption='Plot_Accuracy_MobilNetV2_Ohne_Augmentation')
    st.image(Plot_des_Modellverlustes_MobilNetV2_Ohne_Augmentation, caption = 'Plot_des_Modellverlustes_MobilNetV2_Ohne_Augmentation')
    #st.image(Modellvorhersage_und_Label_Vergleich_MobilNetV2_Ohne_Augmentation,caption= 'Modellvorhersage und Label-Vergleich_MobilNetV2_Ohne_Augmentation')
    st.image(Konfusion_Matrix_MobilNetV2_Ohne_Augmentation,caption = 'Konfusion_Matrix_MobilNetV2_Ohne_Augmentation')

