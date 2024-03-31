import streamlit as st
import pickle
import numpy as np


st.title("Regressionsübung im ML Seminar, WS23/24")
st.header("Prognose der Betonfestigkeit")

    # Abschnitt für SelectSlider-Elemente

st.header("Wählen Sie die Mengen Ihrer Betoninhaltsstoffe aus")
st.button("No Data Augmentation", type="primary")
if st.button('Data Augmentation'):
    st.write('Why hello there')
    filename = "afro-guitar-colorful-music-sunglasses-digital-art-4k-wallpaper-uhdpaper.com-222@1@n.jpg"
    st.image(filename, caption='Sunrise by the mountains')  
else:

    filename = "Plot_Accuracy_MobilNetV2_Ohne_Augmentation.png"
    st.image(filename, caption='Plot_Accuracy_MobilNetV2_Ohne_Augmentation')