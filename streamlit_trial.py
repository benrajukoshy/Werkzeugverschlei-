import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Funktion zum Laden des Skalierers aus der Pickle-Datei
def load_scaler(filename):
    with open(filename, 'rb') as file:
        scaler = pickle.load(file)
    return scaler

# Laden des trainierten Modells
filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))

# Laden der Skalierer
scaler = load_scaler('ScaleFaktors_X.sav')
scaler_y = load_scaler('ScaleFaktors_y.sav')

def scale_input(input_values, scaler):
    X_scaled = scaler.transform(np.array([input_values]))
    return X_scaled
    
def inverse_scale_output(output, scaler_y):
    return scaler_y.inverse_transform(np.array([output]).reshape(-1, 1))

def main():
    st.title("Regressionsübung im ML Seminar, WS23/24")
    st.header("Prognose der Betonfestigkeit")

    # Abschnitt für SelectSlider-Elemente
    st.header("Wählen Sie die Mengen Ihrer Betoninhaltsstoffe aus")

    # Variablen und ihre Bereichsgrenzen
    variables = {
        "cement": (100, 500),
        "slag": (0, 200),
        "flyash": (0, 200),
        "water": (100, 300),
        "superplasticizer": (0, 30),
        "coarseaggregate": (800, 1200),
        "fineaggregate": (600, 1000),
        "age": (1, 365)
    }


    values = []
    for var, (min_val, max_val) in variables.items():
        value = st.select_slider(f"{var.capitalize()} (Einheit)", range(min_val, max_val + 1))
        values.append(value)

    if st.button("Vorhersage machen"):
        input_values_scaled = scale_input(values, scaler)
        prediction_scaled = model.predict(input_values_scaled)
        prediction = inverse_scale_output(prediction_scaled, scaler_y)
        st.write("Prognostizierte Festigkeit Ihres Betons in MPa:")
        st.text_area("Ergebnis", f"{prediction}", height=100)

if __name__ == "__main__":
    main()
