
import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app title
st.title('Prediksi Tagihan Listrik Jakarta')
st.write('Aplikasi untuk memprediksi jumlah tagihan listrik berdasarkan parameter yang diberikan.')

# Sidebar for user inputs
st.sidebar.header('Input Parameter')

def user_input_features():
    kwh = st.sidebar.slider('Konsumsi KWH (kWh)', 150.0, 600.0, 350.0)
    ac_units = st.sidebar.slider('Jumlah AC', 0, 3, 1)
    ac_hours_per_day = st.sidebar.slider('Jam AC per Hari', 0.0, 10.0, 5.0)
    family_size = st.sidebar.slider('Jumlah Anggota Keluarga', 2, 6, 4)
    month_name = st.sidebar.selectbox('Bulan', ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    tariff_class = st.sidebar.selectbox('Kelas Tarif', ['R1', 'R2', 'R3'])

    data = {
        'kwh': kwh,
        'ac_units': ac_units,
        'ac_hours_per_day': ac_hours_per_day,
        'family_size': family_size,
        'month_name': month_name,
        'tariff_class': tariff_class
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()
st.subheader('Parameter Input Pengguna:')
st.write(df_input)

# Prepare input to match model's expected encoded columns
# List of columns during training (excluding target)
expected_columns = [
    'kwh', 'ac_units', 'ac_hours_per_day', 'family_size',
    'month_name_Aug', 'month_name_Dec', 'month_name_Feb', 'month_name_Jan',
    'month_name_Jul', 'month_name_Jun', 'month_name_Mar', 'month_name_May',
    'month_name_Nov', 'month_name_Oct', 'month_name_Sep', 'tariff_class_R2',
    'tariff_class_R3'
]

# Apply One-Hot Encoding like we did in training
df_encoded = pd.get_dummies(df_input, columns=['month_name', 'tariff_class'])

# Reindex to ensure all expected dummy columns exist with correct default of 0/False
for col in expected_columns:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

df_encoded = df_encoded[expected_columns]

# Make prediction
if st.sidebar.button('Prediksi Tagihan'):
    try:
        prediction = model.predict(df_encoded)
        st.subheader('Hasil Prediksi Tagihan Listrik:')
        st.write(f"Tagihan Diprediksi: Rp {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.exception(e)
