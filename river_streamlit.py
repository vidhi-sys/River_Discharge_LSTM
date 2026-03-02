import streamlit as st
import pandas as pd

st.title("River Discharge & Flood Risk System")

sample_month = 6
sample_input = pd.DataFrame({'Month': [sample_month]})

prediction = rf.predict(sample_input)[0]

if prediction < 200:
    st.success("Low Flood Risk")
elif prediction < 500:
    st.warning("Moderate Flood Risk")
else:
    st.error("High Flood Risk")
st.line_chart(station_data.set_index('Time')['Discharge'])