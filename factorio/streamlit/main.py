import datetime

import streamlit as st
import pandas as pd
import numpy as np

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

st.title('Patient Arrival Prediction')


@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data


# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache)")

if st.button('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

c_date = datetime.datetime.now()
st.subheader('Number of pickups by hour')
hour = 2
to_past = 24 - hour
index = pd.date_range(start=c_date - datetime.timedelta(hours=to_past),
                      end=c_date + datetime.timedelta(hours=hour),
                      freq=f"{60}min")

hist_values = pd.DataFrame(np.abs(np.random.randn(24, 1)),
                           columns=['Arrivals'],
                           index=[pd.to_datetime(date) for date in index]
                           )
st.bar_chart(hist_values, width=50)

# Some number in the range 0-23
hour_to_filter = st.slider('hour', 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

st.subheader('Map of all pickups at %s:00' % hour_to_filter)
st.map(filtered_data)
