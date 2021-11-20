import datetime

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import plotly.express as px

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

st.title('Patient Arrival Prediction')

count = st_autorefresh(interval=900000, limit=1000, key="fizzbuzzcounter")


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
hour = st.slider('To Future', 0, 23, 17)

to_past = 23 - hour
index = pd.date_range(start=c_date - datetime.timedelta(hours=to_past),
                      end=c_date + datetime.timedelta(hours=hour),
                      freq=f"{60}min")

hist_values = pd.DataFrame(np.abs(np.random.randn(24, 1)),
                           columns=['Arrivals'],
                           index=[pd.to_datetime(date) for date in index]
                           )
hist_values.index.name = 'Datetime'
width = [1 for i in range(24)]
fig = px.bar(hist_values)
fig.add_vrect(x0=c_date, x1=c_date + datetime.timedelta(minutes=5),
              annotation_text="Current time", annotation_position="top left",
              fillcolor="black", opacity=0.5, line_width=0)

fig.update_yaxes(title='y', visible=False, showticklabels=False)
st.plotly_chart(fig, use_container_width=True)

# Some number in the range 0-23
filtered_data = data[data[DATE_COLUMN].dt.hour == hour]

st.subheader('Map of all pickups at %s:00' % hour)
st.map(filtered_data)
