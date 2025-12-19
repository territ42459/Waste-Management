import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

base = "light" # Choose between "light" or "dark" base themes
primaryColor = "#443c3c" # Color for interactive elements (e.g., buttons)
backgroundColor = "#6b6b6b" # Main app background color
secondaryBackgroundColor = "#000000" # Background for widgets and containers
textColor = "#ffffff" # Default text color

df = pd.read_csv('sustainable_waste_management_dataset_2024.csv')
df.head()

st.title("Sustainable Waste Management Graph")

selected_features = ['population','recyclable_kg','organic_kg' ,'collection_capacity_kg', 'temp_c', 'rain_mm',]
X = df[selected_features]
y = df['waste_kg']

df_combined = pd.concat([X, y], axis=1)
df_combined.dropna(inplace=True)

X = df_combined[selected_features]
y = df_combined['waste_kg']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("MSE: ", mean_squared_error(Y_test, Y_pred))
print("R squared: ", r2_score(Y_test, Y_pred))
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual Waste(Kg) (Y_test)')
plt.ylabel('Predicted Waste(Kg) (Y_pred)')
plt.title('Predicted vs. Actual Waste(kg)')
plt.legend()
plt.grid(True)
st.write("Predicted VS Actual Waste")
st.pyplot()
st.write()
st.line_chart(data=df, x='waste_kg', y='population', x_label='Waste(Kg)', y_label='Population', color="#42bcd4", width='stretch', height="stretch", use_container_width=None)
