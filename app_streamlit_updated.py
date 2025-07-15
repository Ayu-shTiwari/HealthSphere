import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Paths
MODELS_DIR = "models"
DATASETS_DIR = "datasets"
appointments_file = "booked_appointments.csv"
doctors_dataset_file = os.path.join(DATASETS_DIR, "doctors_dataset.csv")

# Load Models
def load_model(model_name):
    model_path = os.path.join(MODELS_DIR, model_name)
    model = joblib.load(model_path)
    return model

# Load Datasets
def load_dataset(file_name):
    file_path = os.path.join(DATASETS_DIR, file_name)
    return pd.read_csv(file_path)

# Decode Disease Predictions
def create_disease_mapping(description_df):
    return {idx: disease for idx, disease in enumerate(description_df["Disease"].unique())}

def decode_disease_id(disease_id, disease_mapping):
    return disease_mapping.get(disease_id, "Unknown Disease")

# Prediction Result Logic
def pred_result(model, X, disease_mapping, sd, sp, d, md):
    proba = model.predict_proba(X)
    top5_idx = np.argsort(proba[0])[-5:][::-1]
    top5_proba = np.sort(proba[0])[-5:][::-1]
    top5_disease_ids = top5_idx
    top5_diseases = [decode_disease_id(disease_id, disease_mapping) for disease_id in top5_disease_ids]

    result_dict = []
    for i in range(len(top5_diseases)):
        disease = top5_diseases[i]
        probability = top5_proba[i]
        disease_info = {
            "Disease Name": disease,
            "Probability": probability,
            "Disease Description": None,
            "Recommended Things to do at home": [],
            "Healthy Diet to Follow": [],
            "Recommended Medication": []
        }

        if disease in sd["Disease"].unique():
            disease_info["Disease Description"] = sd[sd['Disease'] == disease].iloc[0, 1]

        if disease in sp["Disease"].unique():
            disease_info["Recommended Things to do at home"] = sp[sp['Disease'] == disease].iloc[0, 1].split(", ")

        if disease in d["Disease"].unique():
            disease_info["Healthy Diet to Follow"] = d[d['Disease'] == disease].iloc[0, 1].split(", ")

        if disease in md["Disease"].unique():
            disease_info["Recommended Medication"] = md[md['Disease'] == disease].iloc[0, 1].split(", ")

        result_dict.append(disease_info)
    
    return result_dict, top5_diseases, top5_proba

# Load Doctors Dataset
if os.path.exists(doctors_dataset_file):
    doctor_data = pd.read_csv(doctors_dataset_file)
else:
    doctor_data = pd.DataFrame(columns=["Doctor Name", "Clinic Name", "Rating"])

# Load previous appointments if available
if os.path.exists(appointments_file):
    booked_appointments = pd.read_csv(appointments_file)
else:
    booked_appointments = pd.DataFrame(columns=["Doctor Name", "Clinic Name", "Rating"])

# Streamlit App
st.title("Health Management App")

# Sidebar for Navigation
option = st.sidebar.selectbox(
    "Choose an option",
    ["Home", "Upload Dataset", "Predict Disease", "Book a Doctor", "View Appointments"]
)

if option == "Home":
    st.write("Welcome to the Health Management App!")
    st.write("Use the sidebar to navigate through the app.")

elif option == "Upload Dataset":
    st.subheader("Upload a Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset:")
        st.dataframe(df)
        if st.button("Save Dataset"):
            dataset_path = os.path.join(DATASETS_DIR, uploaded_file.name)
            df.to_csv(dataset_path, index=False)
            st.success(f"Dataset saved to {DATASETS_DIR}!")

elif option == "Predict Disease":
    st.subheader("Predict Disease")
    model_name = st.selectbox("Select a Model", os.listdir(MODELS_DIR))
    st.write("Enter Symptoms")
    symptom_input = st.text_area(
        "List your symptoms separated by commas (e.g., fever, headache, fatigue)"
    )
    description_df = load_dataset("description.csv")
    precautions_df = load_dataset("precautions_df.csv")
    diets_df = load_dataset("diets.csv")
    medications_df = load_dataset("medications.csv")
    disease_mapping = create_disease_mapping(description_df)

    if st.button("Predict from Symptoms"):
        if symptom_input.strip():
            try:
                symptoms = [s.strip().lower() for s in symptom_input.split(",")]
                model = load_model(model_name)
                symptom_features = pd.DataFrame(
                    [dict((symptom, 1) for symptom in symptoms)],
                    index=[0]
                ).reindex(columns=model.feature_names_in_, fill_value=0)
                if symptom_features.empty or symptom_features.sum().sum() == 0:
                    st.warning("No valid symptoms found. Please check your input.")
                else:
                    results, top5_diseases, top5_proba = pred_result(
                        model, symptom_features, disease_mapping, description_df, precautions_df, diets_df, medications_df
                    )
                    for result in results:
                        st.write("------------------------------------------------------")
                        st.write(f"Disease Name: {result['Disease Name']}")
                        st.write(f"Probability: {result['Probability']:.2f}")
                        st.write(f"Disease Description: {result['Disease Description']}")
                        st.write(f"Recommended Things to do at home: {result['Recommended Things to do at home']}")
                        st.write(f"Healthy Diet to Follow: {result['Healthy Diet to Follow']}")
                        st.write(f"Recommended Medication: {result['Recommended Medication']}")
                        st.write("------------------------------------------------------")
                    st.subheader("Disease Probability Graph")
                    plt.figure(figsize=(10, 6))
                    plt.bar(top5_diseases, top5_proba, color='skyblue')
                    plt.xlabel("Diseases")
                    plt.ylabel("Probabilities")
                    plt.title("Top 5 Predicted Diseases and Their Probabilities")
                    plt.xticks(rotation=45, ha="right")
                    st.pyplot(plt)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter symptoms.")

elif option == "Book a Doctor":
    st.subheader("Book a Doctor")
    if not doctor_data.empty:
        st.write("Available Doctors:")
        st.dataframe(doctor_data)
        selected_doctor = st.selectbox("Select a Doctor to Book", doctor_data["Doctor Name"].unique())
        if st.button("Book Appointment"):
            selected_row = doctor_data[doctor_data["Doctor Name"] == selected_doctor].iloc[0]
            new_appointment = {
                "Doctor Name": selected_row["Doctor Name"],
                "Clinic Name": selected_row["Clinic Name"],
                "Rating": selected_row["Rating"]
            }
            # Use pd.concat instead of append
            booked_appointments = pd.concat([booked_appointments, pd.DataFrame([new_appointment])], ignore_index=True)
            booked_appointments.to_csv(appointments_file, index=False)
            st.success(f"Appointment booked with {selected_row['Doctor Name']} at {selected_row['Clinic Name']}")
    else:
        st.error("No doctor data available. Please ensure the doctors dataset is present in the datasets folder.")

elif option == "View Appointments":
    st.subheader("Your Booked Appointments")
    if not booked_appointments.empty:
        st.dataframe(booked_appointments)
    else:
        st.write("No appointments booked yet!")