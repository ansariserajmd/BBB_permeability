import pickle
import streamlit as st
import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
from rdkit.Chem import GetMolFrags
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu

# Load the model, selected_features, and scaler from the pickle file
filename = "BBB_model_des_3d.sav"
with open(filename, 'rb') as f:
    loaded_model, selected_features = pickle.load(f)

# BBB prediction function
def predict_bbb_permeability(smiles_string):
    # Convert the input SMILES to isomeric form
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None
    
    isomeric_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    
    # Calculate descriptors using Mordred
    calc = Calculator(descriptors, ignore_3D=False)
    descriptors_df = calc.pandas([mol])
    
    # Drop invalid columns and fill missing values with 0
    for column in descriptors_df.columns:
        if descriptors_df[column].dtype == object:
            descriptors_df[column] = pd.to_numeric(descriptors_df[column], errors='coerce').fillna(0)
        elif pd.api.types.is_numeric_dtype(descriptors_df[column]):
            descriptors_df[column] = pd.to_numeric(descriptors_df[column], errors='coerce').fillna(0)
        else:
            descriptors_df[column] = 0.0

    # Select only the features used during training from 'input_data' based on selected_features
    input_data_selected = descriptors_df[selected_features]
    
    
    input_scaled_df = pd.DataFrame(input_data_selected)
    
    # Make BBB permeability prediction using the loaded model
    prediction = loaded_model.predict(input_scaled_df)
    
    return prediction

# Streamlit app code
st.title("Blood Brain Barrier Permeability using AI")

# Take SMILES input
SMILES_user = st.text_input("Enter SMILES")

# Create a button for prediction
if st.button("BBB Permeability Prediction"):
    # Call the prediction function with the user input
    prediction = predict_bbb_permeability(SMILES_user)
    if prediction is None:
        st.error("Invalid SMILES input. Please enter a valid SMILES string.")
    else:
        if prediction == 0:
            st.success("SMILES is BBB: Negative")
        else:
            st.success("SMILES is BBB: Positive")
