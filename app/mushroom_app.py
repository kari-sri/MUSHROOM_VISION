import sys
import os
import streamlit as st
import pandas as pd
import pickle
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure module import works
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import MushroomClassifier

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load image model once
image_model = MushroomClassifier().to(device)
image_model.load_state_dict(torch.load('./models/mushroom_classifier.pth', map_location=device))
image_model.eval()

# Load tabular model once
with open('./models/mushroom_tabular.pkl', 'rb') as f:
    tabular_model = pickle.load(f)

# Prepare image transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

st.title("🍄 Mushroom Classifier")

option = st.sidebar.selectbox(
    'Choose Classification Method:',
    ('Image-Based', 'Feature-Based')
)

if option == 'Image-Based':
    st.header("Upload Mushroom Image for Classification")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Mushroom Image.')

        # Preprocess and predict
        image_tensor = transform(image).unsqueeze(0).to(device)
        output = image_model(image_tensor)
        _, predicted = torch.max(output, 1)

        if predicted.item() == 0:
            st.success("🍄 The mushroom is classified as **Edible**.")
        else:
            st.error("☠️ The mushroom is classified as **Poisonous**.")

        # Visualization: Softmax Probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prob_np = probabilities.cpu().detach().numpy()[0]

        fig, ax = plt.subplots()
        sns.barplot(x=['Edible', 'Poisonous'], y=prob_np, palette='viridis', ax=ax)
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probabilities')
        st.pyplot(fig)


elif option == 'Feature-Based':
    st.header("Enter Mushroom Features for Classification")

    features = {
        'cap-shape': st.selectbox('Cap Shape', ['b', 'c', 'x', 'f', 'k', 's']),
        'cap-surface': st.selectbox('Cap Surface', ['f', 'g', 'y', 's']),
        'cap-color': st.selectbox('Cap Color', ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y']),
        'bruises': st.selectbox('Bruises', ['t', 'f']),
        'odor': st.selectbox('Odor', ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's']),
    }

    if st.button('Predict'):
        input_df = {key: [val] for key, val in features.items()}
        input_df = pd.DataFrame(input_df)

        # One-hot encoding based on training features
        all_features = pd.read_csv('./data/mushrooms.csv')
        all_features = pd.get_dummies(all_features.drop('class', axis=1))
        input_df = pd.get_dummies(input_df).reindex(columns=all_features.columns, fill_value=0)

        prediction = tabular_model.predict(input_df)
        label = 'Edible' if prediction[0] == 0 else 'Poisonous'

        if label == 'Edible':
            st.success(f"🍄 Prediction: **{label}**")
        else:
            st.error(f"☠️ Prediction: **{label}**")

        # Visualization: Feature Importance
        importances = tabular_model.feature_importances_
        feature_names = input_df.columns

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='coolwarm', ax=ax)
        ax.set_title('Top 10 Feature Importances')
        st.pyplot(fig)
