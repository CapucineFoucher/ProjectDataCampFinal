import streamlit as st
from model import Model
from tempfile import NamedTemporaryFile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from userTest import createAndSave
import os

# Create or load the model
def create_or_load_model():
    if os.path.exists('model.keras'):
        model = Model()
        model.load_model()
    else:
        createAndSave()
        model = Model()
        model.load_model()
    return model

create_or_load_model()


# Define page functions
def homepage():
    st.title("👀 Home Page 👀")
    st.image("pictures/image_home.jpg", use_column_width=True)
    st.markdown('<p style="font-size: 40px; color : lightblue; font-weight: bold;"> Welcome to our Eye Disease Detection App ! </p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 40px; color : lightblue; font-weight: bold;">Want to learn about the health of your eyes ? Go to eye disease detection page !</p>', unsafe_allow_html=True)
    # Add presentation of the project 
    st.markdown('<p style="font-size: 35px; color : lightblue; font-weight: bold;">Presentation of the project</p>', unsafe_allow_html=True)
    st.write("""
    In this project, our primary aim is to create a user-friendly web platform for early eye cancer detection. 
    We will preprocess a dataset of eye images, including resizing, normalization, and data partitioning. 
    Our cloud-based image classification model will focus on detecting eye cancer. Users can easily upload images for assessment. 
    Our goal is to showcase the power of cloud-based AI in medical image analysis and aid in early eye cancer detection.
    """)
    # Add graphics of the dataset 
    st.markdown('<p style="font-size: 35px; color : lightblue; font-weight: bold;">Graphical representation of the dataset</p>', unsafe_allow_html=True)

    # Charger vos données (remplacez cela par votre propre méthode de chargement)
    df_test = pd.read_csv('data/Test_Set/RFMiD_Testing_Labels.csv')

    # Comptez le nombre d'images par classe
    class_counts = df_test['Disease_Risk'].value_counts()

    # Créez le graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    class_counts.plot(kind='bar', ax=ax)
    ax.set_title('Distribution of eye diseases in the test set')
    ax.set_xlabel('Disease Risk')
    ax.set_ylabel('Number of images')

    st.write(""" As we can see a lot of people are affected by eye diseases.
             That means that they might have a risk of eye cancer.
             So our app can be useful for people, to detect eye's diseases early and to be able to treat them. """)
    
    # Affichez le graphique dans Streamlit
    st.pyplot(fig)

    # Add explenation of the data cleaning 
    st.markdown('<p style="font-size: 35px; color : lightblue; font-weight: bold;">Data cleaning</p>', unsafe_allow_html=True)
    
    st.write(""" For the data cleaning we decided to drop all the columns diseases that nobody had.""")

    st.write(""" Here's the correlation matrix before dropping empty columns:""")

    # Exclure les deux premières colonnes (ID et Disease_Risk) pour la matrice de corrélation
    correlation_df = df_test.iloc[:, 2:].astype(float)

    # Calculer la matrice de corrélation
    correlation_matrix = correlation_df.corr()

    # Créer une figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Tracer la matrice de corrélation sous forme de carte de chaleur
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".1f", ax=ax)

    # Afficher le graphique dans Streamlit
    st.pyplot(fig)

    st.write(""" Here's the correlation matrix after dropping empty columns:""")

    df_test_cleaned = pd.read_csv('data/Test_Set/test_cleaned.csv')
    # Exclure les deux premières colonnes (ID et Disease_Risk) pour la matrice de corrélation
    correlation_df = df_test_cleaned.iloc[:, 2:].astype(float)

    # Calculer la matrice de corrélation
    correlation_matrix = correlation_df.corr()

    # Créer une figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Tracer la matrice de corrélation sous forme de carte de chaleur
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".1f", ax=ax)

    # Afficher le graphique dans Streamlit
    st.pyplot(fig)

    # Add presentation of the model 
    st.markdown('<p style="font-size: 35px; color : lightblue; font-weight: bold;">Presentation of the prediction model</p>', unsafe_allow_html=True)

    st.write("The project’s algorithm is an implementation of a machine learning model for disease risk prediction based on medical images. It begins by importing necessary libraries, including TensorFlow and Pandas, to facilitate data processing and neural network construction. The data is loaded from CSV files containing image labels and locations. Images are preprocessed and normalized using the ImageDataGenerator from Keras, and then split into training and testing datasets. The neural network architecture consists of several convolutional and fully connected layers, designed to learn relevant features from the input images. The model is compiled with binary cross-entropy loss and accuracy metrics. After compiling, it is trained using the training dataset, and its performance is evaluated on the test dataset. The model is saved and loaded for future use, and it also makes predictions on individual images (for the user use on the website). This project is designed for predicting disease risk based on medical images and can provide valuable insights for medical diagnosis.")
    st.write("On the user side, they only need to get to the Eye disease page, upload their eye image and learn about their diagnosis.")


def detection_page():
    st.title("👁️ Upload Your Eye Image 👁️")

    # Upload an image
    uploaded_image = st.file_uploader("Upload an eye image", type=["jpg", "png", "jpeg"])
    

    # Check if an image has been uploaded
    if uploaded_image is not None:
        with NamedTemporaryFile(dir='.', suffix='.png') as f:
            f.write(uploaded_image.getbuffer())
            model = Model()
            model.load_model()

            # Display the uploaded image
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)


            # Make a prediction using the model
            prediction = model.predict(f.name)

            # Display the prediction
            st.write(f"Prediction: {prediction}")
            #Display the accuracy
            st.write(f"Accuracy: {model.evaluate_model()[1]}")
            if prediction == 0:
                st.write("✅ You are healthy ! Enjoy life ! ✅")
            else:
                st.markdown('<p style="font-size: 35px; font-weight: bold;">👩🏻‍⚕️👩🏼‍⚕️ You are sick ! You are exposed to Cancer risks ! Please see a doctor ASAP ! 👩🏼‍⚕️👩🏻‍⚕️</p>', unsafe_allow_html=True)


def about_page():
    st.title("👩🏻‍💻👩🏼‍💻👩🏻‍💻 About us 👩🏻‍💻👩🏼‍💻👩🏻‍💻")
    # Add prsentation of the team members
    st.markdown('<p style="font-size: 35px; color : lightblue; font-weight: bold;">Presentation of the team members</p>', unsafe_allow_html=True)
    # Création des trois cadres
    col1, col2, col3 = st.columns(3)

    # Cadre pour la première personne
    with col1:
        st.image("pictures/capucine.png", use_column_width=True)  
        st.subheader("Capucine Foucher")
        st.write("Prediction Model + Contribution to Frontend")

    # Cadre pour la deuxième personne
    with col2:
        st.image("pictures/ariane.png", use_column_width=True)  
        st.subheader("Ariane Rousseau")
        st.write("Data Cleaning + Contribution to Frontend")

    # Cadre pour la troisième personne
    with col3:
        st.image("pictures/Anya.JPG", use_column_width=True)  
        st.subheader("Anya Tabti")
        st.write("Streamlit frontend")


st.markdown('<style>h1 {font-size: 60px; color: lightblue;}</style>', unsafe_allow_html=True)
st.sidebar.markdown('<h1 style="font-size: 60px; color: lightblue;">Menu</h1>', unsafe_allow_html=True)

# Manage session state
if "page" not in st.session_state:
    st.session_state.page = "Home"  # Default to the home page
# Add navigation
page = st.sidebar.selectbox("Select a page", ("Home", "Eye Disease", "About us"))

# Conditional page rendering
if page == "Home":
    homepage()
elif page == "Eye Disease":
    detection_page()
elif page == "About us":
    about_page()

st.sidebar.write("Made with ❤️ by: Ariane Rousseau, Anya Tabti & Capucine Foucher")