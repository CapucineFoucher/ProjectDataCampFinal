#import of the librairies 

import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.optimizers import Adam

# Load the data

current_dir = os.getcwd()

trainDir = os.path.join(current_dir, 'data/Training_Set')
testDir = os.path.join(current_dir, 'data/Test_Set')
evalDir = os.path.join(current_dir, 'data/Evaluation_Set')

# Define data directories 
df_train = pd.read_csv(trainDir + '/RFMiD_Training_Labels.csv')
print(df_train.head())
df_test = pd.read_csv(testDir +'/RFMiD_Testing_Labels.csv')
print(df_test.head())

# Data Prep (normalement fait par ariane)

df_train['ID'] = df_train['ID'].astype(str) + '.png'
df_test['ID'] = df_test['ID'].astype(str) + '.png'


class Model :
    def __init__(self):
        # Load and preprocess images
        # Set your desired image size
        image_size = (224, 224)
        batch_size = 32

        self.modelPath = "model.keras"

        # Normalize pixel values
        datagen = ImageDataGenerator(rescale=1.0/255.0)  

        self.train_data = datagen.flow_from_dataframe(
            df_train, 
            directory=trainDir + "/Train_standardized/", 
            x_col="ID", 
            y_col="Disease_Risk", 
            target_size=image_size, 
            batch_size=batch_size, 
            class_mode='raw',
            shuffle=True,
            seed=42,
        )

        self.test_data = datagen.flow_from_dataframe(
            df_test, 
            directory=testDir + "/Test_standardized/", 
            x_col="ID", 
            y_col="Disease_Risk", 
            target_size=image_size, 
            batch_size=batch_size, 
            class_mode='raw',
            shuffle=True,
            seed=42,
        )

        # Define the Keras model
        self.model = keras.Sequential([
            Conv2D(32, (3, 3), activation='relu', 
            input_shape=(224, 224, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])


    def compile_model(self):
        # Compile the model
        return self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self):
        # Train the model
        return self.model.fit(self.train_data, epochs=10, validation_data=self.test_data)

    def evaluate_model(self):
        # Evaluate the model
        test_loss, test_acc = self.model.evaluate(self.test_data)
        print(f'Test accuracy: {test_acc * 100:.2f}%')
        print(f'Test loss: {test_loss:.2f}')
        return test_loss, test_acc
    
    def create_model(self):
        self.compile_model()
        self.train_model()
        self.evaluate_model()
        self.model.save(self.modelPath)
        print('Model saved to disk, path: ' + self.modelPath)
        return self.model

    def load_model(self):
        self.model = tf.keras.models.load_model(self.modelPath)
        print('Model loaded')
        return self.model
    
    def predict(self, image_path):
        # Load and preprocess the image
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # Use the loaded model to make predictions
        predictions = self.model.predict(image)

        # Assuming you have only one disease risk prediction in your output
        predicted_label = int(round(predictions[0][0]))

        # You can return the predicted label (0 or 1) or the probability as needed 
        return predicted_label

    def predict_model(self):
        # Predict labels for test images
        predictions = self.model.predict(self.test_data)
        # Assuming you have the list of column names from your dataset
        column_names = df_train.columns[1:]

        predictions_from_test = []
        # Print the predictions for each test image
        for i, prediction in enumerate(predictions):
            # Convert predicted probabilities to 0 or 1
            predicted_labels = [int(round(value)) for value in prediction] 
            predicted_labels_dict = dict(zip(column_names, predicted_labels))
            print(f"Predictions for Test Image {i+1}:")
            print(predicted_labels_dict)
            print()
            predictions_from_test.append(predicted_labels_dict)
        return predictions_from_test
        