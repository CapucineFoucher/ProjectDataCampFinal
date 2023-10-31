from model import Model


def createAndSave():
    # Create an instance of the model
    model = Model()

    model.create_model()

    """model.load_model()

    image_path = 'data/Test_Set/Test/1.png'  # Replace with the actual path to your image
    prediction = model.predict(image_path)

    print(f'Prediction for image 1.png: Disease Risk = {prediction}')"""