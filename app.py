import cv2
from sklearn.cluster import KMeans
import webcolors
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
from flask import Flask, request, render_template, jsonify
import pandas as pd
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.models import Sequential
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
import os
from tqdm import tqdm
import pickle
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


app = Flask(__name__)




# Load  Model

# Load the pre-trained model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Load feature list and filenames
feature_list = np.load('C:/Users/Abdelrhman/OneDrive/Desktop/api ml/embeddings_0.npy')
filenames = np.load('C:/Users/Abdelrhman/OneDrive/Desktop/api ml/filenames_0.npy')

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors(features.reshape(1, -1))
    return indices

@app.route("/recommend_clothes", methods=["POST"])
def recommend_clothes():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    img_path = "temp.jpg"  # Save the uploaded image temporarily
    file.save(img_path)

    features = extract_features(img_path, model)
    print("Shape of extracted features:", features.shape)
    print("Shape of feature list:", feature_list.shape)
    indices = recommend(features, feature_list)  # Corrected line

    recommended_images = [filenames[idx] for idx in indices[0]]

    return jsonify({'recommendations': recommended_images})


####################
####################



def extract_colors(image, num_colors=5):
    # Convert image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Flatten the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)
    # Perform color clustering using K-means
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    # Get the RGB values of the cluster centers
    colors_rgb = kmeans.cluster_centers_
    # Convert the RGB values to integers
    colors_rgb = colors_rgb.astype(int)
    # Return the recommended colors as RGB values
    return colors_rgb

def extract_preferred_colors(image, num_colors=5):
    colors = extract_colors(image, num_colors)
    preferred_colors = select_unique_preferred_colors(colors)
    return [(list(rgb), color_name) for rgb, color_name in preferred_colors]


def map_rgb_to_color_name(rgb):
    differences = {}
    for color_hex, color_name in webcolors.CSS3_HEX_TO_NAMES.items():
        r, g, b = webcolors.hex_to_rgb(color_hex)
        differences[sum([(r - rgb[0]) ** 2,
                         (g - rgb[1]) ** 2,
                         (b - rgb[2]) ** 2])] = color_name
    return differences[min(differences.keys())]

def select_unique_preferred_colors(colors):
    preferred_colors = set()
    unique_colors = []

    for color in colors:
        rgb = tuple(color.tolist())
        color_name = map_rgb_to_color_name(rgb)

        if rgb not in preferred_colors and color_name not in [name for _, name in unique_colors]:
            unique_colors.append((rgb, color_name))
            preferred_colors.add(rgb)

        if len(unique_colors) == 5:
            break

    return unique_colors


## load model

#################
###########
@app.route('/', methods=['GET'])
def index():
    return "hello"
@app.route('/api/data',methods=['GET'])
def get_data():
    # Example data to be sent to the frontend
    data = {'message': 'Hello from Flask!'}
    return jsonify(data)

import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv(r'C:\Users\Abdelrhman\OneDrive\Desktop\api ml\final_test.csv')
df.head()
df.info()
sns.scatterplot(x=df['age'], y=df['weight'])
df.drop(df[df['age']>80].index,inplace = True)
sns.scatterplot(x=df['age'], y=df['weight'])
df.drop(df[df['age']<10].index,inplace=True)
sns.scatterplot(x=df['age'], y=df['weight'])
sns.scatterplot(x=df['height'], y=df['weight'])
df.drop(df[df['weight']<40].index,inplace=True)
sns.scatterplot(x=df['height'], y=df['weight'])
df.drop(df[df['height']>190].index, inplace = True)
sns.scatterplot(x=df['height'], y=df['weight'])
df.describe()
df.isna().sum()
df["bmi"] = df["height"]/df["weight"]
df["weight_squared"] = df["weight"] * df["weight"]
df['size'] = df['size'].map({"XXS": 1,
                                     "S": 2,
                                     "M" : 3,
                                     "L" : 4,
                                     "XL" : 5,
                                     "XXL" : 6,
                                     "XXXL" : 7})
df.head()
df.describe()
X = df.drop('size' , axis = 1)
y = df['size']
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 )
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
df_pipeline = Pipeline(steps =[('missing-value', SimpleImputer(strategy ='median'))])
df_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
rf_param_grid = {
    "modeling__n_estimators": np.arange(10, 100, 10),
    "modeling__max_depth": [None, 3, 5, 10],
    "modeling__min_samples_split": np.arange(2, 20, 2),
    "modeling__min_samples_leaf": np.arange(1, 20, 2),
    "modeling__max_features": [0.5, 1, 'sqrt', 'auto'],
    "modeling__max_samples": [10000]
}

# Initialize RandomForestClassifier within the pipeline
final_pipeline = Pipeline(steps=[
    ('df_pipeline', df_pipeline),
    ('modeling', RandomForestClassifier(max_depth=9))  # Default parameter for now
])
final_pipeline.fit(x_train, y_train)
rs_model = RandomizedSearchCV(
    final_pipeline,
    param_distributions=rf_param_grid,
    n_iter=10,
    cv=5,
    verbose=2,
    n_jobs=-1
)

# Fit RandomizedSearchCV to the training data
rs_model.fit(x_train, y_train)

# Print best parameters
print("Best parameters found:")
print(rs_model.best_params_)

# Get best model
best_model = rs_model.best_estimator_

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    weight = float(data.get('weight'))
    age = float(data.get('age'))
    height = float(data.get('height'))

    bmi = height / weight
    weight_squared = weight * weight

    # Create a DataFrame with the user input and additional features
    user_data = pd.DataFrame({
        "weight": [weight],
        "age": [age],
        "height": [height],
        "bmi": [bmi],
        "weight_squared": [weight_squared]
    })

    # Make predictions using the loaded model
    predictions = best_model.predict(user_data)
    prediction = int(predictions[0])
    def transfer(prediction):
      if prediction == 1:
        return 'XS'
      elif prediction == 2:
        return 'S'
      elif prediction == 3:
        return 'M'
      elif prediction == 4:
        return 'L'
      elif prediction == 5:
        return 'XL'
      elif prediction == 6:
        return 'XXL'
      else:
        return 'XXXL'
    # Return the prediction as a JSON response
    response = {'prediction': transfer(prediction)}

    return jsonify(response)


@app.route("/color-extraction", methods=["POST"])
def color_extraction():
    # Check if an image file was uploaded
    if "file" not in request.files:
        return jsonify({"error": "No image file found"}), 400

    file = request.files["file"]

    # Read uploaded image
    image = np.array(Image.open(io.BytesIO(file.read())).convert("RGB"))

    # Extract preferred colors
    preferred_colors = extract_preferred_colors(image)

    # Prepare response
    response = {
        "preferred_colors": preferred_colors
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
