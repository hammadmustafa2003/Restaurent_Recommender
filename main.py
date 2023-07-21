from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle


# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware to handle OPTIONS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model('saved_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('unique_labels.pkl', 'rb') as handle:
    unique_labels = pickle.load(handle)
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(['Location', 'Cuisine', 'Money'])

# Preprocess the input sentence and entity
def preprocess_input(sentence, entity):
    input_text = sentence + ' [SEP] ' + entity
    input_sequences = tokenizer.texts_to_sequences([input_text])
    input_sequences = pad_sequences(input_sequences, padding='post', maxlen=15)
    return input_sequences

# Function to predict the entity
def get_entity(sentence, entity):
    input_sequences = preprocess_input(sentence, entity)
    predictions = model.predict(input_sequences)
    predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]
    entities = ['Location', 'Cuisine', 'Money']
    predicted_entity = unique_labels[predicted_class_index]
    return predicted_entity

# API endpoint to get the next entity
class EntityRequest(BaseModel):
    sentence: str
    entity: Optional[str] = None

class EntityResponse(BaseModel):
    predicted_entity: Optional[str]
    next_message: str
    next_entity: Optional[str]

@app.post("/get_entity/")
def get_next_entity(req: EntityRequest):
    sentence = req.sentence
    entity = req.entity

    if entity is None:
        # First call, return welcome message
        return EntityResponse(predicted_entity=None, next_message="Welcome to our restaurant recommendation system. Could you please tell me your location?", next_entity="Location")

    words = sentence.strip().split()
    next_entity = None

    if len(words) == 1:
        # Second call with single word response
        next_entity = entity
        if entity == "Location":
            next_message = "What type of cuisine are you in the mood for? Italian, Chinese, Indian, or something else?"
            next_entity = "Cuisine"
        elif entity == "Cuisine":
            next_message = "What's your budget for this meal?"
            next_entity = "Money"
        elif entity == "Money":
            next_message = "Thank you! Now let me find the best restaurants for you."
            next_entity = None
        else:
            raise HTTPException(status_code=400, detail="Invalid entity")

        return EntityResponse(predicted_entity=words[0], next_message=next_message, next_entity=next_entity)
    else:
        # Response contains more than one word, pass it through the model
        predicted_entity = get_entity(sentence, entity)
        if entity == "Location":
            print(predicted_entity.lower(),sentence.lower())
            if predicted_entity.lower() not in sentence.lower():
                return EntityResponse(predicted_entity=None, next_message=f"Either your {entity.lower()} is invalid or our service is not available in your area. Please re-enter {entity.lower()}.", next_entity=entity)
            else:
                next_message = "What type of cuisine are you in the mood for? Italian, Chinese, Indian, or something else?"
                next_entity = "Cuisine"
                return EntityResponse(predicted_entity=predicted_entity, next_message=next_message, next_entity=next_entity)
        elif entity == 'Cuisine':
            print(predicted_entity.lower(),sentence.lower())
            if predicted_entity.lower() not in sentence.lower():                
                return EntityResponse(predicted_entity=None, next_message=f"Either your {entity.lower()} is invalid or we don't have record of restaurents that provide this. Please re-enter {entity.lower()}.", next_entity=entity)
            else:
                next_message = "What's your budget for this meal?"
                next_entity = "Money"
                return EntityResponse(predicted_entity=predicted_entity, next_message=next_message, next_entity=next_entity)
        elif entity == 'Money':
            print(predicted_entity.lower(),sentence.lower())
            if predicted_entity.lower() not in sentence.lower():
                return EntityResponse(predicted_entity=None, next_message=f"Either your {entity.lower()} is invalid or we don't have record of restaurents that provide meal in this budget. Please re-enter {entity.lower()}.", next_entity=entity)
            else:
                next_message = "Thank you! Now let me find the best restaurants for you."
                next_entity = None
                return EntityResponse(predicted_entity=predicted_entity, next_message=next_message, next_entity=next_entity)

# API endpoint for restaurant recommendation

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Load the CSV data
df = pd.read_csv('data/restaurants_data_analysis.csv')
df = df.dropna(subset=['latitude', 'longitude', 'main_cuisine', 'budget'])

res_model = tf.keras.models.load_model('restaurant_recommendation_model.h5')

with open('labelEncoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Tokenize and vectorize the data (using the same tokenizer as before)
res_tokenizer = Tokenizer()
res_tokenizer.fit_on_texts(df['main_cuisine'])

# Function to get multiple restaurant recommendations
def get_restuarents(location, cuisine_preference, budget_constraint, num_recommendations=5):
    user_input = cuisine_preference
    user_sequence = res_tokenizer.texts_to_sequences([user_input])
    user_data = pad_sequences(user_sequence, maxlen=3)  # Assuming the input sequence length is 22 (same as during training)

    # Predict the user's preferences using the loaded model
    user_preferences = np.array(res_model.predict(user_data))

    # Get the index of the predicted category with the highest probability
    predicted_category_index = tf.argmax(user_preferences, axis=1).numpy()[0]

    # Get the predicted category using the label encoder
    predicted_category = label_encoder.inverse_transform([predicted_category_index])[0]

    # Filter restaurants based on the predicted category
    filtered_restaurants = df[df['main_cuisine'] == predicted_category]

    filtered_restaurants_data = filtered_restaurants['main_cuisine'] + ' ' + filtered_restaurants['budget'].astype(str)
    filtered_sequence = res_tokenizer.texts_to_sequences(filtered_restaurants_data)
    filtered_data = pad_sequences(filtered_sequence, maxlen=3)
    filtered_predictions = res_model.predict(filtered_data)
    filtered_predictions = np.array(filtered_predictions)
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(user_preferences, filtered_predictions)

    # Get the index of restaurants with highest similarity
    top_indices = cosine_sim.argsort()[0][-num_recommendations:][::-1]

    # Create a list to store the recommended restaurants and their Google Maps links
    recommended_restaurants = ""

    for index in top_indices:
        restaurant_name = filtered_restaurants.iloc[index]['name']
        latitude = filtered_restaurants.iloc[index]['latitude']
        longitude = filtered_restaurants.iloc[index]['longitude']

        # Construct the Google Maps link
        google_maps_link = f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"

        # Combine the restaurant name and Google Maps link
        restaurant_info = f"{restaurant_name} - <a href=\"{google_maps_link}\" target=\"_blank\" > View on Google Maps</a> <br>"
        recommended_restaurants = recommended_restaurants + restaurant_info

    return recommended_restaurants




class RecommendRequest(BaseModel):
    location: str
    cuisine: str
    budget: int

@app.post("/recommend/")
def get_recommendations(req: RecommendRequest):
    location = req.location
    cuisine = req.cuisine
    budget = req.budget

    # Call the function to get restaurant recommendations based on location, cuisine, and budget
    recommended_restaurants = get_restuarents(location, cuisine, budget)
    return {"recommendation": recommended_restaurants}
