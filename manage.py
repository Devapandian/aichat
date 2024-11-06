import openai
import numpy as np
import streamlit as st
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity



# MongoDB connection
client = MongoClient("mongodb+srv://petonadmin:SwXzR7C5M0xJGXVz@peton.mxpksaa.mongodb.net/?retryWrites=true&w=majority&appName=PetON")  # Adjust the URI if needed
db = client['PetON']
collection = db['pet-ai-questions']

# Function to generate embeddings using OpenAI API
def generate_embedding(text: str) -> list:
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

# Function to find the most similar question and return its answer
def find_best_matching_answer(user_query: str):
    query_embedding = generate_embedding(user_query)

    if not query_embedding:
        return "Error generating embedding."

    # Query MongoDB for all documents that have 'answer_embedding'
    answers = collection.find({"answer_embedding": {"$exists": True}})

    best_match = None
    highest_similarity = -1  # Cosine similarity ranges from -1 to 1

    # Iterate over documents and calculate cosine similarity between query and stored embeddings
    for answer in answers:
        stored_embedding = answer['answer_embedding']
        
        # Calculate cosine similarity between query embedding and stored embedding
        similarity = cosine_similarity([query_embedding], [stored_embedding])[0][0]
        
        # Update best match if the current one is better
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = answer

    if best_match:
        return best_match['answer_openai']
    else:
        return "Sorry, no suitable answer found."

# Streamlit Interface for Chatbot
st.title("Pet AI")

# To maintain chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# User input
user_query = st.text_input("Ask a question:")

# Process the user input and find the answer
if user_query:
    answer = find_best_matching_answer(user_query)
    
    # Add both user input and AI answer to the chat history
    st.session_state['chat_history'].append({'role': 'user', 'message': user_query})
    st.session_state['chat_history'].append({'role': 'ai', 'message': answer})

# Display the chat history
if st.session_state['chat_history']:
    for chat in st.session_state['chat_history']:
        if chat['role'] == 'user':
            # User messages on the right
            st.markdown(f"<div style='text-align: right; background-color: lightblue; border-radius: 10px; padding: 10px; margin-bottom: 5px;'>{chat['message']}</div>", unsafe_allow_html=True)
        elif chat['role'] == 'ai':
            # AI responses on the left
            st.markdown(f"<div style='text-align: left; background-color: lightgreen; border-radius: 10px; padding: 10px; margin-bottom: 5px;'>{chat['message']}</div>", unsafe_allow_html=True)
