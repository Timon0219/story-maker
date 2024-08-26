import os

from dotenv import load_dotenv
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
from IPython.display import Audio
import requests
from gemini_utility import (load_gemini_pro_model,
                            gemini_pro_response,
                            gemini_pro_vision_response,
                            embeddings_model_response)
from gradio_client import Client
# from PyPDF2 import PdfReader
# New Code: Load environment variables from .env file
load_dotenv()

# Accessing the API keys from environment variables
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGING_FACE_AUTH_TOKEN = os.getenv("HUGGING_FACE_AUTH_TOKEN")
NEWS_API_ENDPOINT = 'https://newsapi.org/v2/top-headlines'

working_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="BlackCube News Generator",
    page_icon="🧚‍♀️",
    layout="centered",
)

with st.sidebar:
    selected = option_menu('BlackCube',
                           ['AI NEWS GENERATOR',],
                           menu_icon='robot', icons=['play-fill'],
                           default_index=0
                           )

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

def fetch_news(country, category=None):
    params = {
        'country': country,
        'apiKey': st.secrets["general"]["NEWS_API_KEY"]
    }
    if category:
        params['category'] = category
    response = requests.get(NEWS_API_ENDPOINT, params=params)
    return response.json()


def text2speech(text):
    API_URL = "https://api-inference.huggingface.co/models/facebook/mms-tts-eng"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_AUTH_TOKEN}"}
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        audio_bytes = response.content
        return audio_bytes
    else:
        st.error("Failed to generate audio. Please try again.")
        return None


# comic video generator page
if selected == "AI NEWS GENERATOR":

    st.title("BLACKCUBE NEWS")
    # pdf = st.file_uploader('Upload your PDF Document', type='pdf')

    countries = ['US', 'GB', 'IN', 'CA', 'AU', 'FR', 'DE', 'JP', 'CN', 'RU', 'BR', 'MX', 'IT', 'ES', 'KR']# add more countries as needed
    selected_country = st.sidebar.selectbox('Select a country', countries)

    # Choose the category
    # categories = ['Technology']
    categories = ['Technology','Business', 'Entertainment', 'General', 'Health', 'Science', 'Sports', 'All']
    selected_category = st.sidebar.selectbox('Select a category (optional)', categories)

    # Fetch the news
    if selected_category == 'All':
        news = fetch_news(selected_country)
    else:
        news = fetch_news(selected_country, category=selected_category)

    news_titles = titles = [option['title'] for option in news['articles']]

    selected_news = st.sidebar.selectbox('News', news_titles)
    selected_news_id = news_titles.index(selected_news)
    selected_article = next(option for option in news['articles'] if option['title'] == selected_news)

    # content = selected_article['content']
    content = selected_article['title']
    gemini_response = ''
    if content is not None:
        query = "Rewrite a article add more technical terms, more familiar with the users, the result format should be NEWs, "
        if query:
            model = load_gemini_pro_model()

            # Initialize chat session in Streamlit if not already present
            if "chat_session" not in st.session_state:  
                st.session_state.chat_session = model.start_chat(history=[])
            # Send user's message to Gemini-Pro and get the response
            gemini_response = st.session_state.chat_session.send_message(content + query)  
            st.subheader('BlackCube Article:')
            response = embeddings_model_response(gemini_response.text)
            # client = Client("ADOPLE/Video-Generator-AI")
            # result = client.predict(
            #     gemini_response.text,	# str  in 'Comics Text' Textbox component
            #     api_name="/generate_video"
            # )
            # video_data = result['video']
            # st.video(video_data)
            client = Client("ADOPLE/Text_To_Image")
            image = client.predict(
                    gemini_response.text,	# str in 'parameter_7' Textbox component
                    api_name="/text_to_image"
            )
            print(image)
            st.image(image)
            st.write(gemini_response.text)