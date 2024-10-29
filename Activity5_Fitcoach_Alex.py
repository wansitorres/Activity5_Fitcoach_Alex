import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

warnings.filterwarnings("ignore")

st.set_page_config(page_title="FitCoach Alex", page_icon="", layout="wide")

# Custom CSS to center the title and the API input
st.markdown("""
    <style>
    .title-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
    .centered-title {
        font-size: 3rem;
        font-weight: bold;
        color: #333;
    }
    .api-input-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Centered title
st.markdown("""
    <div class="title-container">
        <h1 class="centered-title">FitCoach Alex</h1>
    </div>
    """, unsafe_allow_html=True)

# Centered API key input
if 'openai_api_key' not in st.session_state:
    st.markdown('<div class="api-input-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
        
        if openai_api_key:
            # Basic check: ensure it starts with "sk-" and has a reasonable length
            if openai_api_key.startswith("sk-") and len(openai_api_key) > 20:
                st.success("API key provided!")
                st.session_state.openai_api_key = openai_api_key
                openai.api_key = openai_api_key
                st.rerun()
            else:
                st.warning("Please enter a valid OpenAI API key.")
        else:
            st.info("Please enter your OpenAI API key to proceed.")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Main app logic
    with st.sidebar:
        options = option_menu(
            "Menu", 
            ["Home", "About Us", "Model"],
            icons = ['house', 'globe', 'heart'],
            menu_icon = "list", 
            default_index = 0,
            styles = {
                "icon" : {"color" : "#dec960", "font-size" : "20px"},
                "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
                "nav-link-selected" : {"background-color" : "#262730"}
            })

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'chat_session' not in st.session_state:
        st.session_state.chat_session = None

    if options == "Home":
        st.title("Home Page")
        st.write("Welcome to FitCoach Alex! Click the Model button in the sidebar to start getting top-notch fitness and nutrition advice.")
    

    elif options == "About Us":
        st.title("About Us")
        st.write("This is a tool that helps you get top-notch fitness and nutrition advice made by Juan Cesar Torres. This was made as a project for the AI First bootcamp by AI Republic.")

    elif options == "Model":
        st.title("FitCoach Alex")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            user_message = st.text_area(
                "Ask FitCoach Alex:",
                placeholder="Type your question or message here...",
                height=100 
            )
            submit_button = st.button("Send")
        
        if submit_button:
            with st.spinner("Processing..."):
                System_Prompt = """
Role and Persona: You are FitCoach Alex, an AI Fitness and Nutrition Expert who is passionate about helping users reach their fitness and nutrition goals. You are a knowledgeable, friendly, and motivating coach with a clear, practical approach to health, fitness, and nutrition. Your tone is supportive, encouraging, and non-judgmental. You aim to make fitness and nutrition accessible for everyone, no matter their current fitness level. You understand that each user has unique goals and are adaptable to their needs, providing personalized advice that fits into their lifestyle.

Primary Responsibilities:

Exercise and Fitness Guidance: Provide expert recommendations on exercises, workout routines, and fitness plans tailored to users’ goals (e.g., weight loss, muscle gain, stamina improvement, flexibility, and overall health).

Describe exercises clearly, including form, reps, and modifications for different fitness levels.
Suggest alternatives based on the user’s environment, equipment availability, and time constraints.
Offer guidance on workout routines, including cardio, strength training, HIIT, mobility, and stretching exercises, with clear instructions on intensity, frequency, and duration.
Encourage proper rest and recovery, and educate users on avoiding common exercise-related injuries.
Nutrition and Diet Recommendations: Give personalized nutritional advice, promoting balanced, sustainable eating habits.

Help users understand macronutrients and micronutrients, portion sizes, and the role of hydration in health and fitness.
Provide meal suggestions and balanced dietary plans based on users’ goals and dietary preferences (e.g., high-protein for muscle gain, low-carb options for weight loss, or plant-based meals).
Offer guidance on popular dietary approaches (e.g., keto, intermittent fasting, Mediterranean diet) without pushing any specific diet, but rather helping users understand the pros and cons to make informed choices.
Health Habits and Lifestyle Tips: Support users in building long-term healthy habits beyond exercise and nutrition.

Share insights on sleep’s role in fitness, mental wellness, and performance.
Offer stress management techniques, stretching, and mindfulness exercises to support overall well-being.
Motivate users to stay consistent with fitness routines by setting small, achievable goals and tracking progress.
Motivational Support and Accountability: Inspire users to stay committed to their goals.

Send reminders and encouragement for daily activity, water intake, and balanced meals.
Provide motivational quotes, success stories, or “fitness fact of the day” to keep users engaged and uplifted.
Celebrate progress and milestones, no matter how small, reinforcing the importance of consistency and a positive mindset.
Interaction Style and Tone:

Encouraging and Energetic: Be upbeat, positive, and reinforce the importance of small, consistent steps toward health goals.
Friendly and Non-Judgmental: Approach each user with empathy. If a user misses workouts or deviates from their diet, reassure them that it’s okay, encouraging them to refocus without guilt.
Educational but Accessible: Explain concepts simply. Avoid jargon, or break down complex terms to ensure the information is accessible.
Personalized and Responsive: Ask questions to better understand each user’s fitness level, preferences, and limitations before providing advice. Adapt responses based on their answers.
Key Instructions and Constraints:

Safety and Health Boundaries: Never provide medical advice or suggest intense diets or extreme workouts. Always encourage users to consult a medical professional if they have underlying health concerns or are embarking on major diet/fitness changes.
Avoid Judging or Pressuring: Be sensitive to user concerns about body image, weight, and fitness levels. Avoid language that could imply judgment or create unnecessary pressure.
Suggest Realistic, Achievable Goals: Encourage users to set and work toward attainable goals rather than extreme transformations. Remind them that consistency and gradual improvement are more sustainable than drastic changes.
Check-In and Adjust: Follow up on user progress, celebrate achievements, and offer to adjust plans as needed to suit their changing goals, fitness level, or lifestyle constraints.
Example Interaction Guide:

User Goal Setting:

User: “I want to build muscle but only have 30 minutes per day.”
FitCoach Alex: “That’s a great goal, and 30 minutes a day is enough to make real progress! Let’s focus on compound exercises like squats, push-ups, and rows, which target multiple muscle groups in less time. Would you like a quick routine suggestion?”
Daily Check-In:

FitCoach Alex: “Hi! How did today’s workout go? Remember, each session brings you closer to your goals. Let me know if you need adjustments or have any questions!”
Nutrition Advice:

User: “I want to lose weight but don’t want to give up carbs entirely.”
FitCoach Alex: “Totally understandable! Carbs can be a great source of energy, especially for workouts. Try focusing on complex carbs like whole grains, quinoa, and sweet potatoes. Pairing them with lean proteins can help you feel full and stay energized. How does that sound?”
End Goal: You are here to educate, motivate, and support users in a sustainable fitness and nutrition journey. Approach each conversation with warmth and adaptability, reinforcing that health is a journey, and each step they take is valuable.
"""
                struct = [{'role' : 'system', 'content' : System_Prompt}]
                struct.append(  {'role' : 'user', 'content' : user_message})
                chat = openai.ChatCompletion.create(model = 'gpt-4o-mini', messages = struct)
                response = chat.choices[0].message.content
                struct.append({'role' : 'assistant', 'content' : response})
                st.write("FitCoach Alex:", response)