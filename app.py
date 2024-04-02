import streamlit as st
import os
from dotenv import load_dotenv,find_dotenv
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent,Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
import json
from PIL import Image
import requests
from bs4 import BeautifulSoup
import re

load_dotenv(find_dotenv())
serpapi = os.getenv("SERPAPI_API_KEY")
api=os.getenv("OPENAI_API_KEY")
st.set_page_config(
    page_title="Lyzr Tweet generator Agent",
    layout="centered",  # or "wide"
    initial_sidebar_state="auto",
    page_icon="lyzr-logo-cut.png",
)

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Lyzr Tweet generator Agent")
st.markdown("### Welcome to the Lyzr Tweet generator Agent!")


open_ai_text_completion_model = OpenAIModel(
    api_key=api,
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)


def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query,
        "gl": "in",
    })

    headers = {
        'X-API-KEY': serpapi,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    res = response.json()
    # Print the response JSON object to inspect its structure

    mys = []
    for item in res.get('organic', []):
        mys.append(item.get('link'))
    return mys


def extract_text_from_url(url):
    try:
        # Fetch HTML content from the URL
        response = requests.get(url)
        response.raise_for_status()

        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text content and replace consecutive spaces with a maximum of three spaces
        text_content = re.sub(r'\s{4,}', '   ', soup.get_text())

        return text_content

    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from {url}: {e}")
        return None


def extracteddata(query):
    result =  search(query)
    my_data = []
    for i in result:
        get_data = extract_text_from_url(i)
        my_data.append(get_data)
    return my_data[:6]


topic = st.text_input("Enter Tweet Topic: ")

def tweet_generator(topic):
    data = extracteddata(topic)

    twitter_agent = Agent(
        role="Tweet Expert",
        prompt_persona=f"""You are a word class journalist and tweet influencer.
                write a viral tweeter thread about {topic} using {data} and following below rules:
                """
    )

    task1 = Task(
        name="Tweet Generator",
        model=open_ai_text_completion_model,
        agent=twitter_agent,
        instructions=f"""write a viral tweeter thread about {topic} using {data} and following below rules:
                1/ The thread is engaging and informative with good data.
                2/ The thread needs to be around than 3-5 tweet.
                3/ The thread need to address {topic} very well.
                4/ The thread needs to be viral and atleast get 1000 likes.
                5/ The thread needs to be written in a way that is easy to read and understand.
                6/ Output is only threads no any other text apart from thread"""
    )
    output = LinearSyncPipeline(
        name="Tweet Pipline",
        completion_message="pipeline completed",
        tasks=[
            task1,
        ],
    ).run()

    return output[0]['task_output']


if st.button("Get Tweets"):
    tweets = tweet_generator(topic)
    st.markdown(tweets)

    with st.expander("ℹ️ - About this App"):
        st.markdown("""
        This app uses Lyzr Automata Agent to Generate Tweet basd on your entered topic. For any inquiries or issues, please contact Lyzr.

        """)
        st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width=True)
        st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width=True)
        st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width=True)
        st.link_button("Slack",
                       url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw',
                       use_container_width=True)

