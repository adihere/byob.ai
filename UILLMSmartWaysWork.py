import random
import gradio as gr
import time
import requests
import os

#UX stuff
# Get the absolute path to the image file
#background-image: url('https://www.kasandbox.org/programming-images/avatars/spunky-sam.png');
#image_path = os.path.abspath(os.path.join('images', 'botagileAddy.jpg'))

css = """
.gradio-container {    
    color: green;
}
.gradio-title {
    color: yellow;
    font-family: "Comic Sans MS", cursive, sans-serif;
}
"""


def slow_echo(message, history):
    for i in range(len(message)):
        time.sleep(0.3)
        yield "You typed: " + message[: i+1]

def smartways_ai(message, history):
    try:
        # Send a POST request to the Flask API with the user message as JSON data
        response = requests.post('http://127.0.0.1:5000/chatbot', json={'message': message})
        
        # Get the chatbot response from the JSON response
        chatbot_response = response.json()
        
        # Return the chatbot response
        return chatbot_response
    except requests.exceptions.RequestException as e:
        # Handle any exceptions that may occur when sending the POST request
        return "An error occurred while sending the request to the Flask API: " + str(e)
    

#gr.ChatInterface(slow_echo).queue().launch()

gr.ChatInterface(
    smartways_ai,
    chatbot=gr.Chatbot(height=350),
    textbox=gr.Textbox(placeholder="Ask me about agile , ways of working ... lean Kanban scrum and more", container=False, scale=7),
    title="Just another opinionated agile expert!",
    description="UI bot responding to queries on  ways of working - agile devops lean Kanban Scrum ... amazon,spotify ... ",
    theme="soft",
    examples=["Hello agile bot!", "Do we need to have to stand-ups every day?", "What is the agile operating model?","Explain Kanban to a layman","what is spotify model?"],
    cache_examples=True,
    retry_btn=None,
    undo_btn="Delete Previous",
    autofocus=True,
    clear_btn="Clear",
    css=css
).queue().launch()

