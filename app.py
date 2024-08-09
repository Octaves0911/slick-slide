import gradio as gr
from ai71 import AI71
import faiss
from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi
import numpy as np
from collections import deque
import time
from wordcloud import WordCloud
from sklearn.feature_extraction import text
import matplotlib.pyplot as plt
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

with open(".env", "r") as file:
    api_key = "".join(file.read().splitlines())
client = AI71(api_key)
messages_list = []
embeddings = None
model = None
index = None
context_indices = deque(maxlen=5)

def segregate_transcript(transcript):
    '''
    transcript: transcript of the video
    return: min_segregated_data

    This function takes a transcript of the video and returns a list of dictionaries with start time and text.
    The list is segregated into 60 second intervals.
    '''
    min_segregated_data = []
    start_time = None
    text_data = "" 
    for el in transcript:
        if start_time is None:
            start_time = el['start']
            text_data = el['text']

        elif el['start'] - start_time < 60:
            text_data += el['text']

        else:
            min_segregated_data.append({'start': start_time, 'end': el['start'] + el['duration'], 'text': text_data})
            start_time = el['start']
            text_data = el['text']

    min_segregated_data.append({'start': start_time, 'end': el['start'], 'text': text_data})

    return min_segregated_data


def get_video_subtitile(video_id):
    '''
    url: youtube url
    return: transcript

    This function takes a youtube url and returns the transcript of the video.
    '''

    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    segregated_data = segregate_transcript(transcript)

    return segregated_data

def get_embeddings(text_array):
    '''
    text_array: list of text
    return: embeddings, model

    This function takes a list of text and returns a list of embeddings and a model.
    '''
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(text_array)

    return embeddings, model

def FAISS_Update(embeddings):
    '''
    embeddings: list of embeddings
    return: FAISS document store

    This function takes a list of embeddings and returns a FAISS document store.
    '''

    # Initialize FAISS index
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)

    # Add embeddings to the index
    index.add(np.array(embeddings))

    # Save the index
    faiss.write_index(index, 'document_index.faiss')
    return index

def get_timestamp():
    global subtitle
    global context_indices
    timestamp_data = []
    for i in (context_indices):
        start_sec = subtitle[i]['start'] % 60
        start_min = subtitle[i]['start'] // 60

        end_sec = subtitle[i]['end'] % 60
        end_min = subtitle[i]['end'] // 60

        timestamp_data.append(f"{int(start_min):02d}:{int(start_sec):02d} - {int(end_min):02d}:{int(end_sec):02d} , ")
    return timestamp_data[-3:]

def query_search(index, query, model, k = 5):
    '''
    index: FAISS document store
    query: query
    model: sentence transformer model
    k: number of results
    return: D, I

    This function takes a FAISS document store, a query, a sentence transformer model and a number of results and returns the distance and the index of the results.
    '''
    index = faiss.read_index('document_index.faiss')
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    return D, I

def get_context(indices, min_segregated_data):
    '''
    indices: index of the results
    min_segregated_data: list of dictionaries with start time and text
    return: answer

    This function takes a list of indices and a list of dictionaries with start time and text and returns the answer.
    '''
    context = ""
    for index in indices[0][:3]:
        context += min_segregated_data[index]["text"] + ". "
    return context

def display_video_and_initialize_chat(youtube_url):
    global subtitle
    global text_array
    global embeddings
    global model
    global index

    video_id = youtube_url.split("v=")[1]
    video_embed = f'<iframe width=720 height=500 src="https://www.youtube.com/embed/{video_id}"  allowfullscreen></iframe>'

    start = time.time()
    subtitle = get_video_subtitile(video_id)
    print(f"------------ Loaded Subtitles in {time.time() - start } seconds ----------------")
    text_array = [data["text"] for data in subtitle]
    start = time.time()
    embeddings, model = get_embeddings(text_array)
    print(f"------------ Generated embeddings in {time.time() - start} sec ----------------")
    start = time.time()
    index = FAISS_Update(embeddings)
    print(f"------------ Updated FAISS in {time.time() - start} sec ----------------")

    return video_embed, []

def update_context(query):
    global messages_list
    global index
    global model
    global subtitle
    global context_indices

    start = time.time()
    distance, idx = query_search(index, query, model)
    print(f"---------------------- Vector Search Completed {time.time() - start } sec -------------------------------")
    if len(set(idx[0]).symmetric_difference(set(context_indices))) > 3:
        for i in idx[0][:3]:
            if i not in context_indices:
                context_indices.append(i)

        context = get_context(idx, subtitle)
        print(f"----------- Updating Context --------------")
        messages_list.append({"role": "system", "content": context})

    return messages_list

def get_wordcloud():
    global context_indices
    global subtitle

    if len(context_indices):
        text_data = [subtitle[i]["text"] for i in context_indices]
        texts = " ".join(text_data).split()

    # else :
    #     texts = " ".join(subtitle).split()

    stop_words = list(text.ENGLISH_STOP_WORDS.union(['additional', 'stopwords', 'if', 'needed'])) 
    filtered_text = [word for word in texts if word.lower() not in stop_words and len(word) > 2]

    wordcloud = WordCloud(width=600, height=375, background_color='white').generate(" ".join(filtered_text)).to_image()

    return wordcloud, filtered_text

def generate_word_freq_plot(text):
    # Count word frequencies
    word_counts = Counter(text)
    
    # Plot word frequencies
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
    word, freq = zip(*word_counts.most_common(15))
    ax.barh(word, freq)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Word')
    ax.set_title('Top Word Frequencies')
    plt.gca().invert_yaxis()

    return fig

def handle_chatbot_input(query, history):
    # Simulating a chatbot response
    global messages_list
    global context_indices
    start = time.time()
    messages_list = update_context(query)
    print(f"------------------ Updated Context {time.time() - start } sec ------------------")
    messages_list.append({"role": "user", "content": query}) 
    message = ""
    start = time.time()
    message = client.chat.completions.create(messages=messages_list,model="tiiuae/falcon-180B-chat",# stream=True,
                                     ).choices[0].message.content
    if message[-5:] == "User:":
        message = message[:-6]

    if message[:10] != " I'm sorry":
        print(f"------------ Updating Assistant ----------------")
        messages_list.append({"role": "assistant", "content": message})

    time_stamp = get_timestamp()
    response = f"Falcon: '{message} || Time -> {' '.join(time_stamp)}'"
    history.append(("User: " + query, response))
    print(f"----------------- Got Answer {time.time() - start } sec --------------------")
    start = time.time()
    wordcloud_output, filterd_text = get_wordcloud()
    word_frequency = generate_word_freq_plot(filterd_text)
    print(f"----------------- Generated WordCloud {time.time() - start } sec --------------------")
    #buttons = create_buttons()

    
    return history, history, wordcloud_output, word_frequency

with gr.Blocks(css="""
    .centered {display: flex; justify-content: center; align-items: center; flex-direction: column; text-align: center;}
    .full-width {width: 100%;}
    # .half-width {width: 730px; height: 500px; overflow: hidden;}
    .half-width {width: 720px; height: 600px; overflow: hidden;}
""") as demo:
    
    with gr.Column(elem_classes="centered"):
        gr.Markdown("# SLICK SLIDE", elem_classes="centered")
        youtube_input = gr.Textbox(label="Enter YouTube Link", elem_classes="full-width")
        search_button = gr.Button("Search")
    
    with gr.Row():
        with gr.Column():
            video_output = gr.HTML(label="Video", elem_classes="half-width")
            #time_stamp_display = gr.Textbox(label = "Display Timestamp", value = "Nothing to display")

        with gr.Column():
            chatbot = gr.Chatbot(elem_classes="half-width", height = 500)
            chatbot_input = gr.Textbox(label="Type your Query")
            send_button = gr.Button("Send")

    
    with gr.Row():
        wordcloud_output = gr.Image(label = "Word Cloud")
        word_frequency_map = gr.Plot(label = "Word Frequency map")
    
    gr.on(triggers = [send_button.click, chatbot_input.submit], fn=handle_chatbot_input, inputs=[chatbot_input, chatbot], outputs=[chatbot, chatbot, wordcloud_output, word_frequency_map])
    gr.on(triggers = [search_button.click, youtube_input.submit], fn=display_video_and_initialize_chat, inputs=youtube_input, outputs=[video_output, chatbot])

demo.launch()
    

