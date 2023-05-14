import streamlit as st
#import streamlit_analytics
import pandas as pd
from nltk import FreqDist

from wordcloud import WordCloud, get_single_color_func
import matplotlib.pyplot as plt

from datetime import datetime
#from io import BytesIO

# If we want to store app analytics in db 
#from google.cloud import firestore 
#from google.oauth2 import service_account

import requests
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

import model

# setting a wide layout
st.set_page_config(layout="wide", page_icon="💬", page_title="LIT OR NOT")

# function for lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url_hello = "https://assets1.lottiefiles.com/packages/lf20_onegrkmr.json"
lottie_url_download = "https://assets10.lottiefiles.com/packages/lf20_zoe5oujy.json"
lottie_hello = load_lottieurl(lottie_url_hello)
lottie_download = load_lottieurl(lottie_url_download)

# In case, we want to store app analytics in google's firebase storage
#key_dict = json.loads(st.secrets["textkey"])
#creds = service_account.Credentials.from_service_account_info(key_dict)
#db = firestore.Client(credentials=creds, project="lit-or-not-on-reddit")

COMMENT_TEMPLATE_MD = """{} - {}
> {}"""

def space(num_lines=1):
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")

with st.sidebar:
    st_lottie(lottie_hello, speed=1, height=200, key="hello_on_side")

st.title('🔥Lit or Not on Reddit🔥')
st.caption('Streamlit App by [Anudeep](https://www.linkedin.com/in/anudeepvanjavakam/)')

st.write('Have you wondered 👀 if a product/platform/service is worth it or find yourself going through endless reddit posts to browse authentic reviews?😞')

st.info("""
Looking to buy something💰, but not sure if it's dope?👀\n
Let my app be your guide💪, it's the ultimate Reddit scope!🔎\n
No need to waste your money💲, on something that's not legit👎,\n
Just use this app to know if it's worth it👌\n
Made with love on streamlit❤,\n
To help you find if it's LIT or NOT on REDDIT!🔥\n
""")

# counts page views, tracks all widget interactions across users
#streamlit_analytics.start_tracking() # add ?analytics=on at the end of the app url to see app analytics

## USER INPUTS ##
st.sidebar.markdown("**Select how many posts & comments you want the app to scrape:** 👇")

no_of_posts = st.sidebar.slider(label = "No. of reddit posts to scrape", min_value=1, max_value=20, value=10, step=1, format=None,
                  key=None, help='More posts take longer time for results. Ex: "10" gets 10 most relevant posts for the search term. If the no. of posts scraped reaches this limit, then no more comments are scraped regardless of your choice for no. of comments', on_change=None, label_visibility="visible")


no_of_comments = st.sidebar.slider(label = "No. of comments to scrape", min_value=1, max_value=500, value=100, step=1, format=None,
                  key=None, help='More comments take longer time for results. If the no. of comments scraped reaches this limit, then no more posts are scraped regardless of your choice for no. of posts', on_change=None, label_visibility="visible")

no_of_top_comments = st.sidebar.slider(label = "No. of top comments to display", min_value=1, max_value=20, value=5, step=1, format=None,
                  key=None, help="App displays Top comments and their upvotes at the bottom of the page", on_change=None, label_visibility="visible")

replies_check = st.sidebar.checkbox(label = "Include replies", value=False,
                            help="Replies are not taken into consideration if this is not checked and only top-level comments are analyzed. Looping through multiple posts and comments and their nested replies is computationally expensive but results may be more accurate",
                            )

search_term = st.sidebar.text_input("**Enter your search term below**👇", placeholder="👉Enter here...") 

button_input = st.sidebar.button("**🔥Click me to find out if its Lit🔥**", type="primary") ## button

with st.sidebar:
    st.info('For example, type "regal unlimited susbcription" or "Saatva Classic mattress reviews"🛌 or "Is ___ worth it?" and click the button above')

if button_input:
    with st_lottie_spinner(lottie_download, speed=1, height=200, key="download"):

        # get best comments from top reddit posts
        comments, top_comments, no_of_posts, no_of_comments = model.get_comments(search_term = search_term, no_of_posts=no_of_posts, no_of_comments=no_of_comments, no_of_top_comments=no_of_top_comments, include_replies=replies_check)

        if no_of_posts == 0:
            st.warning("No posts found! Please enter another search term", icon= "⚠️")
            st.stop()

        # pre process comments and get tokens
        lemmatized_tokens, no_of_tokens = model.pre_process_comments(comments = comments)

        # apply sentiment intensity analyzer
        df = model.apply_sentiment_analyzer(lemmatized_tokens = lemmatized_tokens, threshold = 0.10)
        
        # get percentage of postive and negative words in all the comments 
        sentiment_perc_of_words = df.loc[df['label'] != 0]['label'].value_counts(normalize=True) * 100
        
        # if all words have positive sentiment (100%), add 0% for label '-1'
        if sentiment_perc_of_words[1] == 100:
            sentiment_perc_of_words[-1] = 0

        # if all words have negative sentiment (100%), add 0% for label '1'
        if sentiment_perc_of_words[-1] == 100:
            sentiment_perc_of_words[1] = 0

        perc_of_positive_words = round(sentiment_perc_of_words[1],2)
        perc_of_negative_words = round(sentiment_perc_of_words[-1],2)

        # if search term is not empty
        if search_term!="":

            # if percentage of positive words is greater than that of negative words, it is LIT
            if perc_of_positive_words > perc_of_negative_words:
                st.success(f'🔥LIT!🔥 😀 Positive Sentiment: {perc_of_positive_words}%')
                #st.success(f'Positive Sentiment: {perc_of_positive_words}%')
            else:
                st.info(f'👎NOT SO LIT... 😑 Negative Sentiment: {perc_of_negative_words}%')
                st.info(f'Negative Sentiment: {perc_of_negative_words}%')

        col1, col2, col3 = st.columns(3)
        col1.metric(label = 'No. of posts scraped', value = no_of_posts, delta=None, delta_color="normal", help=None, label_visibility="visible")
        col2.metric(label = 'No. of comments scraped', value = no_of_comments, delta=None, delta_color="normal", help=None, label_visibility="visible")
        col3.metric(label = 'No. of tokens analyzed', value = no_of_tokens, delta=None, delta_color="normal", help=None, label_visibility="visible")                

        # Frequency distribution of the positive and negative words
        frequent_pos_words = FreqDist(df.loc[df['label'] == 1].words)
        frequent_neg_words = FreqDist(df.loc[df['label'] == -1].words)

        if len(frequent_pos_words) == 0:
            st.warning("There are no positive words to display this chart")
        else:            
            # Bar charts for most common postive words
            fig = model.bar_chart_for_freq_words(words_dict = frequent_pos_words, title = 'Commonly Used Positive Words Count', color = 'green', no_of_words = 20)
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        if len(frequent_neg_words) == 0:
            st.warning("There are no negative words to display this chart")
        else:
            # Bar charts for most common negative words
            fig = model.bar_chart_for_freq_words(words_dict = frequent_neg_words, title = 'Commonly Used Negative Words Count', color = 'red', no_of_words = 20)
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        # Bar Chart for postive and negative percentage of words
        fig = model.bar_chart_for_sentiment(perc_of_pos_words = perc_of_positive_words, perc_of_neg_words = perc_of_negative_words)
        st.plotly_chart(fig, use_container_width=True)

        #### word cloud ####

        # positive words are green and negative words are red
        color_words_dict = {
            'green': list(df.loc[df['label'] == 1].words),
            'red': list(df.loc[df['label'] == -1].words)
        }
        # neutral words are grey
        default_color = 'grey'        

        wc = WordCloud(collocations=False, background_color='white').generate_from_frequencies(frequencies=FreqDist(df.words))
        grouped_color_func = model.SimpleGroupedColorFunc(color_words_dict, default_color)
        wc.recolor(color_func=grouped_color_func)
        
        # st.pyplot without these columns would just display the image to fit 100% of the entire column width, hence stretching it.
        # as a workaround, columns can be used to display the plot unstretched 
        col1, col2, col3, col4, col5= st.columns([1, 1, 2, 1, 1])
        with col3:
            st.markdown('**Word cloud to display :green[positive], :red[negative] and neutral words**')               
            plt.subplots(figsize=(5, 4))
            plt.figure()
            plt.imshow(wc, interpolation='bilinear')        
            plt.axis('off')            
            st.pyplot(plt)
        
        # workaround is saving the plot temporarily and getting it with st.image but it quite doesn't look right
        #buf = BytesIO()
        #plt.savefig(buf, format="png")
        #st.image(buf)

        #### end of word cloud ####
        space()
        
        st.subheader('Top comments and their upvotes:')
        st.json(top_comments)

    # celebratory balloons in order after displaying the results
    st.balloons()

st.info("""
        This app searches reddit posts and comments across many subreddits to determine if it has a positive or negative sentiment based on sentiment intensity analyzer (VADER).
        Text in both original posts and comments is analyzed. If the results did not give you enough information, try phrasing the search term differently and be more specific.
        Feel free to increase no. of posts and no. of comments to get more breadth and depth about what redditors think😉
        """)

#  counts page views, tracks all widget interactions across users
#streamlit_analytics.stop_tracking()        