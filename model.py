
import plotly.express as px
from wordcloud import WordCloud, get_single_color_func
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os

# For gathering data from reddit
import praw  # Python Reddit API Wrapper
from praw.models import MoreComments

import pandas as pd
import nltk
nltk.download('wordnet')
nltk.downloader.download('vader_lexicon')
import en_core_web_sm
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

# for visualization
sns.set(style='whitegrid', palette='Dark2')


# loading development settings for Reddit API
load_dotenv()

# Create a reddit connection with reddit API details
reddit = praw.Reddit(client_id=os.getenv("CLIENT_ID"),
                     client_secret=os.getenv("CLIENT_SECRET"),
                     user_agent=os.getenv("USER_AGENT"))


subreddit = reddit.subreddit('all')

# function to get comments
def get_comments(search_term, no_of_posts=10, no_of_comments=20, no_of_top_comments=5, include_replies=False):
    """function to get comments from reddit posts.

    Args:
        search_term (str): strings to search on reddit
        no_of_posts (int, optional): number of posts to fetch. Defaults to 10.
        no_of_comments (int, optional): number of comments to fetch. Defaults to 20.
        no_of_top_comments (int, optional): number of comments with most upvotes to fetch. Defaults to 5.
        include_replies (bool, optional): set True to include replies. Defaults to False.

    Returns:
        comments_body: list of all comments (including text in posts)
        top_comments_body: a dict of of top comments (sorted by upvotes) and number of upvotes
        len(posts): number of posts fetched
        len(all_comments): number of comments fetched (excluding post texts)
    """
    posts = {}
    all_comments = []
    top_comments_body = {}

    # Search all subreddits and get comments from top posts
    for submission in subreddit.search(search_term, limit=no_of_posts, sort='relevance'):
        # Get comments from each submission
        submission.comments.replace_more(limit=None)

        if include_replies:
            comments = submission.comments.list()
        else:
            comments = submission.comments

        # Exclude stickied posts
        if not submission.stickied:
            posts[submission.id] = submission.selftext
            all_comments += comments

        # If the number of comments exceeds user input, stop searching for more posts
        if len(all_comments) >= no_of_comments:
            break

    # Sort all comments by number of upvotes
    all_comments.sort(key=lambda comment: comment.score, reverse=True)

    # Store top n comments with most upvotes
    top_comments = all_comments[:no_of_top_comments]

    # Store top comments body
    for comment in top_comments:
        top_comments_body[comment.body] = comment.score

    # Store all comments body
    comments_body = [comment.body for comment in all_comments]

    # Add original post text to comments body
    for text in posts.values():
        comments_body.append(text)

    return comments_body, top_comments_body, len(posts), len(all_comments)

# Pre-processing comments
def pre_process_comments(comments):
    """function to tokenize, convert to lower case, remove stop words and lemmatize comments

    Args:
        comments (list): a list of comments (text)

    Returns:
        lemmatized_tokens: a list of pre-processed tokens
        len(lemmatized_tokens): number to pre-processed tokens
    """

    # convert to a string object
    # map to a list of strings
    strings_all = [str(i) for i in comments]

    # join all strings spearated by a commma
    strings_uncleaned = ' , '.join(strings_all)

    # tokenizing and cleaning strings
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|http\S+')
    tokenized_string = tokenizer.tokenize(strings_uncleaned)

    # converting all tokens to lowercase
    tokenized_string_lower = [word.lower() for word in tokenized_string]

    # removing stop words
    nlp = en_core_web_sm.load()
    all_stopwords = nlp.Defaults.stop_words
    tokens_without_stopwords = [
        word for word in tokenized_string_lower if not word in all_stopwords]

    # stemming might return a root word that is not an actual word
    # whereas, lemmatizing returns a root word that is an actual language word.
    # let's normalize words via lemmatizing
    lemmatizer = WordNetLemmatizer()

    lemmatized_tokens = ([lemmatizer.lemmatize(w)
                         for w in tokens_without_stopwords])

    return lemmatized_tokens, len(lemmatized_tokens)

# function to apply sentiment analyzer and get a data frame with scores and labels (based on threshold score) for each word
def apply_sentiment_analyzer(lemmatized_tokens, threshold=0.10):
    """function to apply sentiment analyzer to tokens

    Args:
        lemmatized_tokens (list): a list of tokens
        threshold (float, optional): threshold to categorize compound polarity score as positive or negative. Defaults to 0.10.

    Returns:
        df: a DataFrame with columns neg (0 or 1), neu (0 or 1), pos (0 or 1), compound (polarity score), words, label (0 for neutral, 1 for positive, and -1 for negative)
    """
    # Applying a sentiment analyzer (VADER)
    sia = SIA()
    results = []

    # get polarity score for words
    for word in lemmatized_tokens:
        pol_score = sia.polarity_scores(word)
        pol_score['words'] = word
        results.append(pol_score)

    df = pd.DataFrame.from_records(results)

    # compound score as seen above is a normalized single unideimensional measure of sentiment for a given word
    # adding a label column to denote neutral, positive or negative sentiment with 0, 1 or -1
    df['label'] = 0  # neutral
    df.loc[df['compound'] > threshold, 'label'] = 1  # positive
    df.loc[df['compound'] < -threshold, 'label'] = -1  # negative

    return df

# class for assigning colors to positive and negative words
class SimpleGroupedColorFunc(object):
    """class to assign colors to positive, negative and neutral words
    """

    def __init__(self, color_to_words, default_color):
        """
        Args:
            color_to_words (dict): a dict with colors for positive and negative words as dict keys and a list of words as dict values
            default_color (string): a color for neutral words
        """
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)

# generate bar charts for frequent words
def bar_chart_for_freq_words(words_dict, title, color, no_of_words=20):
    """function to generate a bar chart for frequency of words

    Args:
        words_dict (FreqDist): words and counts. Ex: FreqDist({'happy': 1, 'bad': -1, 'fan': 1, 'cheerlead': 1, 'hat': 0})
        title (string): title for the chart
        color (string): color for the bars
        no_of_words (int, optional): number of most common words to display. Defaults to 20.

    Returns:
        fig: a plotly.graph_objs._figure.Figure
    """
    # Most common positive words
    freq_df = pd.DataFrame(words_dict.most_common(no_of_words))
    freq_df = freq_df.rename(
        columns={0: "Bar Graph of Frequent Words", 1: "Count"}, inplace=False)

    fig = px.bar(freq_df, x="Bar Graph of Frequent Words",
                 y="Count", title=title, text_auto='.2s')
    fig.update_traces(textfont_size=12, textangle=0,
                      textposition="outside", cliponaxis=False, marker_color=color)

    return fig

# generate bar chart for percentage of postive, negative and neutral words
def bar_chart_for_sentiment(perc_of_pos_words, perc_of_neg_words):
    """function to generate a bar chart showing overall percentage of postive and negative words

    Args:
        perc_of_pos_words (int): proportion of positive words
        perc_of_neg_words (int): proportion of negative words

    Returns:
        fig: a plotly.graph_objs._figure.Figure
    """
    colors = ['green', 'red']
    fig = px.bar(x=['positive', 'negative'],
                 y=[perc_of_pos_words, perc_of_neg_words],
                 labels={'x': 'Sentiment', 'y': 'Percentage'},
                 text_auto='.2s',
                 text=[f'{perc_of_pos_words}%', f'{perc_of_neg_words}%'],
                 title="Percentage of positive and negative words")
    fig.update_traces(textfont_size=12, textangle=0,
                      textposition="outside", cliponaxis=False, marker_color=colors)

    return fig
