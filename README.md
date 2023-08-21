# LIT or NOT on REDDIT

Looking to buy something💰, <br />
but not sure if it's dope?👀 <br />
Let my app be your guide💪, <br />
it's the ultimate Reddit scope!🔎 <br />
No need to waste your money💲, <br />
on something that's not legit👎, <br />
Just use this app <br />
to know if it's worth it👌 <br />
Made with love on streamlit❤, <br />
To help you find if it's LIT or NOT on REDDIT!🔥

## App

[![Streamlit App](<https://static.streamlit.io/badges/streamlit_badge_black_white.svg>)](<https://litornot.streamlit.app//>)

## Project Overview

This app searches reddit posts and comments across many subreddits to determine if a product/service has a positive or negative sentiment and predicts top product mentions using Named Entity Recognition.

If the results did not give you enough information, try phrasing the search term differently and be more specific.
Feel free to increase no. of posts and no. of comments to get more breadth and depth about what redditors think😉

## Blog post

https://medium.com/@anudeepvanjavakam/unlock-the-power-of-reddit-find-out-if-that-product-you-want-to-buy-is-worth-it-with-this-914593998bdf

## Libraries
praw <br />
pandas <br />
nltk==3.8.1 <br />
spacy>=3.0.0,<4.0.0 <br />
en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.0/en_core_web_sm-3.4.0-py3-none-any.whl <br />
python-dotenv <br />
seaborn <br />
matplotlib <br />
wordcloud <br />
plotly <br />
streamlit <br />
streamlit_lottie <br />
datetime <br />
requests <br />
pytorch-lightning==2.0 <br />
torch==1.13.0 <br />
torchtext==0.14.0 <br />
torchvision <br />
torchmetrics>=0.7.0 <br />
datasets==2.3.2 <br />
transformers==4.28.0 <br />
datasets <br />
evaluate <br />
seqeval <br />
promptify==0.1.4 <br />
markupsafe==2.0.1 <br />
jinja2==2.11.3 <br />

## Files


- requirements.txt - all package requirements

- app_ner.py - code for streamlit app with both sentiment analysis and named entity recognition features

- util.py - Helper functions for fetching reddit comments, pre-processing, applying sentiment analysis, generating plots and for inference of the hugging face model (only needed for app_ner.py)

- model.py - Helper functions for fetching reddit comments, pre-processing, applying sentiment analysis, and generating plots (only needed for app.py)

- app.py - code for streamlit app with only sentiment analysis feature

- set_up_notebook.ipynb - a notebook for exploring sentiment analysis of reddit comments

- explore_data_and_model.ipynb - a deep dive into the end-to-end model building process. Gives overview, explores data, discusses strategy and expected solution, metrics, EDA, data pre-processing, modeling, tuning, results, comparison table, conclusion, ideas for improvement, and Acknowledgement 

- demo-2-streamlit-app-2023-05-11-14-05-61.webm - a short clip for app demo

## Problem Statement

We can all agree researching products, platforms or services is painful and time taking.

I often find myself researching multiple sources across numerous chrome tabs to see if something is reliable, worthy, and reasonable to buy and still feel like I am lost. Information is scattered, hard to find, and hard to sort through (multiple mentions of different products, mixed reviews) and get a clear summary.

Moreover, Misleading ads, SEO spam (best 5 product articles with the least research), and fake reviews (Affiliated links, sponsored content) can make the whole experience frustrating. Reddit is one place where people leave honest reviews and discourage any fake ones.

Reddit is one place where people leave honest reviews and discourage any fake ones.
With over 52 million daily active users, Reddit provides a platform for people from all walks of life to share their experiences and opinions.

This app searches posts (sorted by relevancy) and comments (sorted by upvotes) across subreddits to analyze sentiment and predict top product mentions. 
## Features

User choices:
- No. of reddit posts to scrape: More posts take longer time for results. Ex: "10" gets 10 most relevant posts for the  search term. If the no. of posts scraped reaches this limit, then no more comments are scraped regardless of your choice for no. of comments
- No. of comments to scrape: More comments take longer time for results. If the no. of comments scraped reaches this limit, then no more posts are scraped regardless of your choice for no. of posts
- No. of top comments to display: App displays Top comments and their upvotes at the bottom of the page. Comments are sorted by no. of upvotes on reddit.
- Search term (ex: is xxx product worth it? is __ subscription worth it? product/platform/service reviews)
- Include replies: option to include replies. Checking this box may increase repsonse time for results. It is set to False by default.

After clicking the button 'Click me to find out if its Lit' button, app displays the following:
- Percentage of positive or negative sentiment depending on whether the overall percentage of positive words is greater than or less than that of negative words
- No. of posts scraped
- No. of comments scraped
- No. of tokens analyzed
- A bar graph of commonly used positive words and their counts
- A bar graph of commonly used negative words and their counts
- A bar graph of percentage of positive and negative words
- A word cloud showing positive (green), negative (red) and neutral (grey) words
- Top comments and their upvotes
- Top product mentions (predicted by the token classification model)

Text in both original posts and comments is analyzed. If the results did not give you enough information, try phrasing the search term differently and be more specific. Feel free to increase no. of posts and no. of comments to get more breadth and depth about what redditors think.

## Methodology

Sentiment analysis is the process of extracting subjective information from text, such as opinions, emotions, and attitudes.

The best algorithm for analyzing the sentiment of reviews depends on the specific task and data being analyzed. In general, machine learning-based approaches (Naive Bayes, Support Vector Machines (SVM), and Recurrent Neural Networks (RNNs)) tend to perform well on sentiment analysis tasks, especially when trained on large amounts of data. However, lexicon-based approaches can also be effective when dealing with specific domains or languages, particularly for social media opinions and reviews which is what this app focuses on.

Lexicon-based approach: This approach uses pre-defined dictionaries of words that are classified with positive or negative sentiment scores. The algorithm then assigns a score to a given text by summing up the scores of all the words in the text. A popular example of this approach is the VADER (Valence Aware Dictionary and sentiment Reasoner) algorithm, which uses a combination of rule-based and statistical techniques to analyze the sentiment of social media text.

Named Entity Recognition:

Token classification assigns a label to individual tokens in a sentence. One of the most common token classification tasks is Named Entity Recognition (NER). NER attempts to find a label for each entity in a sentence, such as a person, location, or organization.

A pre-trained model DistilBERT (Hugging Face) is finetuned with WNUT17 dataset to detect new entities. This finetuned model is used for inference in the app.
Check out explore_data_and_model.ipynb notebook for more details.

## Why VADER?

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based sentiment analysis tool that is particularly well-suited for analyzing social media texts, including Reddit comments. Here are a few reasons why VADER is a good approach for sentiment analysis of Reddit comments:

-   VADER is optimized for social media language: VADER was specifically designed to handle the nuances of social media language, which can include a lot of slang, misspellings, and other non-standard language that might trip up other sentiment analysis tools. As a result, VADER tends to perform better on social media texts like Reddit comments.

-   VADER uses a more nuanced approach to sentiment analysis: Unlike other sentiment analysis tools that might      simply classify a text as "positive" or "negative," VADER uses a more nuanced approach that takes into account the intensity and polarity of different sentiment words in a text. This allows VADER to better capture the complexity of sentiment in social media texts, which can often be more ambiguous than other forms of text.

-   VADER is open-source and freely available: Unlike some other sentiment analysis tools that require expensive licenses or subscriptions, VADER is open-source and freely available, which makes it a more accessible option for researchers and data analysts who want to perform sentiment analysis on large datasets of Reddit comments.

Overall, VADER's combination of specialized social media language handling, nuanced sentiment analysis, and open-source accessibility make it a strong choice for analyzing the sentiment of Reddit comments.

## Why DistilBERT?

Distilbert aims to optimize the training by reducing the size of BERT and increase the speed of BERT — all while trying to retain as much performance as possible. Specifically, Distilbert is 40% smaller than the original BERT-base model, is 60% faster than it, and retains 97% of its functionality.

## How does it work?

Sentiment Analysis:
-   Preprocessing: The first step is to preprocess the text by removing any noise, such as stop words, punctuations, and special characters. The remaining words are then converted to lowercase to make the analysis case-insensitive.

-   Scoring: The SentimentIntensityAnalyzer then scores each word in the text using a sentiment lexicon, which contains a list of words and their associated polarity scores (positive, negative, or neutral). The lexicon used by VADER is specifically designed to handle sentiment in social media contexts like Twitter and Reddit, and it contains over 7,500 lexical features.

-   Intensity adjustment: VADER then adjusts the scores based on the intensity of the sentiment conveyed by the text. This is done by examining the degree modifiers (such as "very" or "extremely") and negations (such as "not" or "never") in the text.

-   Rule-based heuristics: VADER also uses rule-based heuristics to handle sentiment analysis in contexts where individual words might not be sufficient to determine the overall sentiment. For example, VADER can detect sentiment in phrases like "the bomb" (which can be positive or negative depending on the context), and it can handle emoticons and slang.

-   Aggregation: Finally, VADER aggregates the scores for each word in the text to produce an overall sentiment score between -1 (extremely negative) and 1 (extremely positive).

Named-Entity-Recognition:
- First, a pre-trained model is loaded from the Hugging Face.
    Encoder	models such as ALBERT, BERT, DistilBERT, ELECTRA, RoBERTa can be used for NER (also called token-classification).
    These models are really large. I chose DistilBERT for it's faster training while meeting the functionality and objectives of the project.

- This model is fine tuned with wnut dataset in order to predict product entities.

- Evaluate the model performance on test dataset based on accuracy (why this metric was chosen is explained in the notebook).

- Test with a few example comments from reddit.

- Push the model to Hugging Face hub so it can be used by the streamlit app for inference.

## Limitations

- Lack of context: The VADER algorithm analyzes sentiment based on individual words and phrases, without considering the context in which they are used. This can lead to inaccurate results if the sentiment of a word changes depending on its context.

- Inability to detect sarcasm and irony: VADER's algorithm is not designed to detect sarcasm or irony, which can be prevalent in Reddit comments. This can lead to misinterpretations of sentiment.

- Language limitations: VADER's algorithm is designed for English language sentiment analysis, which means that it may not work as well for other languages or for comments with mixed languages.

- A low F1 score but high accuracy is observed. when it comes to the F1 score, which is a measure of the balance between precision and recall, DistilBERT may sometimes show slightly lower performance compared to the larger BERT model or other more complex models. This is because DistilBERT sacrifices some model capacity to achieve faster inference times and a smaller model size. Consequently, it may have slightly lower precision or recall compared to larger models, leading to a lower F1 score in some cases.

- Class imbalance: The dataset is imbalanced, meaning one class has significantly more samples than the other, the model might have a high accuracy by predicting the majority class correctly most of the time. However, it may struggle with the minority class, resulting in lower recall and F1 score for that class.

## Improvement considerations

- Customized Lexicon: VADER uses a pre-built lexicon of words and their associated sentiment scores. However, this lexicon may not be optimal for certain domains or contexts. Creating a custom lexicon specific to the types of language used in Reddit comments could improve the accuracy of sentiment analysis.

- Multi-Level Sentiment Analysis: VADER currently assigns a single sentiment score to a given text. However, Reddit comments often contain multiple sentiments, sarcasm, or other nuanced language. Incorporating multi-level sentiment analysis could improve the accuracy of VADER's sentiment analysis on Reddit comments.

- Contextual Analysis: Reddit comments are often highly contextual. The same comment may have a different sentiment depending on the context in which it is used. Incorporating contextual analysis techniques, such as topic modeling or named entity recognition, could improve the accuracy of VADER's sentiment analysis on Reddit comments. Named Entity Recognition (NER) could be a particularly useful extension to this app for most mentioned or popular products/services if we do not know exactly what product/service we are searching for. For example, We can search for "Best racquet for tennis beginners" instead of "HEAD Ti6 racquet reviews"

- Machine Learning Techniques: Machine learning techniques, such as deep learning or neural networks, have shown promise in improving the accuracy of sentiment analysis. Incorporating these techniques into VADER's sentiment analysis algorithm could improve its performance on Reddit comments.

- Human Annotation: Finally, having human annotators review a sample of Reddit comments and their associated sentiment scores could provide valuable feedback on the accuracy of VADER's sentiment analysis. This feedback could be used to refine VADER's algorithm and improve its performance on Reddit comments.

- Sentences instead of words: Might have to try sentiment for sentences instead of words. There may be some sentences such a "They try to take away my uncertainity and nervousness" which is obviously positive but the individual words "uncertainity" and "nervousness" end up contributing more to negative score.

- Some posts are deleted but are still fetched and counted towards the no. of posts. These should be excluded.

- GCP for faster computation:
Looping through multiple posts and comments and their nested replies is computationally expensive even with praw (Python Reddit API wrapper). Can use gcp for faster results.

- Examine the data and check how many examples have each tags/token. Model needs more product labels to be trained on.

- Cross-validation (with a stratified split since classes aren’t balanced) may help mitigate some of the problems that come from doing a train/test split with small datasets.

- Class imbalance: there is a class imbalance (model may predict the majority class “O” majority of the time) and needs to be addressed by employing techniques like oversampling the minority class, undersampling the majority class, or using class weighting.

- Consider other evaluation metrics.

- Fine-tuning with more reddit comments labelled with product entities can help the model learn better representations and improve its performance on the task at hand.

- Some functions may be moved to the app script so that they can take advantage of the streamlit cache ability resulting in faster app response time.

- Predicting products can be done with promptify package as shown in the notebook.This does not need a hugging face model and uses chat gpt to predict products and incurs costs based on app usage.

## References

- https://www.learndatasci.com/tutorials/predicting-reddit-news-sentiment-naive-bayes-text-classifiers/
- https://www.nltk.org/
- https://praw.readthedocs.io/en/latest/tutorials/comments.html#extracting-comments-with-praw
- https://stackoverflow.com/questions/61919884/mapping-wordcloud-color-to-a-value-for-sentiment-analysis
- https://www.learndatasci.com/tutorials/sentiment-analysis-reddit-headlines-pythons-nltk/
- https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f
- https://docs.streamlit.io/
