# Sentiment Analysis on NYT Comments

## Overview
This Python code performs sentiment analysis on comments sourced from the [NYT Comments dataset](https://www.kaggle.com/datasets/aashita/nyt-comments) available on Kaggle. The dataset has been modified to conduct a polarity analysis on snippets of New York Times (NYT) reports from April 2017.

## Data Loading and Modification
The dataset is loaded using the Pandas library from a CSV file named 'Article.csv'. A single column containing comments, named 'Comments', is selected for analysis.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV
data = pd.read_csv('Article.csv')

# Select only the 'snippet' column for analysis
df = pd.DataFrame(data["snippet"])

# Rename the column to 'Comments'
df.rename(columns={"snippet": "Comments"}, inplace=True)
```

## Sentiment Analysis
Sentiment analysis is performed using the TextBlob library, which calculates the sentiment polarity of each comment on a scale from -1 to 1. The sentiments are then classified as Positive, Neutral, or Negative.

```python
from textblob import TextBlob

# Calculate sentiment polarity and classify as Positive, Neutral, or Negative
df['Polarity'] = df['Comments'].apply(lambda p: TextBlob(str(p)).sentiment.polarity)
df['Sentiment'] = df['Polarity'].apply(lambda s: 'Positive' if s > 0 else ('Neutral' if s == 0 else 'Negative'))
```

## Visualization
The results of the sentiment analysis are visualized using a pie chart, highlighting the distribution of Positive, Neutral, and Negative sentiments among the comments.

```python
# Visualize sentiment distribution in a pie chart
sentiment = df['Sentiment'].value_counts()
color_palette = ['red', (1.0, 0.75, 0.0), 'green']

plt.style.use('classic')
sentiment.plot(kind='pie', title='Polarity Score of Comments', colors=color_palette, autopct='%1.1f%%', wedgeprops=dict(width=0.6))
```

## Word Cloud
A word cloud is generated to visualize the most frequently occurring words in all the comments.

```python
from wordcloud import WordCloud

# Generate word cloud from all comments
text = ''.join(comment for comment in df['Comments'])
wordcloud = WordCloud(background_color='white').generate(text)

plt.figure(figsize=(8, 10))
plt.axis('off')
plt.imshow(wordcloud);
```

Feel free to customize and use this code for your sentiment analysis projects!
