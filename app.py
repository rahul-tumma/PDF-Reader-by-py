import PyPDF2
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from heapq import nlargest
from nltk import pos_tag, ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

def read_pdf(pdf_file):
    text = ""
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

def preprocess_text(text):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words("english"))
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]

    return words

def summarize_text(text, num_sentences=3):
    # Tokenize text into sentences
    sentences = sent_tokenize(text)

    # Calculate word frequency
    words = preprocess_text(text)
    word_freq = FreqDist(words)

    # Rank sentences based on the sum of frequencies of their words
    ranking = {}
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                if i in ranking:
                    ranking[i] += word_freq[word]
                else:
                    ranking[i] = word_freq[word]

    # Get top-ranked sentences
    top_sentences_index = nlargest(num_sentences, ranking, key=ranking.get)
    summary = [sentences[j] for j in sorted(top_sentences_index)]
    return ' '.join(summary)

def extract_named_entities(text):
    sentences = sent_tokenize(text)
    named_entities = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        pos_tags = pos_tag(words)
        entities = ne_chunk(pos_tags)
        for entity in entities:
            if hasattr(entity, 'label'):
                named_entities.append(' '.join([word for word, tag in entity.leaves()]))
    return named_entities

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    return sentiment_scores

if __name__ == "__main__":
    pdf_file = 'testing.pdf'  # Replace 'your_pdf_file.pdf' with the path to your PDF file
    pdf_text = read_pdf(pdf_file)

    # Summarize text
    summary = summarize_text(pdf_text)
    print("Summary:")
    print(summary)

    # Preprocess text and tokenize words
    words = preprocess_text(pdf_text)

    # Calculate word frequencies
    word_freq = FreqDist(words)
    print("\nMost Common Words:")
    print(word_freq.most_common(10))  # Display the 10 most common words

    # Extract named entities
    named_entities = extract_named_entities(pdf_text)
    print("\nNamed Entities:")
    print(named_entities)

    # Perform sentiment analysis
    sentiment_scores = analyze_sentiment(pdf_text)
    print("\nSentiment Scores:")
    print(sentiment_scores)
