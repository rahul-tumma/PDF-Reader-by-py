# PDF Text Analysis

## Description
This Python script analyzes text extracted from PDF files. It performs various text-related operations such as text summarization, tokenization, extraction of named entities, and sentiment analysis.

## Features
- **Text Summarization**: Summarizes the text extracted from PDF files.
- **Tokenization**: Splits the text into individual words.
- **Named Entity Extraction**: Identifies and extracts named entities (e.g., persons, organizations, locations).
- **Sentiment Analysis**: Analyzes the sentiment of the text using NLTK's VADER lexicon.

## Usage
1. Clone or download the repository.
2. Install the required dependencies:
   ```bash
   pip install PyPDF2 nltk
   ```
3. Run the script by providing the path to the PDF file as an argument:
   ```bash
   python app.py path/to/your/pdf_file.pdf
   ```
4. View the output, which includes the text summary, tokenized words, named entities, and sentiment scores.

## Example
```bash
python app.py 
