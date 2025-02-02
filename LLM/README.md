# QA Bot for PDF Files


This repository contains a QA bot for PDF files. The bot extracts text from PDF files, generates embeddings, and stores them in a vector database. Users can query the system, which retrieves relevant text chunks based on similarity scores. The system also integrates with a GPT model for refining user queries.
### Github code link: https://github.com/nitishsati82/nagp_llm/tree/main/LLM
### Assignment video recording link: https://nagarro-my.sharepoint.com/:v:/p/nitish_sati/Eb2wgBznoBlLvXmcwPI0A5gBDsFEkmhIIESBGG37_7DiDw?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=vGT6mf
## Table of Contents
1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [System Architecture](#system-architecture)
6. [Execution](#execution)

## Features
- Extract text from PDF files.
- Segment text into chunks.
- Generate embeddings using a Sentence Transformer-based model.
- Store and retrieve embeddings from a vector database (PineCone).
- Refine user queries using a GPT model.
- Retrieve relevant text chunks based on similarity scores.

## Requirements
- Python 3.8+
- PyMuPDF
- SentenceTransformers
- PineCone (Api key and index)
- Hugging Face Transformers
- GPT 2
- Google Colab

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/qa-bot-pdf.git
    cd qa-bot-pdf
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
        # Install necessary libraries
        !pip install PyMuPDF sentence-transformers pinecone-client
        !pip install transformers

## Usage
### Extract Text from PDF
To extract text from a PDF file, use the following function:
```python
text = extract_text_from_pdf('path_to_your_pdf.pdf')
```
print(text)

### Segment Text into Chunks
To generate segment text into chunks below method is used:
```python
text_chunks = chunk_text_by_sentence(text)
```
### Generate Embeddings
To generate embeddings from text chunks, use the following function:
```python
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = generate_embeddings(text_chunks)
```
### Store Embeddings
To store and retrieve embeddings, use the following function:
```python
result = save_embedding(query)
```
### Retrieve Embeddings
To Retrieve embeddings, use the following function:
```python
result = save_embedding(query)
```
### Integrate GPT model
To integrate GPT model, use the following code:
from transformers import GPT2LMHeadModel, GPT2Tokenizer
##### Load pre-trained GPT-2 model and tokenizer
```python
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
gpt_model = GPT2LMHeadModel.from_pretrained(model_name)
```
### Query the Vector Database and Get refine query to perform search
To query integrate GPT model and vector db,  use the following functions:
##### To refine query using gpt model below function can be used:
```python
refine_query_more(userQuery)
```
##### To query vector db below function can be used:
```python
query_pinecone(userQuery)
```

## System Architecture
The system architecture consists of the following components:
1. User Query (Natural Language)
2. Text Extraction from PDF (PyMuPDF)
3. Text Chunking and Embeddings (Sentence Transformer-based Model)
4. Vector Database (PineCone)
5. Similarity Matching and Retrieval (Retrieve Relevant Text Chunks)
6. Query Refinement (GPT Model)
7. Final Answer to User


## Execution
1. Extract text from PDFs and generate embeddings
```python
text = extract_text_from_pdf('path_to_your_pdf.pdf')
text_chunks = chunk_text_by_sentence(text)
embeddings = generate_embeddings(text_chunks)
```
2. Refine user queries / retrieve relevant text chunks and analyze the result:
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def run_search_query(query):
    # Use the query_pinecone function to get results
    result = query_pinecone(query)
    # Assuming result['matches'] is a list of ScoredVector objects
    if 'matches' in result:
        return result['matches']  # Return all matches for inspection
    else:
        return []

def analyze_results(matches):
    # Extract scores
    scores = [match['score'] for match in matches]

    # Plot scores
    plt.figure(figsize=(10, 6))
    plt.plot(scores, marker='o', linestyle='-', color='b')
    plt.title('Match Scores')
    plt.xlabel('Match Index')
    plt.ylabel('Score')
    plt.show()

    # Print top 5 matches
    top_matches = sorted(matches, key=lambda x: x['score'], reverse=True)[:5]
    for match in top_matches:
        print(f"ID: {match['id']}, Score: {match['score']}")

    # Calculate average score
    average_score = np.mean(scores)
    print(f"Average Score: {average_score}")

refined_query = refine_query(query)
matches = run_search_query(refined_query)
analyze_results(matches)
```