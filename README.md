# QA Bot for PDF Files

This repository contains a QA bot for PDF files. The bot extracts text from PDF files, generates embeddings, and stores them in a vector database. Users can query the system, which retrieves relevant text chunks based on similarity scores. The system also integrates with a GPT model for refining user queries.

## Table of Contents
1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [System Architecture](#system-architecture)
6. [Deployment](#deployment)
7. [Execution](#execution)
8. [Documentation](#documentation)

## Features
- Extract text from PDF files.
- Segment text into chunks.
- Generate embeddings using a Sentence Transformer-based model.
- Store and retrieve embeddings from a vector database (e.g., Faiss).
- Refine user queries using a GPT model.
- Retrieve relevant text chunks based on similarity scores.

## Requirements
- Python 3.8+
- PyMuPDF
- SentenceTransformers
- Faiss
- Hugging Face Transformers
- OpenAI GPT-3/4 API access
- Flask (for API deployment)

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
    pip install -r requirements.txt
    ```

## Usage
### Extract Text from PDF
To extract text from a PDF file, use the following function:
```python
from your_module import extract_text_from_pdf

text = extract_text_from_pdf('path_to_your_pdf.pdf')
print(text)
