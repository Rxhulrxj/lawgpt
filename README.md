# LawGPT Project

This repository contains a legal assistant application built with Streamlit. The project consists of two main components:
- Data ingestion script (`injest.py`) for processing legal documents
- Main application (`app.py`) for the interactive interface

## Installation Instructions

### 1. Create and Activate Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows(use command prompt)
venv\Scripts\activate


```

### 2. Install Dependencies

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Data Ingestion

To process and ingest the legal documents:

```bash
streamlit run injest.py
```

### Running the Application

To start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501` by default.(it will open by default)

## Requirements
python version i was trained on was 3.10.11
The project requires Python 3.8+ and the following main dependencies:
- streamlit
- langchain
- chromadb
- sentence-transformers
- transformers
- torch
- and other dependencies listed in `requirements.txt`

For a complete list of dependencies and their versions, please refer to the `requirements.txt` file.

## Note

Make sure all environment variables and necessary configurations are properly set up before running the application. If you encounter any issues during installation or execution, please check the error messages and ensure all dependencies are correctly installed.
