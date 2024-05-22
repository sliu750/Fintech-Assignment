**Financial Services Innovation Lab, Programming Task for Summer Research**

This project analyzes SEC-EDGAR 10-K filings for the five Big Tech companies (Amazon, Apple, Google, Meta, and Microsoft), focusing on extracting and presenting insights from the management's discussion and analysis of the firm's performance. 
The goal is to show users what each firm's management thinks of the firm's performance, helping users gauge the company's future prospects and investment potential.

**Overview:**

  **Task 1: Download Data from the SEC-EDGAR**

    - **Downloading Filings**
  
      - Utilizes Python's sec_edgar_downloader package to download 10-K filings
      - Users can select any of the five Big Tech companies to download filings.

  **Task 2: Text Analysis**

    - **Natural Language Processing**
  
      - Uses Python's transformers package, specifically the AutoTokenizer and AutoModelForSequenceClassification based on the ProsusAI/Finbert model, designed for financial sentiment analysis
      - Employs nltk (Natural Language Toolkit) for additional NLP tasks, such as tokenizing text into sentences
      - Extracts key information from the filings (i.e. management's discussion and analysis of financial conditions, operations, and market risks) and classifies those sentences as positive, negative, or neutral
      - Focuses on management's discussion to ensure reasonable running time while maintaining as much core information as possible
  
    - **Visualization**
  
      - Calculates the ratio of positive to total strongly sentimental sentences for each year and visually displays this trend across the years
      - Uses linear regression to predict trends in management's reviews for the next five years
  
    - **Value to Users**
  
      - Provides users with a holistic view of the firm's performance from the management's perspective
      - Helps users understand the management's outlook and make informed decisions regarding investments or purchasing products from the company

  **Task 3: Construct and Deploy Simple App**
    
    - **Local Deployment**
      - Run "streamlit run streamlit_app.py" in terminal.
  
    - **Web Deployment**
      - Access at [https://fintech-assignment.streamlit.app/](https://fintech-assignment.streamlit.app/)

Then, I performed natural language processing and analysis on the files using Python's transformers (specifically the Autotokenizer and AutoModelForSequenceClassification based on the ProsusAI/Finbert model, which is specially trained to analyze sentiments in financial literature)
and nltk (Natural Language Toolkit) packages. Through these large language models (LLMs), I extracted the key information from the filings and classified the sentences in the management's discussion as positive, negative, or neutral. 

I calculated the ratio of positive to total strongly sentimental sentences for each year and visually displayed this trend across the years.
Additionally, I used linear regression to predict trends in management's reviews for the next five years.

Each filing is very large, so to ensure reasonable running time while maintaining as much core information as possible, I analyzed the management's discussion and analysis of financial conditions, operations, and market risk. 
Because the management has a holistic view of the firm's performance, users would probably be interested in seeing the management's appraisal of a firm's performance and its trend over time. 
This helps users understand the management's outlook and make informed decisions regarding investments or purchasing products from the company.

Finally, I deployed my program to an application, both locally and on the web. To run the app locally, enter "streamlit run streamlit_app.py" in terminal. To run it as a web app, go to https://fintech-assignment.streamlit.app/. 
I chose to use Streamlit because it is designed to construct and publish apps for data science and machine learning- Streamlit is easy to use, lightweight, and customizable. For more information, check out https://streamlit.io/. 
Note that the app will take about 10-15 minutes to run due to the large volume of text and the NLP/LLM models running in the backend.

Tools, libraries, and frameworks used: 
- Programming Language: Python, CSS
- IDE: VS Code
- Libraries: sec_edgar_downloader, transformers, nltk, torch, beautifulsoup4, lxml, matplotlib, numpy, Streamlit
