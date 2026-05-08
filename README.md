# 🧾 Customer Support Ticket Classification using Transformers

## 📌 Project Overview

This project builds an AI system that automatically classifies customer
support tickets into predefined categories such as **Billing, Technical
Support, and IT Support** using a transformer-based deep learning model.

The goal is to reduce manual effort in tagging tickets and improve response time in customer support systems.

------------------------------------------------------------------------

## 🚀 Problem Statement

Customer support teams handle a large number of tickets daily. 

Manually categorizing these tickets is:
-   Time-consuming
-   Prone to human error
-   Difficult to scale

This project solves this problem by using **Natural Language Processing (NLP)** and **Deep Learning** to automatically classify tickets based on
their text content.

------------------------------------------------------------------------

## 🧠 Model Used

-   Pre-trained Transformer Model: **DeBERTa v3 base**
-   Framework: Hugging Face Transformers
    (https://huggingface.co/docs/transformers)

------------------------------------------------------------------------

## 🤖 How DeBERTa Works

DeBERTa (Decoding-enhanced BERT with Disentangled Attention) is a transformer-based language model developed by Microsoft.

The model processes text using a mechanism called self-attention, where each word looks at other words in the sentence to understand context and meaning.

For example, in the sentence:

"Payment failed during checkout"

the model focuses more on important words like:
- "Payment"
- "Failed"

This helps the model understand that the ticket is related to a billing issue.

During training:
1. The input text is tokenized into numerical representations.
2. Attention masks are created to ignore padding tokens.
3. The transformer model analyzes relationships between words.
4. The classification layer predicts the most appropriate ticket category.

The model learns by comparing predictions with actual labels and updating internal weights using backpropagation.

------------------------------------------------------------------------

##🧠 Attention Mechanism

The model uses self-attention to determine which words are important in a sentence. For example, in “payment failed during checkout”, the model focuses more on “payment” and “failed”.

------------------------------------------------------------------------

## ⚙️ Workflow

1.  Data Loading
2.  Data Cleaning
3.  Label Encoding
4.  Tokenization
5.  Train-Test Split
6.  Model Fine-tuning
7.  Evaluation
8.  Prediction

------------------------------------------------------------------------

## 🔄 Model Pipeline

Raw Ticket Text 
      ↓ 
Label Encoding 
      ↓ 
Tokenization 
      ↓  
Transformer Model(DeBERTa) 
      ↓ 
Training (Fine-tuning)
      ↓ 
Prediction (Category Output)

------------------------------------------------------------------------

## 📊 Example Predictions

  Input Ticket                        Predicted Category
  ----------------------------------- --------------------
  My payment failed during checkout   Billing
  App crashes when I open it          Technical Support
  Need access to company VPN          IT Support

------------------------------------------------------------------------

## 📦 Installation

pip install -r requirements.txt

------------------------------------------------------------------------

## ▶️ How to Run

1.  Open Jupyter Notebook
2.  Run ticket-tagger.ipynb

------------------------------------------------------------------------

## 🛠 Requirements

transformers 
datasets 
torch 
scikit-learn 
pandas 
numpy

------------------------------------------------------------------------

## 📁 Project Structure

customer-support-ticket-classifier/ 
│ 
├── ticket-tagger.ipynb 
├── README.md
├── requirements.txt

------------------------------------------------------------------------

## 📌 Key Highlights

-   Uses state-of-the-art transformer model (DeBERTa)
-   Automates customer support ticket classification
-   Reduces manual effort and improves efficiency
-   End-to-end NLP pipeline implementation

------------------------------------------------------------------------

## 📌 Future Improvements

-   Deploy as a web app using Streamlit
-   Add real-time API for predictions
-   Improve accuracy with hyperparameter tuning
-   Add more ticket categories
