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

The model is fine-tuned for a **text classification task**, where it learns to map ticket text to specific categories.
The model uses self-attention to understand relationships between words in a ticket.

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

## 🧪 Training Details

-   Batch Size: 8\
-   Learning Rate: 2e-5\
-   Epochs: 10\
-   Optimizer: AdamW\
-   Loss Function: Cross-Entropy Loss

------------------------------------------------------------------------

## 📈 Evaluation

The model is evaluated using: 
- Accuracy 
- Precision 
- Recall 
- F1-score

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
