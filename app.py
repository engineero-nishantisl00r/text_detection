import streamlit as st
from transformers import ElectraForSequenceClassification, ElectraTokenizer
import torch

# Load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = ElectraForSequenceClassification.from_pretrained("isln/aitextdetection")
    tokenizer = ElectraTokenizer.from_pretrained("isln/aitextdetection")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Prediction function
def predict_essay(essay, model, tokenizer, threshold=0.005):
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    inputs = tokenizer(essay, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    is_ai_generated = probs > threshold
    return is_ai_generated, probs

# Streamlit UI
st.title("AI Text Detection")
user_essay = st.text_area("Enter your essay:")

if st.button("Check if AI-generated"):
    if user_essay.strip():
        is_ai_generated, probabilities = predict_essay(user_essay, model, tokenizer, threshold=0.005)
        st.write(f"This Essay is AI-generated: {is_ai_generated[0]}")
    else:
        st.write("Please enter an essay.")
