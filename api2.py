from fastapi import FastAPI
from transformers import pipeline
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch
app = FastAPI()

@app.get("/generate_pipe")
def first_model(input_text:str):
    # Load the tokenizer and model
    try:
        
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        prediction = pipe(input_text)
        return prediction[0]["label"]
    except Exception as e:
        print(f"Error Occured: {e}")
        return {"error": str(e)}

@app.get("/generate_nopipe")
def first_model_nopipeline(input_text:str):
    # Load the tokenizer and model
    try:

        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        inputs= tokenizer(input_text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        logits =outputs.logits
        probabilities =F.softmax(logits, dim=-1)

        sentiment_score= torch.argmax(probabilities, dim=1).item()

        sentiment_labels= {
            0:"1 star",
            1:"2 star",
            2:"3 star",
            3:"4 star",
            4:"5 star"
        }

        sentiment = sentiment_labels.get(sentiment_score,"Unknown")
        return sentiment

    except Exception as e:
        print(f"Error Occured: {e}")
        return {"error": str(e)}

