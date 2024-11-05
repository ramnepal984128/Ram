from fastapi import FastAPI
from transformers import pipeline
from transformers import BertTokenizer, BertForSequenceClassification
app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

# # @app.get("/generate")
# # def first_model():
# #     return "Hello"
    

# def read_text(input_text: str) -> str:
#     reverse = " "
#     for i in input_text:
#         reverse= i + reverse
#     return reverse

# print(read_text("Ram"))


@app.get("/generate")
def first_model(input_text:str) -> str:

    model = BertForSequenceClassification.from_pretrained(r"C:\Users\aasma\bert_sequence\content\bert_sequance")
    tokenizer = BertTokenizer.from_pretrained(r"C:\Users\aasma\bert_sequence\content\bert_sequance")
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    prediction = pipe(input_text)
    mapping= { 
        "LABEL_3": "Extremely Positive"
    }
    return mapping [prediction[0]["label"]]
    
# print(first_model("I am happy today"))
    




