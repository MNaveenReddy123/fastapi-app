from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi import HTTPException
app = FastAPI()

model_path = "./t5_finetuned_qa_large/"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

class QARequest(BaseModel):
    question: str
    context: str

@app.post("/predict")


@app.post("/predict")
def predict_qa(request: QARequest):
    if not request.question or not request.context:
        raise HTTPException(status_code=400, detail="Both question and context are required")
    input_text = f"question: {data.question} context: {data.context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(input_ids)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"answer": answer}
