from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import _chain  # Import the chain setup from utils


class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# FastAPI App
app = FastAPI()

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        response = _chain.invoke({"question": request.question})
        return QueryResponse(answer=response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
