from pydantic import BaseModel, validator
from typing import List
from datetime import datetime, date, time
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fraud_detection_model import predict_fraud
import uvicorn

# Define the Pydantic model for the request body
class TransData(BaseModel):
    transaction_id: int
    trans_type_ID: int
    transaction_amount: int
    trans_date: datetime
    status_type_id: int
    from_acc_id: int
    to_acc_id: int

    @validator('trans_type_ID')
    def validate_trans_type_ID(cls, value):
      trans_type_allowed_values = [1, 2]
      if value not in trans_type_allowed_values:
          raise ValueError(f"must be one of: {trans_type_allowed_values}")
      return value

    @validator('status_type_id')
    def validate_status_type_id(cls, value):
      status_type_allowed_values = [1, 2]
      if value not in status_type_allowed_values:
          raise ValueError(f"must be one of: {status_type_allowed_values}")
      return value


# Initialize FastAPI app
app = FastAPI()

# Define POST endpoint
@app.post("/transaction/validate/")
async def validate_transaction(trans: TransData):
  is_fraudulent = predict_fraud(trans.dict())
  return {"is_fraudulent": bool(is_fraudulent)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
