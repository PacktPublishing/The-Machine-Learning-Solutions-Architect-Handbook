import logging
import os
import sys
import json
 
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer
 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

MAX_LEN = 315

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model = BertForSequenceClassification.from_pretrained(model_dir)
    
    return loaded_model.to(device)

def input_fn(request_body, request_content_type):
    
    if request_content_type == "application/json":
        data = json.loads(request_body)
        print("================ input sentences ===============")
        print(data)
        
        if isinstance(data, str):
            data = [data]
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
            pass
        else:
            raise ValueError("Unsupported input type. Input type can be a string or an non-empty list. \
                             I got {}".format(data))
                       
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        
        input_ids = [tokenizer.encode(x, add_special_tokens=True) for x in data]
        
 
        # pad shorter sentence
        padded =  torch.zeros(len(input_ids), MAX_LEN) 
        for i, p in enumerate(input_ids):
            padded[i, :len(p)] = torch.tensor(p)
     
        # create mask
        mask = (padded != 0)
 
        return padded.long(), mask.long()
    raise ValueError("Unsupported content type: {}".format(request_content_type))
 
 
def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
 
    input_id, input_mask = input_data
    input_id = input_id.to(device)
    input_mask = input_mask.to(device)

    with torch.no_grad():
        y = model(input_id, attention_mask=input_mask)[0]
        print("=============== inference result =================")
        print(y)
    return y
