import numpy as np
import torch
import random
import os
from transformers import AutoModel, AutoTokenizer
import argparse
from datasets import load_dataset
import json
import pandas as pd

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def load_statcodesearch(jsonl_path='./statcodesearch/test_statcodesearch.jsonl'):
    # List to store each row's data as a dictionary with specified column names
    rows = []

    with open(jsonl_path, 'r', encoding='utf-8') as file:
        # Read each line in the jsonl file
        for line in file:
            # Parse JSON line to a dictionary
            data = json.loads(line.strip())
            
            # Check if the "input" field exists
            if "input" in data:
                # Split the input at "[CODESPLIT]"
                split_data = data["input"].split("[CODESPLIT]", 1)
                
                # Check that the split resulted in exactly two parts
                if len(split_data) == 2:
                    # Create a dictionary for this row
                    row = {
                        "func_documentation_tokens": split_data[0].strip(),
                        "func_code_tokens": split_data[1].strip()
                    }
                    rows.append(row)
                else:
                    print("Warning: 'input' field does not split into exactly two parts:", data["input"])
            else:
                print("Warning: 'input' field not found in line:", data)

    # Create DataFrame from rows
    df = pd.DataFrame(rows)

    return df

def create_embs(input_file, type, ckp, args):
    # Select the appropriate input based on type
    if type == "doc":
        inputs = [' '.join(i).strip() for i in input_file['func_documentation_tokens']]
    elif type == "code":
        inputs = [' '.join(i).strip() for i in input_file['func_code_tokens']]
    else:
        raise ValueError("Invalid type. Must be 'doc' or 'code'.")

    tokenizer = AutoTokenizer.from_pretrained(ckp, trust_remote_code=True)
    model = AutoModel.from_pretrained(ckp, trust_remote_code=True)
    
    if args.is_finetuned:
        model.load_state_dict(torch.load(f'./models/codebert_{args.lang}.pth'))
    
    model.to(args.device)
    
    representations = []
    # Process each input separately
    for input_text in inputs:
        # Tokenize without padding
        encoded_input = tokenizer(input_text, padding=False, truncation=True, max_length=256, return_tensors="pt")
        
        # Move input to the same device as the model
        encoded_input = {k: v.to(model.device) for k, v in encoded_input.items()}
        
        # Get model output
        with torch.no_grad():
            #output = model.encoder(**encoded_input) # For CodeT5+ last hidden layer
            output = model(**encoded_input) # Standard forwarding
        
        # Extract embedding (assuming last hidden state is used)
        
        if ckp == "Salesforce/codet5p-110m-embedding":
            #embedding = output.last_hidden_state.squeeze(0) # For CodeT5+ last hidden layer
            #mean_pooled = embedding.mean(dim=0) # For CodeT5+ last hidden layer
            mean_pooled = output.squeeze() # Standard forwarding
        else:
            embedding = output.last_hidden_state.squeeze(0)  # Remove batch dimension
        
            # Average pooling to get a single vector
            mean_pooled = embedding.mean(dim=0)
        
        # Add to representations list
        representations.append(mean_pooled.cpu().numpy())

    return np.array(representations)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", default='codebert', type=str, required=False, help="Type of model: codebert, codet5p, codellama")
    parser.add_argument("--lang", default='ruby', type=str, required=False, help="Language of the code")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--is_finetuned", action="store_true", help="use finetuned codebert")
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    
    # Set seed
    set_seed(args.seed)
    
    # Set model ckp    
    if args.model == "codebert":
        model_ckp = "microsoft/codebert-base"
    elif args.model == "codet5p":
        model_ckp = "Salesforce/codet5p-110m-embedding"
    elif args.model == "codellama":
        model_ckp = "codellama/CodeLlama-7b-hf"
    else:
        print("Invalid model name")
        exit(1)
    
    if args.lang == 'r':
        dataset = load_statcodesearch()
    else:
        dataset = load_dataset('code_search_net', args.lang)["test"]
    
    code_embs = create_embs(dataset, "code", model_ckp, args)
    if args.is_finetuned:
        np.save(f'./embeddings/code_embs_{args.model}_{args.lang}_finetuned',code_embs)
    else:
        np.save(f'./embeddings/code_embs_{args.model}_{args.lang}',code_embs)
        
    doc_embs = create_embs(dataset, "doc", model_ckp, args)
    if args.is_finetuned:
        np.save(f'./embeddings/doc_embs_{args.model}_{args.lang}_finetuned',doc_embs)
    else:
        np.save(f'./embeddings/doc_embs_{args.model}_{args.lang}',doc_embs)
        
if __name__ == '__main__':
    main()