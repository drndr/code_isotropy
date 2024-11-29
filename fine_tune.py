import argparse
import random
import os
import pandas as pd

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

from tqdm import tqdm

from info_nce import InfoNCE

def info_nce_loss(query, positive_key, negative_keys):
    query = torch.flatten(query).view(-1).unsqueeze(0)
    positive_key = torch.flatten(positive_key).view(-1).unsqueeze(0)
    negative_keys = torch.stack([torch.flatten(code_emb).view(-1) for code_emb in negative_keys])
    return InfoNCE(negative_mode='unpaired')(query, positive_key, negative_keys)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
        
class CustomDataset(TensorDataset):

    def __init__(self, dataframe, tokenizer):
        self.tokenizer = tokenizer
        self.doc = dataframe.doc
        self.code = dataframe.code
        # self.targets = dataframe.labels
        self.max_len = 256

    def __len__(self):
        assert len(self.doc) == len(self.code)
        return len(self.doc)

    def __getitem__(self, index):
        doc = str(self.doc[index])
        doc = " ".join(doc.split())

        code = str(self.code[index])
        doc_inputs = self.tokenizer.encode_plus(
            doc,
            add_special_tokens=False,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_token_type_ids=False
        )
        doc_ids = doc_inputs['input_ids']
        doc_mask = doc_inputs['attention_mask']

        code_inputs = self.tokenizer.encode_plus(
            code,
            add_special_tokens=False,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_token_type_ids=False
        )

        code_ids = code_inputs['input_ids']
        code_mask = code_inputs['attention_mask']

        return {
            'doc_ids': torch.tensor(doc_ids, dtype=torch.long),
            'doc_mask': torch.tensor(doc_mask, dtype=torch.long),
            'code_ids': torch.tensor(code_ids, dtype=torch.long),
            'code_mask': torch.tensor(code_mask, dtype=torch.long),
        }
        
def load_data(lang):
    code_search_dataset = load_dataset('code_search_net', lang)

    # train_data
    train_data = code_search_dataset['train']
    print("train size: ",len(train_data))

    function_code = [' '.join(i).strip() for i in train_data['func_code_tokens']]
    function_documentation = [' '.join(i).strip() for i in train_data['func_documentation_tokens']]
    #function_code = train_data['func_code_string']
    #function_documentation = train_data['func_documentation_string']

    train_df = pd.DataFrame()
    train_df['doc'] = function_documentation
    train_df['code'] = function_code

    # test_data
    test_data = code_search_dataset['test']

    function_code_test = [' '.join(i).strip() for i in test_data['func_code_tokens']]
    function_documentation_test = [' '.join(i).strip() for i in test_data['func_documentation_tokens']]
    #function_code_test = test_data['func_code_string']
    #function_documentation_test = test_data['func_documentation_string']

    test_df = pd.DataFrame()
    test_df['doc'] = function_documentation_test
    test_df['code'] = function_code_test

    return train_df, test_df
        
def train(args, model, optimizer, training_set):
    print("Start training")
    model.train()
    train_dataloader = DataLoader(training_set, batch_size=args.train_batch_size, shuffle=True)
    for epoch in tqdm(range(1, args.num_train_epochs + 1)):

        all_losses = []   
        all_contr_losses = []
        all_decorr_losses = []
        all_decorr_losses2 = []
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch}')
        
        for idx, batch in enumerate(progress_bar):

            if batch['code_ids'].size(0) < args.train_batch_size:
                continue

            query_id = batch['doc_ids'].to(args.device)#.unsqueeze(0)
            query_mask = batch['doc_mask'].to(args.device)#.unsqueeze(0)
            inputs = {'input_ids': query_id[0,:].unsqueeze(0), 'attention_mask': query_mask[0,:].unsqueeze(0)}
            
            query = model(**inputs)
            #print("query shape: ", query[0].shape)
            query = query.last_hidden_state.squeeze(0).mean(dim=0)
            #print("query shape: ", query.shape)
            
            code_ids = batch['code_ids'].to(args.device)
            code_masks = batch['code_mask'].to(args.device)
            

            inputs = {'input_ids': code_ids[0,:].unsqueeze(0), 'attention_mask': code_masks[0,:].unsqueeze(0)}
            positive_code_key = model(**inputs)
            positive_code_key = positive_code_key.last_hidden_state.squeeze(0).mean(dim=0)
            #print("p code", positive_code_key.shape)
            

            inputs = {'input_ids': code_ids[1:], 'attention_mask': code_masks[1:]}
            negative_code_keys = model(**inputs)
            negative_code_keys = negative_code_keys.last_hidden_state.mean(dim=1)
            #print("n code", negative_code_keys.shape)
            
            #break
            
            loss = info_nce_loss(query, positive_code_key, negative_code_keys)
            
            loss.backward()
            
            all_losses.append(loss.to("cpu").detach().numpy())

            if (idx + 1) % args.num_of_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                      
        train_mean_loss = np.mean(all_losses)
        print(f'Epoch {epoch} - Train-Loss: {train_mean_loss}')
        
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    parser.add_argument("--lang", default='ruby', type=str, required=False, help="Language of the code")

    parser.add_argument("--train_batch_size", default=32, type=int, required=False, help="Training batch size")

    parser.add_argument("--learning_rate", default=5e-5, type=float, required=False, help="Learning rate")

    parser.add_argument("--num_train_epochs", default=5, type=int, required=False, help="Number of training epochs")
                                                                                      
    parser.add_argument("--num_of_accumulation_steps", default=4, type=int, required=False,
                        help="Number of accumulation steps")
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    args.device = device
    
    set_seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", trust_remote_code=True)
    model = AutoModel.from_pretrained("microsoft/codebert-base", trust_remote_code=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    model.to(device)
    
    train_df, test_df = load_data(args.lang)

    train_dataset = train_df.reset_index(drop=True)
    test_dataset = test_df.reset_index(drop=True)

    training_set = CustomDataset(train_dataset, tokenizer)
    test_set = CustomDataset(test_dataset, tokenizer)
    
    train(args, model, optimizer, training_set)
    
    torch.save(model.state_dict(), f'./models/codebert_{args.lang}.pth')
    
if __name__ == '__main__':
    main()