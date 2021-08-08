import argparse
from pathlib import Path
import os
import datetime

import torch
import numpy as np
import torch.nn as nn

from models import GHDE
from dataset import IncongruentHeadlineNewsDataset
import utils
from train import *
import preprocessing
from preprocessing import *
import pickle
max_seq_len = preprocessing.MAX_NUM_PARA

def preprocess(file_path,voca_path):
    df = load_data_as_df(file_path)
   
    with open(voca_path, 'rb') as f:
        voca =  pickle.load(f)
    np_data = process_dataset(df, voca)
    save_path = Path("temp/test/")
    print("Saving data...")
    os.makedirs(save_path, exist_ok=True)
    for name, data in np_data.items():
        np.save(save_path / '{}.npy'.format(name), data)
    return save_path

def inference(file_path,model,args):
    print(file_path)
    processed_data_path = preprocess(file_path,args.glove_voca_path)

    test_dataset = IncongruentHeadlineNewsDataset(processed_data_path ,
                                                    seq_type='para', max_seq_len=max_seq_len, construct_graph=True)
    print(f"Test dataset size: {len(test_dataset):9,}")
    
    _, test_para_acc, test_acc, test_auc = evaluate(model, test_dataset, device)
    print(f"TEST ACC(PARA): {test_para_acc:.4f} | TEST ACC(DOC): {test_acc:.4f} | TEST AUROC(DOC): {test_auc:.4f}")
    

if __name__=="__main__":
    # Parse Args ---------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='ghde')
    parser.add_argument("--model_path", type=str,
                        help="pretrained model path",default='saved/ckpt_ghde_2021-08-05_11 09 48.476392.pt')
    parser.add_argument("--input_file_path", type=str,default='test.tsv',help="news data text file")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--glove_voca_path",default='saved/voca.pkl', action="store_true")
    parser.add_argument("--glove_path", type=Path,default='saved/voca_glove_embedding.npy',
        help="Path for GloVe embeddings to load and extract embeddings for the voca of data")
    model_parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    print(model_parser)
    print(args, unknown)

     # model-specific params
    model_parser = argparse.ArgumentParser()
    for arg_name, arg_type, arg_default in GHDE.param_args():
        if arg_type is bool:
            arg_type=lambda x: (str(x).lower() in ['true','1', 'yes'])
        model_parser.add_argument(arg_name, type=arg_type, default=arg_default)
    model_args = model_parser.parse_args(unknown)

    device = torch.device('cuda' if (not args.no_cuda) and torch.cuda.is_available() else 'cpu')

    # Initialize Model ---------------------------------------------------------
    # shared model params
    model_kwargs = {
        'params': vars(model_args)
    }
    # Load embeddings ----------------------------------------------------------
    voca_glove_embedding_path =args.glove_path 
    glove_embeds = torch.tensor(np.load(voca_glove_embedding_path))
    model_kwargs['pretrained_embeds'] = glove_embeds

    print("Initializing model...")
    model = GHDE(**model_kwargs)
    model.to(device)
    print(f"model: {args.model}")
    # Load pretrained model  ----------------------------------------------------------
    model.load_state_dict(torch.load(args.model_path,map_location=torch.device('cpu')))
    print("Loading test dataset...")
    
    inference(args.input_file_path,model,args)