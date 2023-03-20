import time, random, numpy as np, argparse, sys, re, os
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm
import wandb

from datasets import SentenceClassificationDataset, SentencePairDataset, SentencePairSTSDataset, convert_logits_to_label_STS, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, model_eval_multitask, test_model_multitask

from utils import AverageMeter, count_parameters
from multitask_classifier_model import MultitaskBERT

TQDM_DISABLE=False
BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

# -------------------------------------------------------
# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# -------------------------------------------------------
def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"{Fore.GREEN}--> Save the model to {filepath}{Style.RESET_ALL}")

# -------------------------------------------------------
def corr_coef(y, y_hat):
    vy = y - torch.mean(y)
    vy_hat = y_hat - torch.mean(y_hat)

    cost = torch.sum(vy * vy_hat) / (torch.sqrt(torch.sum(vy ** 2)) * torch.sqrt(torch.sum(vy_hat ** 2)) + 1e-4)
    
    return cost

# -------------------------------------------------------
def create_para_data_loader(para_train_data, para_dev_data, args):
    
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.para_batch_size, collate_fn=para_train_data.collate_fn, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.para_batch_size, collate_fn=para_dev_data.collate_fn, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
        
    return para_train_dataloader, para_dev_dataloader
            
            
def create_sst_data_loader(sst_train_data, sst_dev_data, args):
    
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.sst_batch_size, collate_fn=sst_train_data.collate_fn, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.sst_batch_size, collate_fn=sst_dev_data.collate_fn, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
        
    return sst_train_dataloader, sst_dev_dataloader

def create_sts_data_loader(sts_train_data, sts_dev_data, args):
    
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.sts_batch_size, collate_fn=sts_train_data.collate_fn, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.sts_batch_size, collate_fn=sts_dev_data.collate_fn, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
        
    return sts_train_dataloader, sts_dev_dataloader

# -------------------------------------------------------
def train_multitask_base(args):
    
    # check    
    assert not args.without_para or not args.without_sts or not args.without_sst
    
    # -------------------------------------------------------
    
    # writer = SummaryWriter(log_dir=f"multi-task/{args.experiment}")
    
    # -------------------------------------------------------
    print(f"{Fore.YELLOW}--{Style.RESET_ALL}" * 32)
    print(f"Start training ... ")
    
    # device
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    # -------------------------------------------------------
    print(f"{Fore.YELLOW}--{Style.RESET_ALL}" * 32)
    
    # -------------------------------------------------------
    # Load data    
    num_workers = 2
    
    # Create the data and its corresponding datasets and dataloader
    sst_train_dataset, num_labels, para_train_dataset, sts_train_dataset = load_multitask_data(args.sst_train, args.para_train, args.sts_train, percentage_to_use=args.percentage_data_for_train, split ='train')
    sst_dev_dataset, num_labels, para_dev_dataset, sts_dev_dataset = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, percentage_to_use=100, split ='train')

    # -------------------------------------------------------
    # sst datasets
    sst_train_data = SentenceClassificationDataset(sst_train_dataset, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_dataset, args)

    sst_train_dataloader, sst_dev_dataloader = create_sst_data_loader(sst_train_data, sst_dev_data, args)

    num_sst = len(sst_train_dataloader)
    print(f"sst train data has {num_sst} batches ...")
    
    # -------------------------------------------------------
    # para datasets
    para_train_data = SentencePairDataset(para_train_dataset, args, isRegression=False)
    para_dev_data = SentencePairDataset(para_dev_dataset, args, isRegression=False)

    para_train_dataloader, para_dev_dataloader = create_sst_data_loader(para_train_data, para_dev_data, args)
        
    num_para = len(para_train_dataloader)
    print(f"para train data has {num_para} batches ...")
    
    # -------------------------------------------------------
    # sts datasets
    sts_train_data = SentencePairSTSDataset(sts_train_dataset, args, isRegression=True)
    sts_dev_data = SentencePairSTSDataset(sts_dev_dataset, args, isRegression=True)

    sts_train_dataloader, sts_dev_dataloader = create_sst_data_loader(sts_train_data, sts_dev_data, args)
    
    num_sts = len(sts_train_dataloader)
    print(f"sts train data has {num_sts} batches ...")

    return device, sst_train_dataset, num_labels, para_train_dataset, sts_train_dataset, \
            sst_dev_dataset, num_labels, para_dev_dataset, sts_dev_dataset, \
            sst_train_data, sst_dev_data, sst_train_dataloader, sst_dev_dataloader, \
            para_train_data, para_dev_data, para_train_dataloader, para_dev_dataloader, \
            sts_train_data, sts_dev_data, sts_train_dataloader, sts_dev_dataloader
        
    
# -----------------------------------------------------------------------------------------------------------------------------------------

def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)

# ------------------------------------------------------------------------

def get_args(parser = argparse.ArgumentParser("multi-task")):

    # input data
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    # output results
    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # traning parameter
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--use_amp", action='store_true')
    parser.add_argument("--dp", help='if set, perform data parallel training', action='store_true')

    parser.add_argument("--without_para", action='store_true')
    parser.add_argument("--without_sst", action='store_true')
    parser.add_argument("--without_sts", action='store_true')

    parser.add_argument("--without_train_for_evaluation", action='store_true')

    parser.add_argument("--percentage_data_for_train", type=float, default=100.0, help='percentage of all training data used')

    parser.add_argument("--wandb", action='store_true')
    
    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=64)
    
    parser.add_argument("--para_batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=64)
    parser.add_argument("--sst_batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=64)
    parser.add_argument("--sts_batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=64)
    
    parser.add_argument("--num_workers", help='number of workers for loader', type=int, default=2)
    parser.add_argument("--prefetch_factor", help='number of prefetched batches', type=int, default=4)
    
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help='Adam, Adamw, SGD, NAdam'
    )
                
    parser.add_argument(
        "--scheduler",
        type=str,
        default="OneCycleLR",
        help='ReduceLROnPlateau, StepLR, or OneCycleLR'
    )
    
    parser.add_argument('--StepLR_step_size', type=int, default=4, help='step size to reduce lr for StepLR scheduler')
    parser.add_argument('--StepLR_gamma', type=float, default=0.8, help='gamma to reduce lr for StepLR scheduler')
    
    parser.add_argument(
        "--activation",
        type=str,
        default="LeakyReLU",
        help='ReLU, LeakyReLU, or ELU'
    )
    
    parser.add_argument(
        "--sts_train_method",
        type=str,
        default="classification",
        help='classification or regression; whether to train sts as a classification or regression problem'
    )
    
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    
    parser.add_argument('--num_steps', type=str, default="mean", help='max or min or mean; use the max of all tasks as num_step or minimal value or mean')
    
    parser.add_argument('--experiment', type=str, default="multi-task", help='experiment string')

    return parser

# ------------------------------------------------------------------------

if __name__ == "__main__":
   pass
