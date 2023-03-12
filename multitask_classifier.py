import time, random, numpy as np, argparse, sys, re, os
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, model_eval_multitask, test_model_multitask

from utils import AverageMeter

TQDM_DISABLE=False

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO
        # sentiment
        self.sentiment_drop_out = torch.nn.Dropout(0.1)
        self.sentiment_output_proj = torch.nn.Linear(config.hidden_size, 5)

        # paraphrase
        self.paraphrase_drop_out = torch.nn.Dropout(0.1)
        self.paraphrase_output_proj1 = torch.nn.Linear(2*config.hidden_size, config.hidden_size)
        self.paraphrase_nl = F.gelu
        self.paraphrase_output_proj2 = torch.nn.Linear(config.hidden_size, 1)
                
        # similarity
        self.similarity_drop_out = torch.nn.Dropout(0.1)
        self.similarity_output_proj1 = torch.nn.Linear(2*config.hidden_size, config.hidden_size)
        self.similarity_nl = F.gelu
        self.similarity_output_proj2 = torch.nn.Linear(config.hidden_size, 1)
        
    def call_backbone(self, input_ids, attention_mask):
        res = self.bert(input_ids, attention_mask)
        return res['pooler_output'], res['last_hidden_state']
        
    def forward(self, input_ids, attention_mask, task_str):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        if task_str == 'sst':
            return self.predict_sentiment(input_ids, attention_mask)
        
        if task_str == 'sts':
            return self.predict_similarity(input_ids[0], attention_mask[0], input_ids[1], attention_mask[1])
        
        if task_str == 'para':
            return self.predict_paraphrase(input_ids[0], attention_mask[0], input_ids[1], attention_mask[1])

        raise "incorrect task str ..."

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        pooler_output, last_hidden_state = self.call_backbone(input_ids, attention_mask)
        logits = self.sentiment_output_proj(self.sentiment_drop_out(pooler_output))
        return logits

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        pooler_output_1, _ = self.call_backbone(input_ids_1, attention_mask_1)       
        pooler_output_2, _ = self.call_backbone(input_ids_2, attention_mask_2)
        
        x = torch.concat((pooler_output_1, pooler_output_2), dim=1)
        x = self.paraphrase_output_proj1(self.paraphrase_drop_out(x))
        logits = self.paraphrase_output_proj2(self.paraphrase_nl(x))
        return logits

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        pooler_output_1, _ = self.call_backbone(input_ids_1, attention_mask_1)       
        pooler_output_2, _ = self.call_backbone(input_ids_2, attention_mask_2)
        
        x = torch.concat((pooler_output_1, pooler_output_2), dim=1)
        x = self.similarity_output_proj1(self.similarity_drop_out(x))
        logits = self.similarity_output_proj2(self.similarity_nl(x))
        return logits




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
    print(f"--> Save the model to {Fore.RED}{filepath}{Style.RESET_ALL}")


## Currently only trains on sst dataset
def train_multitask(args):
    
    # check    
    assert not args.without_para or not args.without_sts or not args.without_sst
    
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
    sst_train_dataset, num_labels, para_train_dataset, sts_train_dataset = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_dataset, num_labels, para_dev_dataset, sts_dev_dataset = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    # -------------------------------------------------------
    # sst datasets
    sst_train_data = SentenceClassificationDataset(sst_train_dataset, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_dataset, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sst_train_data.collate_fn, num_workers=num_workers, prefetch_factor=8)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sst_dev_data.collate_fn, num_workers=num_workers, prefetch_factor=8)

    num_sst = len(sst_train_dataloader)
    print(f"sst train data has {num_sst} batches ...")
    
    # -------------------------------------------------------
    # para datasets
    para_train_data = SentencePairDataset(para_train_dataset, args, isRegression=False)
    para_dev_data = SentencePairDataset(para_dev_dataset, args, isRegression=False)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=para_train_data.collate_fn, num_workers=num_workers, prefetch_factor=8)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=para_dev_data.collate_fn, num_workers=num_workers, prefetch_factor=8)
        
    num_para = len(para_train_dataloader)
    print(f"para train data has {num_para} batches ...")
    
    # -------------------------------------------------------
    # sts datasets
    sts_train_data = SentencePairDataset(sts_train_dataset, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_dataset, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sts_train_data.collate_fn, num_workers=num_workers, prefetch_factor=8)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sts_dev_data.collate_fn, num_workers=num_workers, prefetch_factor=8)
    
    num_sts = len(sts_train_dataloader)
    print(f"sts train data has {num_sts} batches ...")
    
    # -------------------------------------------------------
    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    
    with_data_parallel = False
    if args.dp and torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
        with_data_parallel = True
        print(f"{Fore.RED}--> Model on data parallel.{Style.RESET_ALL}")
        
    model = model.to(device)
   
    # -------------------------------------------------------------

    optimizer = None
    Adam_amsgrad = False
    AdamW_amsgrad = False
    SGD_nesterov = False
    
    if (args.optimizer == "Adam"):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=args.weight_decay, amsgrad=Adam_amsgrad)

    if (args.optimizer  == "AdamW"):
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                weight_decay=args.weight_decay, amsgrad=AdamW_amsgrad)

    if (args.optimizer  == "SGD"):
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
                              nesterov=SGD_nesterov)
    
    if (args.optimizer  == "NAdam"):
        optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                weight_decay=args.weight_decay, momentum_decay=0.004)
        
    # -------------------------------------------------------------
        
    scheduler = None

    if (args.scheduler == "ReduceLROnPlateau"):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                               patience=2,
                                                               min_lr=1e-6,
                                                               cooldown=1,
                                                               factor=0.5,
                                                               verbose=True)
        scheduler_on_batch = False
        
    if (args.scheduler == "StepLR"):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.8, last_epoch=-1, verbose=True)
        scheduler_on_batch = False

    if (args.scheduler == "OneCycleLR"):
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr,
                                                        total_steps=None, epochs=config.num_epochs,
                                                        steps_per_epoch=len(para_train_dataloader), pct_start=0.3,
                                                        anneal_strategy='cos', cycle_momentum=True,
                                                        base_momentum=0.85, max_momentum=0.95,
                                                        div_factor=25,
                                                        final_div_factor=10000,
                                                        three_phase=False,
                                                        last_epoch=-1)

        scheduler_on_batch = True
    
    # --------------------------------------------------------
    best_dev_acc = 0

    bce_logit_loss = nn.BCEWithLogitsLoss(reduction='sum')
    mse_loss = nn.MSELoss(reduction='sum')

    epoch_para = 0
    epoch_sst = 0
    epoch_sts = 0
    
    para_print_start = Fore.GREEN
    sts_print_start = Fore.GREEN
    sst_print_start = Fore.GREEN
    
    active_color = Fore.RED
    
    num_step = 0                       
    if args.without_para is False:
        if len(para_train_dataloader)>num_step:
            num_step = len(para_train_dataloader)
        
        para_print_start = active_color
        
    if args.without_sts is False:
        if len(sts_train_dataloader)>num_step:
            num_step = len(sts_train_dataloader)
        
        sts_print_start = active_color
            
    if args.without_sst is False:
        if len(sst_train_dataloader)>num_step:
            num_step = len(sst_train_dataloader)
        
        sst_print_start = active_color    

    print(f"number of steps per epoch is {num_step}")
    
    # --------------------------------------------------------    
    
    print(f"{Fore.YELLOW}--{Style.RESET_ALL}" * 32)
    
    # --------------------------------------------------------    
    
    if(args.use_amp):
        scaler = torch.cuda.amp.GradScaler()
    
    # --------------------------------------------------------    
    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        
        model.train()
        
        sst_train_loss = AverageMeter()
        para_train_loss = AverageMeter()
        sts_train_loss = AverageMeter()
              
        if args.without_para is False:
            iter_para = iter(para_train_dataloader)
        
        if args.without_sst is False:
            iter_sst = iter(sst_train_dataloader)
            
        if args.without_sts is False:
            iter_sts = iter(sts_train_dataloader)

        loop = tqdm(range(num_step), desc=f'training loop', bar_format='{percentage:3.0f}%|{bar:40}{r_bar}')
                    
        # loop over the largest batches
        for ind_step in loop:
            
            # para
            if args.without_para is False:
                try:
                    batch_para = next(iter_para)
                except StopIteration:
                    iter_para = iter(para_train_dataloader)
                    batch_para = next(iter_para)
                    epoch_para += 1
                    
                para_token_ids_1 = batch_para['token_ids_1'].to(device, non_blocking=True)
                para_attention_mask_1 = batch_para['attention_mask_1'].to(device, non_blocking=True)
                para_token_ids_2 = batch_para['token_ids_2'].to(device, non_blocking=True)
                para_attention_mask_2 = batch_para['attention_mask_2'].to(device, non_blocking=True)
                para_labels = batch_para['labels'].float().to(device, non_blocking=True)
                
                if(args.use_amp):
                    with torch.cuda.amp.autocast():
                        para_logits = model([para_token_ids_1, para_token_ids_2], [para_attention_mask_1, para_attention_mask_2], 'para')
                        para_loss = bce_logit_loss(para_logits, para_labels[:, None]) / args.batch_size
                else:
                    para_logits = model([para_token_ids_1, para_token_ids_2], [para_attention_mask_1, para_attention_mask_2], 'para')                
                    para_loss = bce_logit_loss(para_logits, para_labels[:, None]) / args.batch_size
                
                para_train_loss.update(para_loss.item(), args.batch_size)        
            else:
                para_loss = 0
                
            # sst      
            if args.without_sst is False:
                try:
                    batch_sst = next(iter_sst)
                except StopIteration:
                    iter_sst = iter(sst_train_dataloader)
                    batch_sst = next(iter_sst)
                    epoch_sst += 1
                    
                b_ids, b_mask, b_labels = (batch_sst['token_ids'],
                                        batch_sst['attention_mask'], 
                                        batch_sst['labels'])

                b_ids = b_ids.to(device, non_blocking=True)
                b_mask = b_mask.to(device, non_blocking=True)
                b_labels = b_labels.to(device, non_blocking=True)

                if(args.use_amp):
                    with torch.cuda.amp.autocast():
                        sst_logits = model(b_ids, b_mask, 'sst')
                        sst_loss = F.cross_entropy(sst_logits, b_labels.view(-1), reduction='sum') / args.batch_size
                else:
                    sst_logits = model(b_ids, b_mask, 'sst')                    
                    sst_loss = F.cross_entropy(sst_logits, b_labels.view(-1), reduction='sum') / args.batch_size
                
                sst_train_loss.update(sst_loss.item(), args.batch_size)
            else:
                sst_loss = 0
    
            # sts
            if args.without_sts is False:
                try:
                    batch_sts = next(iter_sts)
                except StopIteration:
                    iter_sts = iter(sts_train_dataloader)
                    batch_sts = next(iter_sts)
                    epoch_sts += 1
                    
                sts_token_ids_1 = batch_sts['token_ids_1'].to(device, non_blocking=True)
                sts_attention_mask_1 = batch_sts['attention_mask_1'].to(device, non_blocking=True)
                sts_token_ids_2 = batch_sts['token_ids_2'].to(device, non_blocking=True)
                sts_attention_mask_2 = batch_sts['attention_mask_2'].to(device, non_blocking=True)
                sts_labels = batch_sts['labels'].float().to(device, non_blocking=True)
                
                if(args.use_amp):
                    with torch.cuda.amp.autocast():
                        sts_logits = model([sts_token_ids_1, sts_token_ids_2], [sts_attention_mask_1, sts_attention_mask_2], 'sts')
                        sts_loss = mse_loss(sts_logits, sts_labels[:, None]) / args.batch_size
                else:
                    sts_logits = model([sts_token_ids_1, sts_token_ids_2], [sts_attention_mask_1, sts_attention_mask_2], 'sts')                    
                    sts_loss = mse_loss(sts_logits, sts_labels[:, None]) / args.batch_size
                    
                sts_train_loss.update(sts_loss.item(), args.batch_size)        
            else:
                sts_loss = 0
            
            # combined loss
            loss = para_loss + sts_loss + sst_loss
            
            # backprop
            optimizer.zero_grad(set_to_none=True)
            
            if(args.use_amp):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                                     
            # scheduler              
            if (scheduler is not None) and scheduler_on_batch:
                scheduler.step()           
                curr_lr = scheduler.get_last_lr()[0]
            else:
                curr_lr = scheduler.optimizer.param_groups[0]['lr']
                if ind_step == 0:
                    curr_lr = args.lr
                
            # set the loop
            loop.set_postfix_str(f"{Fore.GREEN} lr {curr_lr:g}, {Fore.YELLOW} epoch {epoch} - step {ind_step}, {para_print_start} para {epoch_para} : {para_train_loss.avg:.4f}, {sst_print_start} sst {epoch_sst} : {sst_train_loss.avg:.4f}, {sts_print_start} sts {epoch_sts} : {sts_train_loss.avg:.4f}")
        # --------------------------------------------------------------------
        
        epoch_loss = para_train_loss.avg + sst_train_loss.avg + sts_train_loss.avg
        
        para_train_loss.reset()
        sst_train_loss.reset()
        sts_train_loss.reset()
        
        # --------------------------------------------------------------------
        if (scheduler is not None) and (scheduler_on_batch == False):
            if(isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                scheduler.step(epoch_loss)
            else:
                scheduler.step()
            print(f"{Fore.YELLOW}for epoch {epoch}, loss is {epoch_loss:.4f}, current learning rate is {scheduler.optimizer.param_groups[0]['lr']}{Style.RESET_ALL}")
            
        # --------------------------------------------------------------------
        # validation
        # para_train_accuracy, para_y_pred, para_sent_ids, \
        #     sst_train_accuracy,sst_y_pred, sst_sent_ids, \
        #     sts_train_corr, sts_y_pred, sts_sent_ids = model_eval_multitask(sst_train_dataloader,
        #                                                                 para_train_dataloader,
        #                                                                 sts_train_dataloader,
        #                                                                 model, device)
            
        para_dev_accuracy, para_y_pred, para_sent_ids, \
            sst_dev_accuracy,sst_y_pred, sst_sent_ids, \
            sts_dev_corr, sts_y_pred, sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                        para_dev_dataloader,
                                                                        sts_dev_dataloader,
                                                                        model, device, args)
        dev_acc = para_dev_accuracy + sst_dev_accuracy + sts_dev_corr
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            if with_data_parallel:
                model_saved = model.module
            else:
                model_saved = model
                
            save_model(model_saved, optimizer, args, config, args.filepath)

        print(f"{Fore.YELLOW}Epoch {epoch}: {sst_print_start} sentimental analysis, train loss :: {sst_train_loss.avg :.3f}, dev acc :: {sst_dev_accuracy :.3f}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Epoch {epoch}: {para_print_start} paraphrase analysis, train loss :: {para_train_loss.avg :.3f}, dev acc :: {para_dev_accuracy :.3f}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Epoch {epoch}: {sts_print_start} sentence similarity analysis, train loss :: {sts_train_loss.avg :.3f}, dev corr :: {sts_dev_corr :.3f}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}--{Style.RESET_ALL}" * 32)

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


def get_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--use_amp", action='store_true')
    parser.add_argument("--dp", help='if set, perform data parallel training', action='store_true')

    parser.add_argument("--without_para", action='store_true')
    parser.add_argument("--without_sst", action='store_true')
    parser.add_argument("--without_sts", action='store_true')

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
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
    
    parser.add_argument(
        "--activation",
        type=str,
        default="LeakyReLU",
        help='ReLU, LeakyReLU, or ELU'
    )
    
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    
    # -------------------------------
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    colorama_init()
    
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
