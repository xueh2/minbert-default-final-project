
from multitask_classifier_base_training import * 

# -------------------------------------------------------
## Currently only trains on sst dataset
def train_multitask(args):

    device, sst_train_dataset, num_labels, para_train_dataset, sts_train_dataset, \
    sst_dev_dataset, num_labels, para_dev_dataset, sts_dev_dataset, \
    sst_train_data, sst_dev_data, sst_train_dataloader, sst_dev_dataloader, \
    para_train_data, para_dev_data, para_train_dataloader, para_dev_dataloader, \
    sts_train_data, sts_dev_data, sts_train_dataloader, sts_dev_dataloader = train_multitask_base(args)    
    
    # -------------------------------------------------------
    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option,
              'sts_train_method': args.sts_train_method}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    num_paras = count_parameters(model)
    print(f"{Fore.RED}--> Number of model parameters {num_paras} - {num_paras/1024/1024:.2f} MB.{Style.RESET_ALL}")
    
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
        
    # --------------------------------------------------------
    best_dev_acc = 0

    bce_logit_loss = nn.BCEWithLogitsLoss(reduction='sum')
    mse_loss = nn.MSELoss(reduction='mean')
    l1_loss = nn.L1Loss(reduction='sum')
    kl_loss = nn.KLDivLoss(reduction="sum")

    epoch_para = 0
    epoch_sst = 0
    epoch_sts = 0
    
    para_print_start = Fore.GREEN
    sts_print_start = Fore.GREEN
    sst_print_start = Fore.GREEN
    
    active_color = Fore.RED
    
    num_steps = []
    if args.without_para is False:
        num_steps.append(len(para_train_dataloader))
        para_print_start = active_color
        
    if args.without_sts is False:
        num_steps.append(len(sts_train_dataloader))
        sts_print_start = active_color
            
    if args.without_sst is False:        
        num_steps.append(len(sst_train_dataloader))
        sst_print_start = active_color    

    if args.num_steps == "min":
        num_step = min(num_steps)
    elif args.num_steps == "max":
        num_step = max(num_steps)
    else:
        num_step = int(np.mean(np.array(num_steps)))

    print(f"number of steps per epoch is {num_step}")
    
    # -------------------------------------------------------------
        
    scheduler = None

    if (args.scheduler == "ReduceLROnPlateau"):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                               patience=0,
                                                               min_lr=1e-7,
                                                               cooldown=1,
                                                               factor=0.5,
                                                               verbose=True)
        scheduler_on_batch = False
        
    if (args.scheduler == "StepLR"):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.StepLR_step_size, gamma=0.8, last_epoch=-1, verbose=True)
        scheduler_on_batch = False

    if (args.scheduler == "OneCycleLR"):
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                        total_steps=None, epochs=args.epochs,
                                                        steps_per_epoch=num_step, pct_start=0.3,
                                                        anneal_strategy='cos', cycle_momentum=True,
                                                        base_momentum=0.85, max_momentum=0.95,
                                                        div_factor=25,
                                                        final_div_factor=10000,
                                                        three_phase=False,
                                                        last_epoch=-1)

        scheduler_on_batch = True
            
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

        print(f"{Fore.GREEN}Epoch {epoch} starts ... {Style.RESET_ALL}")
        
        loop = tqdm(range(num_step), desc=f'training loop', bar_format='{percentage:3.0f}%|{bar:40}{r_bar}')
                    
        # -----------------------------------------------------------------------------------------------------------------------------------------
        # loop over the steps
        for ind_step in loop:
            
            # ---------------------------------------------------------------------
            # para
            # ---------------------------------------------------------------------
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
                        para_loss = bce_logit_loss(para_logits, para_labels[:, None]) / para_train_dataloader.batch_size
                else:
                    para_logits = model([para_token_ids_1, para_token_ids_2], [para_attention_mask_1, para_attention_mask_2], 'para')                
                    para_loss = bce_logit_loss(para_logits, para_labels[:, None]) / para_train_dataloader.batch_size
                
                para_train_loss.update(para_loss.item(), para_train_dataloader.batch_size)        
            else:
                para_loss = 0
                
            # ---------------------------------------------------------------------
            # sst      
            # ---------------------------------------------------------------------
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
                        sst_loss = F.cross_entropy(sst_logits, b_labels.view(-1), reduction='sum') / sst_train_dataloader.batch_size
                else:
                    sst_logits = model(b_ids, b_mask, 'sst')                    
                    sst_loss = F.cross_entropy(sst_logits, b_labels.view(-1), reduction='sum') / sst_train_dataloader.batch_size
                
                sst_train_loss.update(sst_loss.item(), sst_train_dataloader.batch_size)
            else:
                sst_loss = 0
    
            # ---------------------------------------------------------------------
            # sts
            # ---------------------------------------------------------------------
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
                sts_probs = batch_sts['probs'].float().to(device, non_blocking=True)
                
                if args.sts_train_method == 'regression':
                    if(args.use_amp):
                        with torch.cuda.amp.autocast():
                            sts_logits = model([sts_token_ids_1, sts_token_ids_2], [sts_attention_mask_1, sts_attention_mask_2], 'sts')
                            #sts_loss = l1_loss(sts_logits, sts_labels[:, None]) / sts_train_dataloader.batch_size + (1.0 - corr_coef(sts_logits, sts_labels[:, None]))
                            sts_loss = mse_loss(sts_logits, sts_labels[:, None])
                    else:
                        sts_logits = model([sts_token_ids_1, sts_token_ids_2], [sts_attention_mask_1, sts_attention_mask_2], 'sts')                    
                        #sts_loss = l1_loss(sts_logits, sts_labels[:, None]) / sts_train_dataloader.batch_size + (1.0 - corr_coef(sts_logits, sts_labels[:, None]))
                        # only mse give best results
                        sts_loss = mse_loss(sts_logits, sts_labels[:, None])
                else:
                    if(args.use_amp):
                        with torch.cuda.amp.autocast():
                            sts_logits = model([sts_token_ids_1, sts_token_ids_2], [sts_attention_mask_1, sts_attention_mask_2], 'sts')
                            sts_y_hat_prob = F.log_softmax(sts_logits, dim=1)
                            sts_loss = kl_loss(sts_y_hat_prob, sts_probs) / sts_train_dataloader.batch_size
                    else:
                        sts_logits = model([sts_token_ids_1, sts_token_ids_2], [sts_attention_mask_1, sts_attention_mask_2], 'sts')                    
                        sts_y_hat_prob = F.log_softmax(sts_logits, dim=1)
                        
                        sts_labels_y_hat = convert_logits_to_label_STS(sts_logits)
                        
                        sts_loss = l1_loss(sts_labels_y_hat, sts_labels) / sts_train_dataloader.batch_size + (1.0 - corr_coef(sts_labels_y_hat, sts_labels)) + kl_loss(sts_y_hat_prob, sts_probs) / sts_train_dataloader.batch_size
                    
                sts_train_loss.update(sts_loss.item(), sts_train_dataloader.batch_size)        
            else:
                sts_loss = 0
            
            # ---------------------------------------------------------------------
            # combined loss
            loss = para_loss + sts_loss + sst_loss
                        
            # ---------------------------------------------------------------------
            # backprop
            optimizer.zero_grad(set_to_none=True)
            
            if(args.use_amp):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            # ---------------------------------------------------------------------
            # scheduler              
            if (scheduler is not None) and scheduler_on_batch:
                scheduler.step()           
                curr_lr = scheduler.get_last_lr()[0]
            else:
                curr_lr = scheduler.optimizer.param_groups[0]['lr']
                if ind_step == 0:
                    curr_lr = args.lr
            # ---------------------------------------------------------------------
            # set the loop
            loop.set_postfix_str(f"{Fore.GREEN} lr {curr_lr:g}, {Fore.YELLOW} epoch {epoch} - step {ind_step}, {para_print_start} para {epoch_para} : {para_train_loss.avg:.4f}, {sst_print_start} sst {epoch_sst} : {sst_train_loss.avg:.4f}, {sts_print_start} sts {epoch_sts} : {sts_train_loss.avg:.4f}")
            
            # ---------------------------------------------------------------------
            # add summary
            if args.wandb:
                if args.without_para is False:
                    wandb.log({"para/loss": para_loss.item(), "para/train_loss":para_train_loss.avg})
                        
                if args.without_sts is False:
                    wandb.log({"sts/loss": sts_loss.item(), "sts/train_loss":sts_train_loss.avg})
                        
                if args.without_sst is False:
                    wandb.log({"sst/loss": sst_loss.item(), "sst/train_loss":sst_train_loss.avg})
                
                
        # ------------------------------------------------------------------------------------------------------------

        if args.wandb:
            wandb.define_metric("epoch")
            wandb.log({"epoch":epoch})
            
        epoch_loss = para_train_loss.avg + sst_train_loss.avg + sts_train_loss.avg
        if args.wandb:
            wandb.define_metric("epoch_loss", step_metric='epoch')
            wandb.log({"epoch_loss": epoch_loss})
                          
        # --------------------------------------------------------------------
        if (scheduler is not None) and (scheduler_on_batch == False):
            if(isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                scheduler.step(epoch_loss)
            else:
                scheduler.step()
                
            epoch_lr = scheduler.optimizer.param_groups[0]['lr']
            print(f"{Fore.YELLOW}for epoch {epoch}, loss is {epoch_loss:.4f}, current learning rate is {epoch_lr}{Style.RESET_ALL}")
                        
            if args.wandb:
                wandb.define_metric("epoch_lr", step_metric='epoch')
                wandb.log({"epoch_lr": epoch_lr})
        # --------------------------------------------------------------------
        # validation
        para_train_accuracy = 0
        sst_train_accuracy = 0
        sts_train_corr = 0

        if args.without_train_for_evaluation is False:
            para_train_accuracy, para_y_pred, para_sent_ids, \
                sst_train_accuracy,sst_y_pred, sst_sent_ids, \
                sts_train_corr, sts_y_pred, sts_sent_ids = model_eval_multitask(sst_train_dataloader,
                                                                            para_train_dataloader,
                                                                            sts_train_dataloader,
                                                                            model, device, args)
            
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
            print(f"{Fore.GREEN}--> for saved model, para_dev_accuracy is {para_dev_accuracy:.4f}, sst_dev_accuracy is {sst_dev_accuracy:.4f}, sts_dev_corr is {sts_dev_corr:.4f}{Style.RESET_ALL}")
            if args.wandb:
                wandb.run.summary["best_model_para_dev_accuracy"] = para_dev_accuracy
                wandb.run.summary["best_model_sst_dev_accuracy"] = sst_dev_accuracy
                wandb.run.summary["best_model_sts_dev_corr"] = sts_dev_corr
                
        print(f"{Fore.YELLOW}--> dev acc is {dev_acc:.4f} for epoch {epoch}.{Style.RESET_ALL}")
        
        if args.without_sst is False:
            print(f"{Fore.YELLOW}Epoch {epoch}: {sst_print_start} sentimental analysis, train loss :: {sst_train_loss.avg :.3f}, train acc :: {sst_train_accuracy :.3f}, dev acc :: {sst_dev_accuracy :.3f}{Style.RESET_ALL}")
            if args.wandb:
                wandb.define_metric("sst/epoch")
                wandb.define_metric("sst/train_accuracy", step_metric='sst/epoch')
                wandb.define_metric("sst/dev_accuracy", step_metric='sst/epoch')
                wandb.log({"sst/epoch":epoch, "sst/train_accuracy": sst_train_accuracy, "sst/dev_accuracy":sst_dev_accuracy})
                
                wandb.define_metric("sst_train_accuracy", step_metric='epoch')
                wandb.log({"sst_train_accuracy": sst_train_accuracy})
                wandb.define_metric("sst_dev_accuracy", step_metric='epoch')
                wandb.log({"sst_dev_accuracy": sst_dev_accuracy})
            
        if args.without_para is False:
            print(f"{Fore.YELLOW}Epoch {epoch}: {para_print_start} paraphrase analysis, train loss :: {para_train_loss.avg :.3f}, train acc :: {para_train_accuracy :.3f}, dev acc :: {para_dev_accuracy :.3f}{Style.RESET_ALL}")
            if args.wandb:
                wandb.define_metric("para/epoch")
                wandb.define_metric("para/train_accuracy", step_metric='para/epoch')
                wandb.define_metric("para/dev_accuracy", step_metric='para/epoch')
                wandb.log({"para/epoch":epoch, "para/train_accuracy": para_train_accuracy, "para/dev_accuracy":para_dev_accuracy})
                
                wandb.define_metric("para_train_accuracy", step_metric='epoch')
                wandb.log({"para_train_accuracy": para_train_accuracy})
                wandb.define_metric("para_dev_accuracy", step_metric='epoch')
                wandb.log({"para_dev_accuracy": para_dev_accuracy})
                
        if args.without_sts is False:
            print(f"{Fore.YELLOW}Epoch {epoch}: {sts_print_start} sentence similarity analysis, train loss :: {sts_train_loss.avg :.3f}, train corr :: {sts_train_corr :.3f}, dev corr :: {sts_dev_corr :.3f}{Style.RESET_ALL}")
            if args.wandb:
                wandb.define_metric("sts/epoch")
                wandb.define_metric("sts/*", step_metric='sts/epoch')
                wandb.log({"sts/epoch":epoch, "sts/train_corr": sts_train_corr, "sts/dev_corr":sts_dev_corr})
                
                wandb.define_metric("sts_train_corr", step_metric='epoch')
                wandb.log({"sts_train_corr": sts_train_corr})
                wandb.define_metric("sts_dev_corr", step_metric='epoch')
                wandb.log({"sts_dev_corr": sts_dev_corr})
                
        print(f"{Fore.YELLOW}--{Style.RESET_ALL}" * 32)
        
        # clean up
        para_train_loss.reset()
        sst_train_loss.reset()
        sts_train_loss.reset()

# -----------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    print(f"{Fore.YELLOW}--{Style.RESET_ALL}" * 32)

    colorama_init()
    
    parser = get_args()
    args = parser.parse_args()
    
    moment = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    
    os.makedirs(os.path.join("runs", args.experiment), exist_ok=True)
    args.filepath = os.path.join("runs", args.experiment, f'{args.option}-{args.epochs}-{args.lr}-{args.experiment}-{moment}.pt') # save path
    
    print(args)
    print(f"{Fore.YELLOW}--{Style.RESET_ALL}" * 16)

    if(args.wandb):
        wandb.init(project="CS224", group=args.experiment, config=args, tags=moment)
        wandb.watch_called = False
            
    #seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
