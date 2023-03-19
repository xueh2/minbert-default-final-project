import numpy as np
from multitask_classifier_base_training import * 

# -------------------------------------------------------

def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x
            
# -------------------------------------------------------

def do_learning_para(model, optimizer, iter_para, para_train_dataloader, device, args):

    bce_logit_loss = nn.BCEWithLogitsLoss(reduction='sum')
    
    loss_all = []

    model.train()
    for iteration in range(args.para_iter):
       
        try:
            batch_para = next(iter_para)
        except StopIteration:
            iter_para = iter(para_train_dataloader)
            batch_para = next(iter_para)
            
        para_token_ids_1 = batch_para['token_ids_1'].to(device, non_blocking=True)
        para_attention_mask_1 = batch_para['attention_mask_1'].to(device, non_blocking=True)
        para_token_ids_2 = batch_para['token_ids_2'].to(device, non_blocking=True)
        para_attention_mask_2 = batch_para['attention_mask_2'].to(device, non_blocking=True)
        para_labels = batch_para['labels'].float().to(device, non_blocking=True)
                
        if(args.use_amp):
            with torch.cuda.amp.autocast():
                para_logits = model([para_token_ids_1, para_token_ids_2], [para_attention_mask_1, para_attention_mask_2], 'para')
                loss = bce_logit_loss(para_logits, para_labels[:, None]) / args.para_batch_size
        else:
            para_logits = model([para_token_ids_1, para_token_ids_2], [para_attention_mask_1, para_attention_mask_2], 'para')                
            loss = bce_logit_loss(para_logits, para_labels[:, None]) / args.para_batch_size
              
        # Backward pass - Update fast model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_all.append(loss.item())

    return loss_all, iter_para

# -------------------------------------------------------

def do_learning_sst(model, optimizer, iter_sst, sst_train_dataloader, device, args):

    loss_all = []

    model.train()
    for iteration in range(args.sst_iter):
        
        try:
            batch_sst = next(iter_sst)
        except StopIteration:
            iter_sst = iter(sst_train_dataloader)
            batch_sst = next(iter_sst)
                                        
        b_ids, b_mask, b_labels = (batch_sst['token_ids'], batch_sst['attention_mask'], batch_sst['labels'])

        b_ids = b_ids.to(device, non_blocking=True)
        b_mask = b_mask.to(device, non_blocking=True)
        b_labels = b_labels.to(device, non_blocking=True)

        if(args.use_amp):
            with torch.cuda.amp.autocast():
                sst_logits = model(b_ids, b_mask, 'sst')
                loss = F.cross_entropy(sst_logits, b_labels.view(-1), reduction='sum') / args.sst_batch_size
        else:
            sst_logits = model(b_ids, b_mask, 'sst')                    
            loss = F.cross_entropy(sst_logits, b_labels.view(-1), reduction='sum') / args.sst_batch_size
                
        # Backward pass - Update fast model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_all.append(loss.item())

    return loss_all, iter_sst
    
# -------------------------------------------------------

def do_learning_sts(model, optimizer, iter_sts, sts_train_dataloader, device, args):

    loss_all = []

    mse_loss = nn.MSELoss(reduction='mean')
    l1_loss = nn.L1Loss(reduction='sum')
    kl_loss = nn.KLDivLoss(reduction="sum")
    
    model.train()
    for iteration in range(args.sts_iter):
        
        try:
            batch_sts = next(iter_sts)
        except StopIteration:
            iter_sts = iter(sts_train_dataloader)
            batch_sts = next(iter_sts)
                    
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
                    #loss = l1_loss(sts_logits, sts_labels[:, None]) / args.sts_batch_size + (1.0 - corr_coef(sts_logits, sts_labels[:, None]))
                    loss = mse_loss(sts_logits, sts_labels)
            else:
                sts_logits = model([sts_token_ids_1, sts_token_ids_2], [sts_attention_mask_1, sts_attention_mask_2], 'sts')                    
                #loss = l1_loss(sts_logits, sts_labels[:, None]) / args.sts_batch_size + (1.0 - corr_coef(sts_logits, sts_labels[:, None]))
                loss = mse_loss(sts_logits, sts_labels)
        else:
            if(args.use_amp):
                with torch.cuda.amp.autocast():
                    sts_logits = model([sts_token_ids_1, sts_token_ids_2], [sts_attention_mask_1, sts_attention_mask_2], 'sts')
                    sts_y_hat_prob = F.log_softmax(sts_logits, dim=1)
                    loss = kl_loss(sts_y_hat_prob, sts_probs) / args.sts_batch_size
            else:
                sts_logits = model([sts_token_ids_1, sts_token_ids_2], [sts_attention_mask_1, sts_attention_mask_2], 'sts')                    
                sts_y_hat_prob = F.log_softmax(sts_logits, dim=1)
                
                sts_labels_y_hat = convert_logits_to_label_STS(sts_logits)
                
                loss = l1_loss(sts_labels_y_hat, sts_labels) / args.sts_batch_size + (1.0 - corr_coef(sts_labels_y_hat, sts_labels)) + kl_loss(sts_y_hat_prob, sts_probs) / args.sts_batch_size
                                   
        # Backward pass - Update fast model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_all.append(loss.item())

    return loss_all, iter_sts

# -------------------------------------------------------

def get_inner_optimizer(model, args, task_str, state=None):
    
    optimizer = None
    Adam_amsgrad = False
    AdamW_amsgrad = False
    SGD_nesterov = False
    
    if (args.optimizer == "Adam"):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=args.weight_decay, amsgrad=Adam_amsgrad)

    if (args.optimizer  == "AdamW"):
        #optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=AdamW_amsgrad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=AdamW_amsgrad)

    if (args.optimizer  == "SGD"):
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
                              nesterov=SGD_nesterov)
    
    if (args.optimizer  == "NAdam"):
        optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                weight_decay=args.weight_decay, momentum_decay=0.004)
        
    if state is not None:
        optimizer.load_state_dict(state)

    scheduler = None

    if (args.scheduler == "ReduceLROnPlateau"):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                               patience=1,
                                                               min_lr=1e-7,
                                                               cooldown=1,
                                                               factor=0.5,
                                                               verbose=True)
        scheduler_on_batch = False
        
    if (args.scheduler == "StepLR"):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.StepLR_step_size, 
                                                    gamma=args.StepLR_gamma, last_epoch=-1, verbose=False)
        scheduler_on_batch = False
        
    if (args.scheduler == "CosineAnnealingLR"):
        if args.task_str == "para":
            T_max = args.para_iter
        if args.task_str == "sst":
            T_max = args.sst_iter
        if args.task_str == "sts":
            T_max = args.sts_iter
            
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max,
                                                               eta_min=1e-6, last_epoch=-1)

        scheduler_on_batch = False
        
    return optimizer, scheduler

# -------------------------------------------------------

def get_meta_optimizer(model, args, state=None):
    
    optimizer = None
    Adam_amsgrad = False
    AdamW_amsgrad = False
    SGD_nesterov = False
    
    if (args.meta_optimizer == "Adam"):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr, betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=args.meta_weight_decay, amsgrad=Adam_amsgrad)

    if (args.meta_optimizer  == "AdamW"):
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.meta_lr, betas=(0.9, 0.999), eps=1e-08,
                                weight_decay=args.meta_weight_decay, amsgrad=AdamW_amsgrad)

    if (args.meta_optimizer  == "SGD"):
        optimizer = torch.optim.SGD(model.parameters(), lr=args.meta_lr, momentum=0.9, weight_decay=args.meta_weight_decay,
                              nesterov=SGD_nesterov)
    
    if (args.meta_optimizer  == "NAdam"):
        optimizer = torch.optim.NAdam(model.parameters(), lr=args.meta_lr, betas=(0.9, 0.999), eps=1e-08,
                                weight_decay=args.meta_weight_decay, momentum_decay=0.004)
        
    if state is not None:
        optimizer.load_state_dict(state)

    scheduler = None

    if (args.meta_scheduler == "ReduceLROnPlateau"):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                               patience=1,
                                                               min_lr=1e-7,
                                                               cooldown=1,
                                                               factor=0.5,
                                                               verbose=True)
        scheduler_on_batch = False
        
    if (args.meta_scheduler == "StepLR"):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.meta_StepLR_step_size, 
                                                    gamma=0.8, last_epoch=-1, verbose=True)
        scheduler_on_batch = False

    if (args.meta_scheduler == "OneCycleLR"):
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                        total_steps=None, epochs=args.meta_iter,
                                                        steps_per_epoch=1, pct_start=0.3,
                                                        anneal_strategy='cos', cycle_momentum=True,
                                                        base_momentum=0.85, max_momentum=0.95,
                                                        div_factor=25,
                                                        final_div_factor=10000,
                                                        three_phase=False,
                                                        last_epoch=-1)

        scheduler_on_batch = True
        
    if (args.meta_scheduler == "CosineAnnealingLR"):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.meta_iter,
                                                               eta_min=1e-6, last_epoch=-1, verbose=False)

        scheduler_on_batch = False
        
    return optimizer, scheduler, scheduler_on_batch

# -------------------------------------------------------

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# -------------------------------------------------------
 
def train_multitask_reptile(args):

    device, sst_train_dataset, num_labels, para_train_dataset, sts_train_dataset, \
    sst_dev_dataset, num_labels, para_dev_dataset, sts_dev_dataset, \
    sst_train_data, sst_dev_data, sst_train_dataloader, sst_dev_dataloader, \
    para_train_data, para_dev_data, para_train_dataloader, para_dev_dataloader, \
    sts_train_data, sts_dev_data, sts_train_dataloader, sts_dev_dataloader = train_multitask_base(args)    
    
    para_val_train_dataloader, para_dev_dataloader = create_para_data_loader(para_train_data, para_dev_data, args)
    sst_val_train_dataloader, sst_dev_dataloader = create_sst_data_loader(sst_train_data, sst_dev_data, args)
    sts_val_train_dataloader, sts_dev_dataloader = create_sts_data_loader(sts_train_data, sts_dev_data, args)

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
    # may not work for data parallel
    # if args.dp and torch.cuda.device_count()>1:
    #     model = torch.nn.DataParallel(model)
    #     with_data_parallel = True
    #     print(f"{Fore.RED}--> Model on data parallel.{Style.RESET_ALL}")
        
    model = model.to(device)   
    meta_optimizer, meta_scheduler, meta_scheduler_on_batch = get_meta_optimizer(model, args, state=None)
            
    # --------------------------------------------------------
    best_dev_acc = 0
  
    para_print_start = Fore.GREEN
    sts_print_start = Fore.GREEN
    sst_print_start = Fore.GREEN
    
    active_color = Fore.RED
    
    para_print_start = active_color
    sts_print_start = active_color
    sst_print_start = active_color    
   
    # -------------------------------------------------------------
    
    print(f"{Fore.YELLOW}--{Style.RESET_ALL}" * 32)
    
    # --------------------------------------------------------    
    
    if(args.use_amp):
        scaler = torch.cuda.amp.GradScaler()
    
    sst_train_loss = AverageMeter()
    para_train_loss = AverageMeter()
    sts_train_loss = AverageMeter()
            
    iter_para = iter(para_train_dataloader)
    iter_sst = iter(sst_train_dataloader)
    iter_sts = iter(sts_train_dataloader)
                
    state_para = None
    state_sst = None
    state_sts = None
    
    # --------------------------------------------------------    
    # Run for the specified number of meta iterations
    loop = tqdm(range(args.meta_iter), desc=f'training loop', bar_format='{percentage:3.0f}%|{bar:40}{r_bar}')
    
    step_para = 0
    step_sst = 0
    step_sts = 0

    probs = np.array([args.task_sample_prob_para, args.task_sample_prob_sst, args.task_sample_prob_sts])
    probs /= np.sum(probs)

    iter_inds = np.random.permutation(np.arange(args.meta_iter))

    #for meta_iteration, task_ind in enumerate(iter_inds):
    for meta_iteration in range(args.meta_iter):
               
        model.train()

        meta_lr = args.meta_lr * (1. - meta_iteration/float(args.meta_iter))
        #set_learning_rate(meta_optimizer, meta_lr)
         
        # create inner model
        inner_model = model.clone()
        
        task_ind = np.random.choice(3, 1, p=probs)

        if task_ind%3 == 0:
            task_str = "para"
        elif task_ind%3 == 1:
            task_str = "sst"
        elif task_ind%3 == 2:
            task_str = "sts"
            
        if task_str == "para":
            optimizer, scheduler = get_inner_optimizer(inner_model, args, task_str, state=state_para)
        if task_str == "sst":
            optimizer, scheduler = get_inner_optimizer(inner_model, args, task_str, state=state_sst)
        if task_str == "sts":
            optimizer, scheduler = get_inner_optimizer(inner_model, args, task_str, state=state_sts)
                   
        if task_str == "para": 
            loss, iter_para = do_learning_para(inner_model, optimizer, iter_para, para_train_dataloader, device, args)
            para_train_loss.update_list(loss, 1)
            step_para += 1
        elif task_str == "sst":
            loss, iter_sst = do_learning_sst(inner_model, optimizer, iter_sst, sst_train_dataloader, device, args)
            sst_train_loss.update_list(loss, 1)
            step_sst += 1
        elif task_str == "sts":
            loss, iter_sts = do_learning_sts(inner_model, optimizer, iter_sts, sts_train_dataloader, device, args)
            sts_train_loss.update_list(loss, 1)
            step_sts += 1
        
        inner_lr = optimizer.param_groups[0]['lr']

        if task_str == "para":
            state_para = optimizer.state_dict()
        if task_str == "sst":
            state_sst = optimizer.state_dict()
        if task_str == "sts":
            state_sts = optimizer.state_dict()
        
        model.point_grad_to(inner_model)
        meta_optimizer.step()
    
        meta_scheduler.step()
           
        curr_lr = meta_scheduler.optimizer.param_groups[0]['lr']
           
        if args.lr>1e-7 and meta_iteration>0 and meta_iteration % args.StepLR_step_size == 0:
            args.lr *= args.StepLR_gamma
            if task_str == "para":
                state_para['param_groups'][0]['lr'] = args.lr
            if task_str == "sst":
                state_sst['param_groups'][0]['lr'] = args.lr
            if task_str == "sts":
                state_sts['param_groups'][0]['lr'] = args.lr

        # ---------------------------------------------------------------------
        # set the loop
        loop.update(1)
        
        loop.set_postfix_str(f"{Fore.GREEN} meta_lr {curr_lr:g}, inner_lr {inner_lr:g}, {Fore.YELLOW} meta_iter {meta_iteration}, {task_str}, {para_print_start} para {step_para}: {para_train_loss.avg:.4f}, {sst_print_start} sst {step_sst}: {sst_train_loss.avg:.4f}, {sts_print_start} sts {step_sts}: {sts_train_loss.avg:.4f}")
            
        # ---------------------------------------------------------------------
        # add summary
        if args.wandb:
            if task_str == "para": 
                wandb.define_metric("para/step")
                wandb.log({"para/step":step_para})
                wandb.define_metric("para/loss", step_metric='para/step')
                wandb.define_metric("para/train_loss", step_metric='para/step')
                wandb.log({"para/loss": loss, "para/train_loss":para_train_loss.avg})
            elif task_str == "sst":
                wandb.define_metric("sst/step")
                wandb.log({"sst/step":step_sst})
                wandb.define_metric("sst/loss", step_metric='sst/step')
                wandb.define_metric("sst/train_loss", step_metric='sst/step')
                wandb.log({"sst/loss": loss, "sst/train_loss":sst_train_loss.avg})
            elif task_str == "sts":
                wandb.define_metric("sts/step")
                wandb.log({"sts/step":step_sts})
                wandb.define_metric("sts/loss", step_metric='sts/step')
                wandb.define_metric("sts/train_loss", step_metric='sts/step')
                wandb.log({"sts/loss": loss, "sts/train_loss":sts_train_loss.avg})                           
                
            wandb.define_metric("step")
            wandb.log({"step":meta_iteration})
            wandb.define_metric("meta_lr", step_metric='step')
            wandb.log({"meta_lr": curr_lr})
            
        # ------------------------------------------------------------------------------------------------------------
        # validation
        if (meta_iteration % args.meta_validate_every == 0) or (meta_iteration == args.meta_iter-1):
            
            para_train_accuracy = 0
            sst_train_accuracy = 0
            sts_train_corr = 0

            if args.without_train_for_evaluation is False:
                para_train_accuracy, para_y_pred, para_sent_ids, \
                    sst_train_accuracy,sst_y_pred, sst_sent_ids, \
                    sts_train_corr, sts_y_pred, sts_sent_ids = model_eval_multitask(sst_val_train_dataloader,
                                                                                para_val_train_dataloader,
                                                                                sts_val_train_dataloader,
                                                                                model, device, args)
                
            para_dev_accuracy, para_y_pred, para_sent_ids, \
                sst_dev_accuracy,sst_y_pred, sst_sent_ids, \
                sts_dev_corr, sts_y_pred, sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                            para_dev_dataloader,
                                                                            sts_dev_dataloader,
                                                                            model, device, args)
            
            # -------------------------------------------
            dev_acc = 0
            if args.task_sample_prob_para>0:
                dev_acc += para_dev_accuracy
            if args.task_sample_prob_sst>0:
                dev_acc += sst_dev_accuracy
            if args.task_sample_prob_sts>0:
                dev_acc += sts_dev_corr

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
                    
            print(f"{Fore.YELLOW}--> dev acc is {dev_acc:.4f} for step {meta_iteration}.{Style.RESET_ALL}")
            
            # -------------------------------------------
            
            print(f"{Fore.YELLOW}Step {meta_iteration}: {sst_print_start} sentimental analysis, train loss :: {sst_train_loss.avg :.3f}, train acc :: {sst_train_accuracy :.3f}, dev acc :: {sst_dev_accuracy :.3f}{Style.RESET_ALL}")
            if args.wandb:
                if args.without_train_for_evaluation is False:
                    wandb.define_metric("sst/train_accuracy", step_metric='sst/step')                    
                    wandb.log({"sst/train_accuracy": sst_train_accuracy, "sst/dev_accuracy":sst_dev_accuracy})                    
                    wandb.define_metric("sst_train_accuracy", step_metric='step')
                    wandb.log({"sst_train_accuracy": sst_train_accuracy})

                wandb.define_metric("sst/dev_accuracy", step_metric='sst/step')
                wandb.log({"sst/dev_accuracy":sst_dev_accuracy})
                wandb.define_metric("sst_dev_accuracy", step_metric='step')
                wandb.log({"sst_dev_accuracy": sst_dev_accuracy})
        
            print(f"{Fore.YELLOW}Step {meta_iteration}: {para_print_start} paraphrase analysis, train loss :: {para_train_loss.avg :.3f}, train acc :: {para_train_accuracy :.3f}, dev acc :: {para_dev_accuracy :.3f}{Style.RESET_ALL}")
            if args.wandb:
                if args.without_train_for_evaluation is False:
                    wandb.define_metric("para/train_accuracy", step_metric='para/step')                    
                    wandb.log({"para/train_accuracy": para_train_accuracy, "para/dev_accuracy":para_dev_accuracy})                    
                    wandb.define_metric("para_train_accuracy", step_metric='step')
                    wandb.log({"para_train_accuracy": para_train_accuracy})

                wandb.define_metric("para/dev_accuracy", step_metric='para/step')
                wandb.log({"para/dev_accuracy":para_dev_accuracy})
                wandb.define_metric("para_dev_accuracy", step_metric='step')
                wandb.log({"para_dev_accuracy": para_dev_accuracy})

            print(f"{Fore.YELLOW}Step {meta_iteration}: {sts_print_start} sentence similarity analysis, train loss :: {sts_train_loss.avg :.3f}, train corr :: {sts_train_corr :.3f}, dev corr :: {sts_dev_corr :.3f}{Style.RESET_ALL}")
            if args.wandb:
                if args.without_train_for_evaluation is False:
                    wandb.define_metric("sts/train_accuracy", step_metric='sts/step')                    
                    wandb.log({"sts/train_accuracy": sts_train_corr, "sts/dev_accuracy":sts_dev_corr})                    
                    wandb.define_metric("sts_train_accuracy", step_metric='step')
                    wandb.log({"sts_train_accuracy": sts_train_corr})

                wandb.define_metric("sts/dev_accuracy", step_metric='sts/step')
                wandb.log({"sts/dev_accuracy":sts_dev_corr})
                wandb.define_metric("sts_dev_accuracy", step_metric='step')
                wandb.log({"sts_dev_accuracy": sts_dev_corr})
                
    print(f"{Fore.YELLOW}--{Style.RESET_ALL}" * 32)    

# -----------------------------------------------------------------------------------------------------------------------------------------

def get_args_reptile(parser = argparse.ArgumentParser("multi-task-reptile")):

    parser = get_args(parser=parser)

    # outter, meta iterations
    parser.add_argument(
        "--meta_optimizer",
        type=str,
        default="SGD",
        help='Adam, Adamw, SGD, NAdam, for outter meta optimization'
    )
                
    parser.add_argument(
        "--meta_scheduler",
        type=str,
        default="StepLR",
        help='ReduceLROnPlateau, StepLR, or OneCycleLR or CosineAnnealingLR, for outter meta optimization'
    )
    
    parser.add_argument('--meta_StepLR_step_size', type=int, default=10, help='step size to reduce lr for outter SGD optimizer')
    
    parser.add_argument("--meta_iter", type=int, default=30, help="number of outter meta-iteration")
    
    parser.add_argument('--meta_lr', type=float, default=1.0, help='meta learning rate')
    parser.add_argument('--meta_weight_decay', type=float, default=0.0, help='weight decay')
    
    parser.add_argument("--meta_validate_every", type=int, default=5, help="number of outter meta-iteration to run evaluation")
    
    parser.add_argument('--task_sample_prob_para', type=float, default=0.0, help="sample probablity, task")
    parser.add_argument('--task_sample_prob_sst', type=float, default=0.0, help="sample probablity, sst")
    parser.add_argument('--task_sample_prob_sts', type=float, default=1.0, help="sample probablity, sts")

    # inner optimization
    parser.add_argument("--para_iter", type=int, default=10, help="number of inner meta iterations")   
    parser.add_argument("--sst_iter", type=int, default=10, help="number of inner meta iterations")    
    parser.add_argument("--sts_iter", type=int, default=10, help="number of inner meta iterations")

    return parser

# -----------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    print(f"{Fore.YELLOW}--{Style.RESET_ALL}" * 32)

    colorama_init()
    
    parser = get_args_reptile(parser = argparse.ArgumentParser("multi-task-reptile"))
    args = parser.parse_args()
        
    moment = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    
    os.makedirs(os.path.join("runs", args.experiment), exist_ok=True)
    args.filepath = os.path.join("runs", args.experiment, f'{args.option}-{args.epochs}-{args.lr}-{args.experiment}-reptile-{moment}.pt') # save path
    
    print(args)
    print(f"{Fore.YELLOW}--{Style.RESET_ALL}" * 16)

    if(args.wandb):
        wandb.init(project="CS224", group=args.experiment, config=args, tags=moment)
        wandb.watch_called = False

    #seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask_reptile(args)
    print(f"{Fore.GREEN}--{Style.RESET_ALL}" * 32)
    test_model(args)
    print(f"{Fore.GREEN}--{Style.RESET_ALL}" * 32)
