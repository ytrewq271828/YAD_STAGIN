import os
import utils
import random
import torch
import numpy as np
from model import *
from dataset import *
from tqdm import tqdm
from einops import repeat
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import inspect

def step(model, criterion, dyn_v, dyn_a, sampling_endpoints, t, label, reg_lambda, clip_grad=0.0, device='cpu', optimizer=None, scheduler=None):
    if optimizer is None: model.eval()
    else: model.train()

    # run model
    logit, attention, latent, reg_ortho = model(dyn_v.to(device), dyn_a.to(device), t.to(device), sampling_endpoints)
    loss = criterion(logit, label.to(device))
    reg_ortho *= reg_lambda
    loss += reg_ortho

    # optimize model
    if optimizer is not None:
       optimizer.zero_grad()
       loss.backward()
       if clip_grad > 0.0: torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad)
       optimizer.step()
       if scheduler is not None:
           scheduler.step()

    return logit, loss, attention, latent, reg_ortho


def train(argv):
    prefix = f'[{inspect.getframeinfo(inspect.currentframe()).function}]'
        
    # make directories
    os.makedirs(os.path.join(argv.targetdir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(argv.targetdir, 'summary'), exist_ok=True)

    # set seed and device
    torch.manual_seed(argv.seed)
    np.random.seed(argv.seed)
    random.seed(argv.seed)

    # device setting
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(argv.seed)
    else:
        device = torch.device("cpu")

    # define dataset
    if argv.dataset=='hcp_rest': 
        dataset = DatasetHCPRest(atlas=argv.atlas, target_feature=argv.target, k_fold=argv.k_fold, session='REST1', phase_encoding='RL')
    elif argv.dataset=='yad_rest': 
        dataset = DatasetYADRest(atlas=argv.atlas, target_feature=argv.target, k_fold=argv.k_fold, except_sites=argv.except_sites, except_rois=argv.except_rois)
        
    #elif argv.dataset=='task': dataset = DatasetHCPTask(argv.sourcedir, roi=argv.roi, dynamic_length=argv.dynamic_length, k_fold=argv.k_fold)
    #else: raise
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=argv.minibatch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # resume checkpoint if file exists
    if os.path.isfile(os.path.join(argv.targetdir, 'checkpoint.pth')):
        print(f'{prefix} resuming checkpoint experiment')
        checkpoint = torch.load(os.path.join(argv.targetdir, 'checkpoint.pth'), map_location=device)
    else:
        checkpoint = {
            'fold': 0,
            'epoch': 0,
            'model': None,
            'optimizer': None,
            'scheduler': None}

    # start experiment
    for k in range(checkpoint['fold'], argv.k_fold):
        # make directories per fold
        os.makedirs(os.path.join(argv.targetdir, 'model', str(k)), exist_ok=True)

        # set dataloader
        dataset.set_fold(k, train=True)

        # define model
        model = ModelSTAGIN(
            input_dim=dataset.num_nodes,
            hidden_dim=argv.hidden_dim,
            num_classes=dataset.num_classes,
            num_heads=argv.num_heads,
            num_layers=argv.num_layers,
            sparsity=argv.sparsity,
            dropout=argv.dropout,
            cls_token=argv.cls_token,
            readout=argv.readout)
        model.to(device)
        if checkpoint['model'] is not None: model.load_state_dict(checkpoint['model'])
        criterion = torch.nn.CrossEntropyLoss()

        # define optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=argv.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=argv.max_lr, epochs=argv.num_epochs, steps_per_epoch=len(dataloader), pct_start=0.2, div_factor=argv.max_lr/argv.lr, final_div_factor=1000)
        if checkpoint['optimizer'] is not None: optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler'] is not None: scheduler.load_state_dict(checkpoint['scheduler'])

        # define logging objects
        summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'train'), )
        summary_writer_val = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'val'), )
        logger = utils.logger.LoggerSTAGIN(argv.k_fold, dataset.num_classes)

        # start training
        for epoch in range(checkpoint['epoch'], argv.num_epochs):
            logger.initialize(k)
            dataset.set_fold(k, train=True)
            loss_accumulate = 0.0
            reg_ortho_accumulate = 0.0
            for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k+1} e:{epoch+1}')):
                # process input data -- assumes input shape [minibatch x node x time]
                print(f"{prefix} timeseries shape: {x['timeseries'].permute(0,2,1).shape}")
                dyn_a, sampling_points = utils.bold.process_dynamic_fc(x['timeseries'].permute(0,2,1), argv.window_size, argv.window_stride, argv.dynamic_length)
                sampling_endpoints = [p+argv.window_size for p in sampling_points]
                if i==0: dyn_v = repeat(torch.eye(dataset.num_nodes), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=argv.minibatch_size)
                if len(dyn_a) < argv.minibatch_size: dyn_v = dyn_v[:len(dyn_a)]
                t = x['timeseries'].permute(2,0,1) # [node x minibatch x time]
                label = x['label']

                logit, loss, attention, latent, reg_ortho = step(
                    model=model,
                    criterion=criterion,
                    dyn_v=dyn_v,
                    dyn_a=dyn_a,
                    sampling_endpoints=sampling_endpoints,
                    t=t,
                    label=label,
                    reg_lambda=argv.reg_lambda,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler)
                pred = logit.argmax(1)
                prob = logit.softmax(1)
                loss_accumulate += loss.detach().cpu().numpy()
                reg_ortho_accumulate += reg_ortho.detach().cpu().numpy()
                logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())
                summary_writer.add_scalar('lr', scheduler.get_last_lr()[0], i+epoch*len(dataloader))

            # summarize results
            samples = logger.get(k)
            metrics = logger.evaluate(k)
            summary_writer.add_scalar('loss', loss_accumulate/len(dataloader), epoch)
            summary_writer.add_scalar('reg_ortho', reg_ortho_accumulate/len(dataloader), epoch)
            summary_writer.add_pr_curve('precision-recall', samples['true'], samples['prob'][:,1], epoch)
            [summary_writer.add_scalar(key, value, epoch) for key, value in metrics.items() if not key=='fold']
            [summary_writer.add_image(key, make_grid(value[-1].unsqueeze(1), normalize=True, scale_each=True), epoch) for key, value in attention.items()]
            summary_writer.flush()
            print(metrics)

            # save checkpoint
            torch.save({
                'fold': k,
                'epoch': epoch+1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()},
                os.path.join(argv.targetdir, 'checkpoint.pth'))

        # finalize fold
        torch.save(model.state_dict(), os.path.join(argv.targetdir, 'model', str(k), 'model.pth'))
        checkpoint.update({'epoch': 0, 'model': None, 'optimizer': None, 'scheduler': None})

    summary_writer.close()
    summary_writer_val.close()
    os.remove(os.path.join(argv.targetdir, 'checkpoint.pth'))


def test(argv):
    os.makedirs(os.path.join(argv.targetdir, 'attention'), exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # define dataset
    if argv.dataset=='hcp_rest': 
        dataset = DatasetHCPRest(atlas=argv.atlas, target_feature=argv.target, k_fold=argv.k_fold, session='REST1', phase_encoding='RL')
    elif argv.dataset=='yad_rest': 
        dataset = DatasetYADRest(atlas=argv.atlas, target_feature=argv.target, k_fold=argv.k_fold, except_sites=argv.except_sites, except_rois=argv.except_rois)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    logger = utils.logger.LoggerSTAGIN(argv.k_fold, dataset.num_classes)

    for k in range(argv.k_fold):
        os.makedirs(os.path.join(argv.targetdir, 'attention', str(k)), exist_ok=True)

        model = ModelSTAGIN(
            input_dim=dataset.num_nodes,
            hidden_dim=argv.hidden_dim,
            num_classes=dataset.num_classes,
            num_heads=argv.num_heads,
            num_layers=argv.num_layers,
            sparsity=argv.sparsity,
            dropout=argv.dropout,
            cls_token=argv.cls_token,
            readout=argv.readout)
        model.to(device)
        model.load_state_dict(torch.load(os.path.join(argv.targetdir, 'model', str(k), 'model.pth')))
        criterion = torch.nn.CrossEntropyLoss()

        # define logging objects
        fold_attention = {'node_attention': [], 'time_attention': []}
        summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'test'))

        logger.initialize(k)
        dataset.set_fold(k, train=False)
        loss_accumulate = 0.0
        reg_ortho_accumulate = 0.0
        latent_accumulate = []
        for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k+1}')):
            with torch.no_grad():
                # process input data
                dyn_a, sampling_points = utils.bold.process_dynamic_fc(x['timeseries'].permute(0,2,1), argv.window_size, argv.window_stride)
                sampling_endpoints = [p+argv.window_size for p in sampling_points]
                if i==0: dyn_v = repeat(torch.eye(dataset.num_nodes), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=argv.minibatch_size)
                if not dyn_v.shape[1]==dyn_a.shape[1]: dyn_v = repeat(torch.eye(dataset.num_nodes), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=argv.minibatch_size)
                if len(dyn_a) < argv.minibatch_size: dyn_v = dyn_v[:len(dyn_a)]
                t = x['timeseries'].permute(2,0,1)
                label = x['label']

                logit, loss, attention, latent, reg_ortho = step(
                    model=model,
                    criterion=criterion,
                    dyn_v=dyn_v,
                    dyn_a=dyn_a,
                    sampling_endpoints=sampling_endpoints,
                    t=t,
                    label=label,
                    reg_lambda=argv.reg_lambda,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=None,
                    scheduler=None)
                pred = logit.argmax(1)
                prob = logit.softmax(1)
                logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())
                loss_accumulate += loss.detach().cpu().numpy()
                reg_ortho_accumulate += reg_ortho.detach().cpu().numpy()

                fold_attention['node_attention'].append(attention['node-attention'].detach().cpu().numpy())
                fold_attention['time_attention'].append(attention['time-attention'].detach().cpu().numpy())
                latent_accumulate.append(latent.detach().cpu().numpy())

        # summarize results
        samples = logger.get(k)
        metrics = logger.evaluate(k)
        summary_writer.add_scalar('loss', loss_accumulate/len(dataloader))
        summary_writer.add_scalar('reg_ortho', reg_ortho_accumulate/len(dataloader))
        summary_writer.add_pr_curve('precision-recall', samples['true'], samples['prob'][:,1])
        [summary_writer.add_scalar(key, value) for key, value in metrics.items() if not key=='fold']
        [summary_writer.add_image(key, make_grid(value[-1].unsqueeze(1), normalize=True, scale_each=True)) for key, value in attention.items()]
        summary_writer.flush()
        print(metrics)

        # finalize fold
        logger.to_csv(argv.targetdir, k)
        if argv.dataset=='hcp_task':
            for key, value in fold_attention.items():
                os.makedirs(os.path.join(argv.targetdir, 'attention', str(k), key), exist_ok=True)
                for idx, task in enumerate(dataset.task_list):
                    np.save(os.path.join(argv.targetdir, 'attention', str(k), key, f'{task}.npy'), np.concatenate([v for (v, l) in zip(value, samples['true']) if l==idx]))
        else:
            [np.save(os.path.join(argv.targetdir, 'attention', str(k), f'{key}.npy'), np.concatenate(value)) for key, value in fold_attention.items()]
        
        np.save(os.path.join(argv.targetdir, 'attention', str(k), 'latent.npy'), np.concatenate(latent_accumulate))
        del fold_attention

    # finalize experiment
    logger.to_csv(argv.targetdir)
    final_metrics = logger.evaluate()
    print(final_metrics)
    summary_writer.close()
    torch.save(logger.get(), os.path.join(argv.targetdir, 'samples.pkl'))


