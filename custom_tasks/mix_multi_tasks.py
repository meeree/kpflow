import torch
from torch import nn
import numpy as np
from . import memory_pro

DEFAULT_CFG = {
    'T_context': 30,
}
DEFAULT_CFG.update(memory_pro.DEFAULT_CFG)

def generate(cfg=DEFAULT_CFG, noise=True, debug=False):
    # Inputs are 7 dimensional: 
    # [fixation, stim1 cos, stim1 sin, task1, task2, task3, task4]
    # where task1-4 are 1-hot encoding for:
    # task1: memory-pro, task2: memory-anti, task3: delay-pro, task4: delay-anti
    
    ctx_end = cfg["T_context"]
    
    # Order: mem-pro, mem-anti, delay-pro, delay-anti
    task_flags = {'anti': [False, True, False, True], 
                 'delay': [False, False, True, True]}
    
    inps, targets = [], []
    cfg['n_samples'] = cfg['n_samples'] // 4  # Divide up samples between four tasks
    
    for task_idx in range(4):
        cfg['anti'] = task_flags['anti'][task_idx]
        cfg['delay'] = task_flags['delay'][task_idx]
        
        inp_i, target_i = memory_pro.generate(cfg, debug=debug, noise=noise)
        
        # Create context period with fixation
        ctx_period = torch.zeros((inp_i.shape[0], ctx_end, inp_i.shape[2]))
        ctx_period[:, :, 0] = 1.  # Fixation
        
        # Combine context and task periods
        inp_i = torch.cat((ctx_period, inp_i), 1)
        target_i = torch.cat((ctx_period, target_i), 1)
        
        # Create 1-hot task encoding channels (4 channels, one per task)
        batch_size = inp_i.shape[0]
        time_steps = inp_i.shape[1]
        
        # Initialize all task channels to zero
        task_channels = torch.zeros((batch_size, time_steps, 4))
        
        # Set the appropriate task channel to 1 during context period
        task_channels[:, :ctx_end, task_idx] = 1.0
        
        # Concatenate original input with task channels
        inp_i = torch.cat((inp_i, task_channels), -1)  # [B, T, 3] -> [B, T, 7]
        
        inps.append(inp_i)
        targets.append(target_i)
    
    inp = torch.cat(inps, 0)

    targets_stacked = [torch.zeros((targ.shape[0], targ.shape[1], 12)) for targ in targets]
    for i, (targ_stack, targ) in enumerate(zip(targets_stacked, targets)):
        targ_stack[:, :, i * 3 : (i+1) * 3] = targ

    target = torch.cat(targets, 0) # [4 * B, T, 3]
    
    # Add noise to missing parts if specified
    if noise:
        with torch.no_grad():
            inp[:, :ctx_end, :3] += torch.normal(torch.zeros_like(inp[:, :ctx_end, :3]), 1.) * .1 * (2 ** .5)
            inp[:, :, 3:] += torch.normal(torch.zeros_like(inp[:, :, 3:]), 1.) * .1 * (2 ** .5)
    
    # Debug plotting if requested
    if debug:
        import matplotlib.pyplot as plt
        for b in range(4):
            plt.figure(figsize=(12, 10))
            
            # Plot stimulus channels
            for chan, name in enumerate(['Fixation', 'Stim1 Cos', 'Stim1 Sin']):
                for qidx, quant in enumerate([inp, target]):
                    plt.subplot(5, 2, 1 + 2*chan + qidx)
                    plt.plot(quant[b, :, chan], linewidth=4)
                    plt.title(name)
                    plt.ylim(-1.2, 1.2)
            
            # Plot task channels
            task_names = ['Memory-Pro', 'Memory-Anti', 'Delay-Pro', 'Delay-Anti']
            for i in range(4):
                plt.subplot(5, 2, 7 + i)
                plt.plot(inp[b, :, 3 + i], linewidth=4)
                plt.title(f'Task: {task_names[i]}')
                plt.ylim(-1.2, 1.2)
            
            plt.suptitle('Input Left, Target Right')
            plt.tight_layout()
    
    return inp, target

def accuracy(X, Y):
    return memory_pro.accuracy(X, Y)
