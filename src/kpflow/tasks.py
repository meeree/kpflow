# Wrappers for supporting a variety of tasks.
# Task consists of two functions:
# __init__ to setup the task.
# __call__ which returns a batch for training or eval. 

# Optionally, can define an accuracy function. This should return a dictionary of accuracy metrics.

def get_task_wrapper(suite):
    # Given a string, return a wrapper corresponding to that suite of tasks. 
    # E.g., 'neurogym' -> NeuroGymWrapper
    mapping = {'neurogym': NeuroGymWrapper, 'custom': CustomTaskWrapper}
    if suite not in list(mapping.keys()):
        raise Exception(f'Suite {suite} does not exist. Valid suites: {list(mapping.keys())}')
    return mapping[suite]

class NeuroGymWrapper():
    def __init__(self, task, batch_size, seq_len = 100, **kwargs):
        import neurogym as ngym
        env_kwargs = {}
        env_kwargs['dt'] = kwargs.get('dt', 100)
        self.dataset = ngym.Dataset(task, env_kwargs=env_kwargs, batch_size=batch_size, seq_len=seq_len)

    def __call__(self):
        import torch
        inputs, labels = self.dataset()
        inputs = torch.from_numpy(inputs).float().swapaxes(0,1)
        labels = torch.from_numpy(labels).swapaxes(0,1)
        # Target is one-hot encoded label. Labels are in range [1, n labels], so inputs of zero mean no response. These will become fixation channel automagically!
        target = torch.nn.functional.one_hot(labels).float()
        return inputs, target

    def accuracy(self, out, targets):
        guess, labels = out.argmax(-1), targets.argmax(-1)
        match = (guess == labels)
        acc = match.float().mean().item()
        acc_respond = match[labels != 0].float().mean().item() # Accuracy when no fixation input.
        return {'acc': acc, 'acc_respond': acc_respond}

class CustomTaskWrapper():
    def __init__(self, task, batch_size, use_noise = True, **kwargs):
        import torch
        tasks_avail = ['flip_flop', 'memory_pro', 'memory_pro_oh', 'mix_multi_tasks', 'cont_integration']
        if task == 'flip_flop':
            from .custom_tasks import flip_flop
            self.task = flip_flop
        elif task == 'memory_pro':
            from .custom_tasks import memory_pro
            self.task = memory_pro 
        elif task == 'memory_pro_oh':
            from .custom_tasks import memory_pro_oh
            self.task = memory_pro_oh 
        elif task == 'mix_multi_tasks':
            from .custom_tasks import mix_multi_tasks
            self.task = mix_multi_tasks
        elif task == 'low_rank_forecast':
            from .custom_tasks import low_rank_forecast
            self.task = low_rank_forecast
        else:
            raise Exception(f"No such task: {task}, available tasks: {tasks_avail}")

        self.cfg = self.task.DEFAULT_CFG
        self.cfg.update(kwargs)
        self.use_noise = use_noise 

        inps, targets = self.task.generate(self.cfg, noise = self.use_noise)
        self.dataset = torch.utils.data.TensorDataset(inps, targets)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=use_noise)

    def __call__(self):
        return next(iter(self.dataloader))

    def accuracy(self, out, targets):
        return self.task.accuracy(out, targets)
