import torch
from torch import nn
from tqdm import tqdm

def euler_sampler(model, x0, y, step_size, num_steps, with_traj=False, **kwargs):
    """
    :param model: the diffusion model
    :param x0: input noise
    :param y: input guidance (label/embed)
    :param step_size: dt
    :param num_steps: 
    :param with_traj: return trajectory
    :returns: None or trajectory (num_steps, x0_like)
    """
    x = x0
    ts = torch.linspace(0.0, num_steps*step_size, num_steps)
    if with_traj:
        traj = [x]
    with torch.no_grad():
        for t in tqdm(ts):
            t = torch.full((x.shape[0],), t, device=x.device)
            x = x + step_size * model.forwardCFG(x,t,y, **kwargs)
            if with_traj:
                traj.append(x)
        
    
    return traj if with_traj else x
    