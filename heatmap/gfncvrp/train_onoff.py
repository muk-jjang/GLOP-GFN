import sys
sys.path.insert(0, './')
import torch
import time
import argparse
import numpy as np
import math
import scipy.special
import wandb
from tqdm import tqdm
from heatmap.gfncvrp.inst import gen_inst, gen_pyg_data, trans_tsp
from heatmap.gfncvrp.eval import eval
from heatmap.gfncvrp.sampler import Sampler
from nets.partition_net import Net
from utils import load_model

EPS = 1e-10
LR = 3e-4
K_SPARSE = {
    200: 40,
    500: 100,
    1000: 200,
    2000: 400,
}

##TODO: sanghyeok backward policy
def calculate_log_pb_uniform(paths: torch.Tensor):
    # paths.shape: (batch, max_tour_length)
    # paths are start with 0 and end with 0

    _pi1 = paths.detach().cpu().numpy()
    # shape: (batch, max_tour_length)

    n_nodes = np.count_nonzero(_pi1, axis=1)
    _pi2 = _pi1[:, 1:] - _pi1[:, :-1]
    n_routes = np.count_nonzero(_pi2, axis=1) - n_nodes # count the number of routes in each instance
    _pi3 = _pi1[:, 2:] - _pi1[:, :-2]
    n_multinode_routes = np.count_nonzero(_pi3, axis=1) - n_nodes
    log_b_p = - scipy.special.gammaln(n_routes + 1) - n_multinode_routes * math.log(2)

    return torch.from_numpy(log_b_p).to(paths.device)

def infer_heatmap(model, pyg_data, gfn_loss):
    if gfn_loss == 'tb':
        heatmap, logZ = model(pyg_data, return_logZ=True)
        heatmap = heatmap / (heatmap.min()+1e-5)
        heatmap = model.reshape(pyg_data, heatmap) + 1e-5
        return heatmap, logZ
    elif gfn_loss == 'vargrad':
        heatmap = model(pyg_data)
        heatmap = heatmap / (heatmap.min()+1e-5)
        heatmap = model.reshape(pyg_data, heatmap) + 1e-5
        return heatmap
    
def train_batch(model, optimizer, n, bs, opts, beta, it):
    model.train()
    loss_lst = []
    ##################################################
    # wandb
    _train_mean_cost = 0.0
    _train_min_cost = 0.0
    _train_max_cost = 0.0
    _logZ_mean = torch.tensor(0.0, device=DEVICE)
    # sum_loss = torch.tensor(0.0, device=DEVICE)
    count = 0
    ##################################################
    for _ in range(opts.batch_size):
        coors, demand, capacity = gen_inst(n, DEVICE)# coors: (n+1, 2), tensor  demand: (n+1, ), tensor  capacity: float
        pyg_data = gen_pyg_data(coors, demand, capacity, K_SPARSE[n])# pyg_data: PyGData
        if opts.gfn_loss == 'tb':
            heatmap, logZ = infer_heatmap(model, pyg_data, opts.gfn_loss)# heatmap: (n+1, n+1)
        elif opts.gfn_loss == 'vargrad':
            heatmap = infer_heatmap(model, pyg_data, opts.gfn_loss)
        sampler = Sampler(demand, heatmap, capacity, bs, DEVICE)
        routes, log_probs = sampler.gen_subsets(require_prob=True) # routes: (bs, trajectories), log_probs: (bs, trajectories)
        tsp_insts, n_tsps_per_route = trans_tsp(coors, routes)
        objs = eval(tsp_insts, n_tsps_per_route, opts) # (bs, )
        
        ##TODO: off-policy 업데이트 구성, train 함수를 따로 뺄 것인가...?
        
        log_pf = log_probs.to(DEVICE).sum(dim=1)
        log_pb = calculate_log_pb_uniform(routes)
        costs = objs - objs.mean(0)
        if opts.gfn_loss == 'vargrad':
            log_Z_est = (-beta*costs + log_pb - log_pf)
            gfn_loss = torch.pow(log_Z_est - log_Z_est.mean(0), 2).mean(0) 
        elif opts.gfn_loss == 'tb':
            forward_flow = log_pf + logZ.expand(log_probs.size(0))
            backward_flow = log_pb - beta*costs
            gfn_loss = torch.pow(forward_flow - backward_flow, 2).mean(0)
        
        loss_lst.append(gfn_loss)
        count += 1

        # max_obj_index = objs.argmax()
        # min_obj_index = objs.argmin()
        
        ##################################################
        # wandb
        if USE_WANDB:
            _train_mean_cost += objs.mean().item()
            _train_min_cost += objs.min().item()
            _train_max_cost += objs.max().item()
            if opts.gfn_loss == 'tb':
                _logZ_mean += logZ.item()
            
            # instance_step = it * opts.batch_size + count - 1
            # wandb.log({'instance':{
            #     'max_obj_sample_instance': objs[max_obj_index].item(),
            #     'max_obj_sample_logpf': log_pf[max_obj_index].item(),
            #     'max_obj_sample_logpb': log_pb[max_obj_index].item(),
            #     'min_obj_sample_instance': objs[min_obj_index].item(),
            #     'min_obj_sample_logpf': log_pf[min_obj_index].item(),
            #     'min_obj_sample_logpb': log_pb[min_obj_index].item(),
            # }}, step=instance_step)
        ##################################################

    loss = sum(loss_lst) / opts.batch_size
    optimizer.zero_grad()
    loss.backward()
    if not opts.no_clip:
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=opts.max_norm, norm_type=2)
    optimizer.step()

    #################### 
    ## wandb logging ##
    if USE_WANDB:
        wandb.log(
            {"train":{
                "train_mean_cost": _train_mean_cost / count,
                "train_min_cost": _train_min_cost / count,
                "train_max_cost": _train_max_cost / count,
                "train_loss": loss.item(),
                "logZ": _logZ_mean.item() / count,
                "beta": beta,
            }},
            step=it,
        )
    
def infer_instance(model, inst, opts):
    model.eval()
    coors, demand, capacity = inst
    n = demand.size(0)-1
    pyg_data = gen_pyg_data(coors, demand, capacity, K_SPARSE[n])
    if opts.gfn_loss == 'tb':
        heatmap, logZ = infer_heatmap(model, pyg_data, opts.gfn_loss)
    elif opts.gfn_loss == 'vargrad':
        heatmap = infer_heatmap(model, pyg_data, opts.gfn_loss)
    sampler = Sampler(demand, heatmap, capacity, 1, DEVICE)
    routes = sampler.gen_subsets(require_prob=False, greedy_mode=True)
    tsp_insts, n_tsps_per_route = trans_tsp(coors, routes)
    obj = eval(tsp_insts, n_tsps_per_route, opts).min()
    return obj

def train_epoch(n, bs, steps_per_epoch, net, optimizer, scheduler, opts, beta_schedule_params, epoch, n_epochs):
    #Beta Schedule
    # beta_min, beta_max, beta_flat_epochs = beta_schedule_params
    # beta = beta_min + (beta_max - beta_min) * min(math.log(epoch) / math.log(n_epochs - beta_flat_epochs), 1.0)
    beta = beta_schedule_params
    for i in tqdm(range(steps_per_epoch)):
        it = (epoch - 1) * steps_per_epoch + i
        train_batch(net, optimizer, n, bs, opts, beta, it)
    scheduler.step()
    
    

@torch.no_grad()
def validation(n, net, opts):
    sum_obj = []
    for _ in range(opts.val_size):
        inst = gen_inst(n, DEVICE)
        obj = infer_instance(net, inst, opts)
        sum_obj.append(obj)
    avg_obj = torch.tensor(sum_obj).mean()
    best_obj = torch.tensor(sum_obj).min()
    return avg_obj, best_obj

def train(n, bs, steps_per_epoch, n_epochs, opts, beta_schedule_params=(50, 500, 5)):
    revisers = []
    for reviser_size in opts.revision_lens:
        reviser_path = f'pretrained/Reviser-stage2/reviser_{reviser_size}/epoch-299.pt'
        reviser, _ = load_model(reviser_path, is_local=True)
        revisers.append(reviser)
    for reviser in revisers:
        reviser.to(DEVICE)
        reviser.eval()
        reviser.set_decode_type(opts.decode_strategy)    
    opts.revisers = revisers
    
    net = Net(opts.units, 3, K_SPARSE[n], 2, depth=opts.depth, gfn_loss=opts.gfn_loss).to(DEVICE)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_epochs)
    
    if opts.checkpoint_path == '':
        starting_epoch = 1
    else:
        checkpoint = torch.load(opts.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        
    sum_time = 0
    best_avg_obj, _ = validation(n, net, opts)
    print('epoch 0', best_avg_obj.item())
    for epoch in range(starting_epoch, n_epochs + 1):
        start = time.time()
        train_epoch(n, bs, steps_per_epoch, net, optimizer, scheduler, opts, beta_schedule_params, epoch, n_epochs)
        sum_time += time.time() - start
        avg_obj, best_obj = validation(n, net, opts)
        print(f'epoch {epoch}: ', avg_obj.item())
        if best_avg_obj > avg_obj:
            best_avg_obj = avg_obj
            print(f'Save checkpoint-{epoch}.')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, f'./pretrained/Partitioner/cvrp/cvrp-{n}-{epoch}-cos.pt')
        
        # wandb
        wandb.log({"val":{
            'val_avg_obj': avg_obj.item(),
            'val_best_obj': best_obj.item(),
            'epoch': epoch,
        }}, step=epoch*steps_per_epoch)
    print('total training duration:', sum_time)
    
if __name__ == '__main__':
    import pprint as pp
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem_size', type=int, default=400)
    parser.add_argument('--revision_lens', nargs='+', default=[20] ,type=int,
                        help='The sizes of revisers')
    parser.add_argument('--revision_iters', nargs='+', default=[5], type=int,
                        help='Revision iterations (I_n)')
    parser.add_argument('--decode_strategy', type=str, default='greedy', help='decode strategy of the model')
    parser.add_argument('--width', type=int, default=10)
    parser.add_argument('--no_aug', action='store_true', help='Disable instance augmentation')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--val_size', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--steps_per_epoch', type=int, default=256)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--max_norm', type=float, default=1)
    parser.add_argument('--units', type=int, default=48)
    parser.add_argument('--no_clip', action='store_true')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--depth', type=int, default=12)
    #GFN loss: tb, vargrad
    parser.add_argument('--gfn_loss', type=str, default='tb', help='vargrad or tb')
    #beta
    parser.add_argument('--beta', type=float, default=200)
    #logging
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--run_name", type=str, default="", help="Run name")
    #multi-gpu
    parser.add_argument('--gpu', type=int, default=0)

    opts = parser.parse_args()
    opts.no_aug = True
    opts.no_prune = False
    opts.problem_type = 'tsp'

    #################
    # wandb setting
    USE_WANDB = not opts.disable_wandb

    run_name = opts.run_name if opts.run_name != "" else f"GFN-{opts.problem_size}"
    run_name += f"{opts.gfn_loss}-n{opts.problem_size}-s{opts.width}-b{opts.beta}-sd{opts.seed}"
    if USE_WANDB:
        wandb.init(project=f"glopgfn-cvrp", name=run_name, entity='glop-gfn')
        wandb.config.update(opts)
    #################

    # Device
    DEVICE = f'cuda:{opts.gpu}' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(opts.seed)
    pp.pprint(vars(opts))
    train(opts.problem_size, opts.width, opts.steps_per_epoch, opts.n_epochs, opts, beta_schedule_params=opts.beta)