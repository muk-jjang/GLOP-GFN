import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from utils import load_model
from torch.utils.data import DataLoader
import time
from utils.functions import parse_softmax_temperature
from utils.functions import reconnect
from utils.functions import load_problem
from utils.functions import LCP_TSP
from problems.tsp.tsp_baseline import solve_insertion

mp = torch.multiprocessing.get_context('spawn')


def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]


def eval_dataset(dataset_path, width, softmax_temp, opts):
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    device = torch.device("cuda:0" if use_cuda else "cpu")

    revisers = []
    revisers2 = []
    assert opts.problem == 'tsp'
    revision_lens = opts.revision_lens
    revision_lens2 = opts.revision_lens2

    for reviser_size in revision_lens:
        reviser_path = f'pretrained_LCP/constructions/Reviser-6-FI/reviser_{reviser_size}/epoch-199.pt'
        if reviser_size in [50, 100]:
            reviser_path = f'pretrained_LCP/constructions/Reviser-6-FI/reviser_{reviser_size}/epoch-299.pt'
        reviser, _ = load_model(reviser_path, is_local=True)
        revisers.append(reviser)
        
    for reviser in revisers:
        reviser.to(device)
        reviser.eval()
        reviser.set_decode_type("greedy")
    
    for revision_size in revision_lens2:
        _path = f'pretrained_LCP/improvements/C2/TSP{revision_size}-model-30.pt'
        reviser2 = torch.load(_path, map_location=device)
        reviser2.eval()
        print('  [*] Loading improvement model from {}'.format(_path))
        revisers2.append(reviser2)

    dataset = reviser.problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=opts.offset)
    results = _eval_dataset(dataset, width, softmax_temp, opts, device, revisers, revisers2)

    costs, costs_revised, tours, durations = zip(*results)  # Not really costs since they should be negative
    
    costs = torch.tensor(costs)
    costs_revised = torch.cat(costs_revised, dim=0)
    tours = torch.cat(tours, dim=0)

    print("Average cost: {} +- {}".format(costs.mean(), (2 * torch.std(costs) / math.sqrt(len(costs))).item()))
    print("Average cost_revised: {} +- {}".format(costs_revised.mean().item(), (2 * torch.std(costs_revised) / math.sqrt(len(costs_revised))).item()))
    print("Total duration: {}".format(sum(durations)))
    print("Avg. duration: {}".format(sum(durations) / opts.val_size))

    return costs, tours, durations


def _eval_dataset(dataset, width, softmax_temp, opts, device, revisers, revisers2):

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)
    problem = load_problem(opts.problem_type)
    # cost function for complete solution
    get_cost_func = lambda input, pi: problem.get_costs(input, pi)
    # cost function for partial solution 
    get_cost_func2 = lambda input, pi: problem.get_costs(input, pi, return_local=True)

    results = []
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        # tsp batch shape: (bs, problem size, 2)
        avg_cost = 0
        start = time.time()
        with torch.no_grad():
                
            pi_batch = torch.LongTensor(size=(opts.eval_batch_size, opts.problem_size))

            for instance_id, instance in enumerate(batch):

                cost, pi, duration = solve_insertion(
                                        directory=None, 
                                        name=None, 
                                        loc=instance,
                                        method='farthest',
                                        )
                avg_cost += cost
                pi_batch[instance_id] = torch.tensor(pi)
            avg_cost /= opts.eval_batch_size

            pi_batch = pi_batch
            batch = batch
            seed = batch.gather(1, pi_batch.unsqueeze(-1).expand_as(batch))
            seed = seed.to(device)
        ##################################
            
            start_time = time.time()
            
            for revision_id in range(len(revisers2)):
                seed = LCP_TSP(
                    seed, 
                    get_cost_func2,
                    revisers2[revision_id],
                    opts.revision_lens2[revision_id],
                    opts.revision_iters2[revision_id],
                    mood='improve',
                    opts = opts,
                    shift_len=opts.shift_lens2[revision_id]
                    )
            
                duration = time.time() - start_time
                cost_revised = (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1) + (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)

                cost_revised, _ = cost_revised.reshape(-1, opts.eval_batch_size).min(0)
                print(f'after improvement {revision_id}', cost_revised.mean().item(), f'duration {duration} \n')

            ##############################################################
            if opts.aug:
                
                if opts.aug_shift > 1:
                    # batch = torch.cat([torch.roll(batch, i, 1) for i in range(0, opts.aug_shift)], dim=0)
                    # batch = batch.repeat(opts.aug_shift, 1, 1)
                    seed = torch.cat([torch.roll(seed, i, 1) for i in range(0, opts.aug_shift)], dim=0)
                seed2 = torch.cat((1 - seed[:, :, [0]], seed[:, :, [1]]), dim=2)
                seed3 = torch.cat((seed[:, :, [0]], 1 - seed[:, :, [1]]), dim=2)
                seed4 = torch.cat((1 - seed[:, :, [0]], 1 - seed[:, :, [1]]), dim=2)
                seed = torch.cat((seed, seed2, seed3, seed4), dim=0)
                
                print('\n seed shape after augmentation:', seed.shape)

            # needed shape: (width, graph_size, 2) / (width, graph_size)
            
            tours, costs_revised = reconnect(
                                        get_cost_func=get_cost_func, 
                                        get_cost_func2=get_cost_func2,
                                        batch=seed,
                                        opts=opts,
                                        revisers=revisers,
                                        revisers2=revisers2
                                        )
            # tours shape: problem_size, 2
            # costs: costs before revision
            # costs_revised: costs after revision

        duration = time.time() - start

        if costs_revised is not None:
            results.append((avg_cost, costs_revised, tours, duration))
        else:
            results.append((avg_cost, None, tours, duration))

    return results


if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_size", type=int, default=200)
    parser.add_argument("--problem_type", type=str, default='tsp')
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=16,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=2,
                        help="Softmax temperature (sampling or bs)")
    parser.add_argument('--revision_lens', nargs='+', default=[100,50,20] ,type=int)
    parser.add_argument('--revision_iters', nargs='+', default=[10,25,20], type=int)
    parser.add_argument('--shift_lens', nargs='+', default=[10,2,1], type=int)
    parser.add_argument('--shift_lens2', nargs='+', default=[1], type=int)
    parser.add_argument('--top_k', nargs='+', default=[1,1], type=float)
    parser.add_argument('--revision_lens2', nargs='+', default=[50] ,type=int)
    parser.add_argument('--revision_iters2', nargs='+', default=[1], type=int)
    parser.add_argument('--problem', default='tsp', type=str)
    parser.add_argument('--decode_strategy', type=str, default='sample', help='decode strategy of the model')
    parser.add_argument('--width', type=int, default=1, help='number of candidate solutions / seeds (M)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--aug', action='store_true', help='Apply instance augmentation')
    parser.add_argument('--max_calc_batch_size', type=int, default=10000, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Use multiprocessing to parallelize over multiple GPUs')
    parser.add_argument('--n_steps', default=20, type=int, help='Number of steps in each episode')
    parser.add_argument('--disable_improve', action='store_true', help='Disable improvement revisions')
    parser.add_argument('--improve_shift', default=10, type=int,
           help='The length of tour shift when each time decomposing for improvement')
    parser.add_argument('--aug_shift', default=1, type=int, help='')
    
    opts = parser.parse_args()

    assert opts.o is None or (len(opts.datasets) == 1 and len(opts.width) <= 1), \
        "Cannot specify result filename with more than one dataset or more than one width"

    dataset_path = f'data/tsp/tsp{opts.problem_size}_test.pkl'
    eval_dataset(dataset_path, opts.width, opts.softmax_temperature, opts)
