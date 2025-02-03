import warnings
import torch
import numpy as np
import os
import json
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import torch.nn.functional as F
import math
import time

def load_problem(name):
    from problems import TSP, LOCAL
    problem = {
        'local': LOCAL,
        'tsp': TSP,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem

def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""

    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print('  [*] Loading model from {}'.format(load_path))

    load_data = torch.load(
        os.path.join(
            os.getcwd(),
            load_path
        ), map_location=lambda storage, loc: storage)

    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get('optimizer', None)
        load_model_state_dict = load_data.get('model', load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()

    state_dict.update(load_model_state_dict)

    model.load_state_dict(state_dict)

    return model, load_optimizer_state_dict


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_distribution'] = dist[0]
    return args


def load_model(path, epoch=None, is_local=True):

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)
   


    args = load_args(os.path.join(path, 'args.json'))

    
    if is_local:

        from nets.attention_local import AttentionModel
        model= AttentionModel(
        args['embedding_dim'],
        args['hidden_dim'],
        load_problem('local'),
        n_encode_layers=args['n_encode_layers'],
        mask_inner=True,
        mask_logits=True,
        normalization=args['normalization'],
        tanh_clipping=args['tanh_clipping'],
        checkpoint_encoder=args.get('checkpoint_encoder', False),
        shrink_size=args.get('shrink_size', None),
    )

    else:
        raise NotImplementedError

    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})

    model, *_ = _load_model_file(model_filename, model)

    model.eval()  # Put in eval mode

    return model, args


def parse_softmax_temperature(raw_temp):
    # Load from file
    if os.path.isfile(raw_temp):
        return np.loadtxt(raw_temp)[-1, 0]
    return float(raw_temp)


def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True):
    # # Test
    # res = func((directory, 'test', *dataset[0]))
    # return [res]

    assert opts.cpus is not None
    num_cpus = opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval))

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


def do_batch_rep(v, n):
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)

    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


######## LCP-TSP ##########
def decomposition(seeds, coordinate_dim, revision_len, offset, shift_len = 1):
    # change decomposition point
    seeds = torch.cat([seeds[:, shift_len:],seeds[:, :shift_len]], 1)

    if offset!=0:
        decomposed_seeds = seeds[:, :-offset]
        offset_seeds = seeds[:,-offset:]
    else:
        decomposed_seeds = seeds
        offset_seeds = None
    # decompose original seeds
    decomposed_seeds = decomposed_seeds.reshape(-1, revision_len, coordinate_dim)
    return decomposed_seeds, offset_seeds

def coordinate_transformation(x):
    input = x.clone()
    # 좌표와 인덱스를 분리
    coords = input[:, :, :2]  
    indices = input[:, :, 2:] 
    
    # 좌표에 대해서만 변환 수행
    max_x, _ = coords[:,:,0].max(dim=1)
    max_y, _ = coords[:,:,1].max(dim=1)
    min_x, _ = coords[:,:,0].min(dim=1)
    min_y, _ = coords[:,:,1].min(dim=1)
    
    diff_x = max_x - min_x
    diff_y = max_y - min_y
    xy_exchanged = diff_y > diff_x

    coords[:, :, 0] -= (min_x).unsqueeze(-1)
    coords[:, :, 1] -= (min_y).unsqueeze(-1)
    
    coords[xy_exchanged, :, 0], coords[xy_exchanged, :, 1] = coords[xy_exchanged, :, 1], coords[xy_exchanged, :, 0]
    
    scale_degree = torch.max(diff_x, diff_y)
    scale_degree = scale_degree.view(coords.shape[0], 1, 1)
    coords /= scale_degree + 1e-10
    
    # 변환된 좌표와 원래 인덱스를 다시 결합
    return torch.cat([coords, indices], dim=2)

def revision(opts, revision_cost_func, reviser, decomposed_seeds, original_subtour, iter=None, embeddings=None):
    # 좌표와 인덱스 분리
    coords = decomposed_seeds[:, :, :2]
    indices = decomposed_seeds[:, :, 2:]
    
    reviser_size = original_subtour.shape[0]
    # cost 계산은 좌표만 사용
    init_cost = revision_cost_func(coords, original_subtour)
    
    # coordinate transformation
    transformed = coordinate_transformation(decomposed_seeds)
    transformed_coords = transformed[:, :, :2]
    transformed_indices = transformed[:, :, 2:]
    
    # augmentation
    if not opts.no_aug:
        seed2_coords = torch.cat((1 - transformed_coords[:, :, [0]], transformed_coords[:, :, [1]]), dim=2)
        seed3_coords = torch.cat((transformed_coords[:, :, [0]], 1 - transformed_coords[:, :, [1]]), dim=2)
        seed4_coords = torch.cat((1 - transformed_coords[:, :, [0]], 1 - transformed_coords[:, :, [1]]), dim=2)
        # 인덱스도 같이 복제
        augmented_coords = torch.cat((transformed_coords, seed2_coords, seed3_coords, seed4_coords), dim=0)
        augmented_indices = transformed_indices.repeat(4, 1, 1)
        augmented_seeds = torch.cat([augmented_coords, augmented_indices], dim=2)
    else:
        augmented_seeds = transformed

    # reviser 호출 (좌표만 사용)
    if iter is None:
        cost_revised1, sub_tour1, cost_revised2, sub_tour2 = reviser(augmented_seeds[:, :, :2], return_pi=True)
    elif iter == 0:
        cost_revised1, sub_tour1, cost_revised2, sub_tour2, embeddings = reviser(augmented_seeds[:, :, :2], return_pi=True, return_embedding=True)
    else:
        cost_revised1, sub_tour1, cost_revised2, sub_tour2 = reviser(augmented_seeds[:, :, :2], return_pi=True, embeddings=embeddings)
    
    # sub_tour 선택 로직은 동일
    if not opts.no_aug:
        _, better_tour_idx = torch.cat([cost_revised1, cost_revised2], dim=0).reshape(8,-1).min(dim=0)
        sub_tour = torch.cat([sub_tour1, sub_tour2], dim=0).reshape(8,-1, reviser_size)[better_tour_idx, torch.arange(sub_tour1.shape[0]//4), :]
    else:
        _, better_tour_idx = torch.stack((cost_revised1, cost_revised2)).min(dim=0)
        sub_tour = torch.stack((sub_tour1, sub_tour2))[better_tour_idx, torch.arange(sub_tour1.shape[0])]
    # cost는 좌표만으로 계산
    cost_revised, _ = reviser.problem.get_costs(coords, sub_tour)
    reduced_cost = init_cost - cost_revised
    
    sub_tour[reduced_cost < 0] = original_subtour
    # 좌표와 인덱스 모두 재배열
    decomposed_seeds = decomposed_seeds.gather(1, sub_tour.unsqueeze(-1).expand_as(decomposed_seeds))

    if embeddings is not None:
        if not opts.no_aug:
            embeddings = embeddings.gather(1, sub_tour.repeat(4, 1).unsqueeze(-1).expand_as(embeddings))
        else:
            embeddings = embeddings.gather(1, sub_tour.unsqueeze(-1).expand_as(embeddings))
    return decomposed_seeds, embeddings

def LCP_TSP(
    seeds,
    cost_func,
    reviser,
    revision_len,
    revision_iter,
    opts,
    shift_len
    ):
    
    batch_size, num_nodes, coordinate_dim = seeds.shape
    # print(f'shape of seeds : {seeds.shape}')
    offset = num_nodes % revision_len
    # print(f'offset : {offset}')
    embeddings = None # used only in case problem_size == revision_len for efficiency
    for i in range(revision_iter):

        decomposed_seeds, offset_seed = decomposition(seeds, 
                                        coordinate_dim,
                                        revision_len,
                                        offset,
                                        shift_len
                                        )
        # print(f'decomposed_seeds shape : {decomposed_seeds.shape}')
        # print(f'decomposed_seeds : {decomposed_seeds[0]}')
        original_subtour = torch.arange(0, revision_len, dtype=torch.long).to(decomposed_seeds.device)
        # print(f'original_subtour : {original_subtour}')

        if revision_len == num_nodes:
            decomposed_seeds_revised, embeddings = revision(opts, cost_func, reviser, decomposed_seeds, original_subtour, iter=i, embeddings=embeddings)
            embeddings = torch.cat([embeddings[:, shift_len:],embeddings[:, :shift_len]], 1) # roll the embeddings
        else:
            decomposed_seeds_revised, _ = revision(opts, cost_func, reviser, decomposed_seeds, original_subtour)
        # print(f'decomposed_seeds_revised shape : {decomposed_seeds_revised.shape}')
        # print(f'decomposed_seeds_revised : {decomposed_seeds_revised[0]}')
        seeds = decomposed_seeds_revised.reshape(batch_size, -1, coordinate_dim) 
        # print(f'decomposed seeds reshape : {seeds.shape}')
        if offset_seed is not None:
            seeds = torch.cat([seeds,offset_seed], dim=1)
    return seeds


def reconnect( 
        get_cost_func,
        batch,
        opts, 
        revisers,
    ):
    seed = batch
    problem_size = seed.size(1) 
    if len(revisers) == 0:

        # 좌표만 사용하여 cost 계산
        cost_revised = (seed[:, 1:, :2] - seed[:, :-1, :2]).norm(p=2, dim=2).sum(1) + (seed[:, 0, :2] - seed[:, -1, :2]).norm(p=2, dim=1)
    
    for revision_id in range(len(revisers)):
        assert opts.revision_lens[revision_id] <= seed.size(1)
        start_time = time.time()
        shift_len = max(opts.revision_lens[revision_id]//opts.revision_iters[revision_id], 1)
        seed = LCP_TSP(
            seed, 
            get_cost_func,
            revisers[revision_id],
            opts.revision_lens[revision_id],
            opts.revision_iters[revision_id],
            opts=opts,
            shift_len=shift_len
            )
        # print(f'LCP_TSP seed : {seed[0]}')

        # 좌표만 사용하여 cost 계산
        cost_revised = (seed[:, 1:, :2] - seed[:, :-1, :2]).norm(p=2, dim=2).sum(1) + (seed[:, 0, :2] - seed[:, -1, :2]).norm(p=2, dim=1)      
        duration = time.time() - start_time
        
        if revision_id == 0 and not opts.no_prune:
            cost_revised, cost_revised_minidx = cost_revised.reshape(-1, opts.eval_batch_size).min(0)
            # coordinate_dim + 1로 수정 
            seed = seed.reshape(-1, opts.eval_batch_size, seed.shape[-2], seed.shape[-1])[cost_revised_minidx, torch.arange(opts.eval_batch_size)]
    if opts.no_prune:
            cost_revised, cost_revised_minidx = cost_revised.reshape(-1, opts.eval_batch_size).min(0)
            # coordinate_dim + 1로 수정 
            seed = seed.reshape(-1, opts.eval_batch_size, seed.shape[-2], seed.shape[-1])[cost_revised_minidx, torch.arange(opts.eval_batch_size)]
    assert cost_revised.shape == (opts.eval_batch_size,)
    # coordinate_dim + 1로 수정
    assert seed.shape == (opts.eval_batch_size, problem_size, seed.shape[-1])
    seed = reorder_tour(seed)
    # print(f'shape of seed:{seed.shape}')
    # print(f'gather : {gather[0]}')
    # print(f'gather shape : {gather.shape}')

    return seed, cost_revised


def sample_many():
    raise NotImplementedError

def reorder_tour(tour):
    """
    tour를 재배열 (순환 구조 활용):
    1. tour를 두 번 반복하여 연결
    2. 마지막 0 노드부터 원래 길이만큼 슬라이싱
    
    Args:
        tour: shape (batch_size, num_nodes, feature_dim)
    """
    device = tour.device
    batch_size, num_nodes, feature_dim = tour.shape
    
    # tour를 두 번 반복하여 연결
    doubled_tour = torch.cat([tour, tour], dim=1)  # (batch_size, 2*num_nodes, feature_dim)
    
    # 각 배치에서 마지막 0의 위치 찾기 (원래 tour에서)
    zero_mask = tour[..., -1] == 0
    last_zero_idx = (num_nodes - 1) - torch.flip(zero_mask, [1]).float().argmax(dim=1)  # (batch_size,)
    
    # 노드 인덱스 생성
    node_indices = torch.arange(num_nodes, device=device).unsqueeze(0)    # (1, num_nodes)
    indices = last_zero_idx.unsqueeze(1) + node_indices                   # (batch_size, num_nodes)
    
    # gather로 한 번에 슬라이싱
    reordered_tour = doubled_tour.gather(
        1, 
        indices.unsqueeze(-1).expand(-1, -1, feature_dim)
    )
    
    return reordered_tour




