# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import logging
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.utils.data
import torch.utils.data.distributed

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_tensor(rank, rows, columns):
    device = torch.device("cuda:{}".format(rank) if torch.cuda.is_available() else "cpu")
    print(device)
    tensor = torch.ones(rows, columns) * (rank + 1)
    print('{}: tensor:{}'.format(rank, tensor.to(device)))
    return tensor.to(device)


def _get_zeros_tensor(rows, columns):
    device = torch.device("cuda:{}".format(dist.get_rank()) if torch.cuda.is_available() else "cpu")
    tensor = torch.zeros(rows, columns)
    return tensor.to(device)


def _get_zeros_tensors_list(rows, columns):
    return [_get_zeros_tensor(rows, columns) for _ in range(dist.get_world_size())]


def _get_tensors_sum(rows, columns):
    device = torch.device("cuda:{}".format(dist.get_rank()) if torch.cuda.is_available() else "cpu")
    result = (1 + dist.get_world_size()) * dist.get_world_size() / 2
    tensor = torch.ones(rows, columns) * result
    return tensor.to(device)


def _send_recv(rank, rows, columns):
    source = 0
    tensor = _get_tensor(rank, rows, columns)
    print ('Rank: {},\nTensor BEFORE send_recv: {}'.format(rank, tensor))
    if rank == 0:
        for i in range(1, dist.get_world_size()):
            dist.send(tensor=tensor, dst=i)
    else:
        dist.recv(tensor=tensor, src=source)
    print ('Rank: {},\nTensor AFTER send_recv: {}\n'.format(rank, tensor))

    assert torch.equal(tensor, _get_tensor(source, rows, columns)),\
        'Rank {}: Tensor was not equal to rank {} tensor after send-recv.'.format(rank, source)


def _broadcast(rank, rows, columns):
    source = 0
    tensor = _get_tensor(rank, rows, columns)
    print ('Rank: {},\nTensor BEFORE broadcast: {}'.format(rank, tensor))
    dist.broadcast(tensor, src=source)
    print ('Rank: {},\nTensor AFTER broadcast: {}\n'.format(rank, tensor))

    assert torch.equal(tensor, _get_tensor(source, rows, columns)), \
        'Rank {}: Tensor was not equal to rank {} tensor after broadcast.'.format(rank, source)


def _all_reduce(rank, rows, columns):
    tensor = _get_tensor(rank, rows, columns)
    print ('Rank: {},\nTensor BEFORE all_reduce: {}'.format(rank, tensor))
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    print ('Rank: {},\nTensor AFTER all_reduce: {}\n'.format(rank, tensor))

    assert torch.equal(tensor, _get_tensors_sum(rows, columns)), \
        'Rank {}: Tensor was not equal to SUM of {} tensors after all_reduce.'.format(rank, dist.get_world_size())


def _reduce(rank, rows, columns):
    dest = 0
    tensor = _get_tensor(rank, rows, columns)
    print ('Rank: {},\nTensor BEFORE reduce: {}'.format(rank, tensor))
    # this is inplace operation
    dist.reduce(tensor, op=dist.reduce_op.SUM, dst=dest)
    print ('Rank: {},\nTensor AFTER reduce: {}\n'.format(rank, tensor))

    if rank == dest:
        assert torch.equal(tensor, _get_tensors_sum(rows, columns)), \
            'Rank {}: Tensor was not equal to SUM of {} tensors after reduce.'.format(rank, dist.get_world_size())


def _all_gather(rank, rows, columns):
    tensor = _get_tensor(rank, rows, columns)
    tensors_list = _get_zeros_tensors_list(rows, columns)
    print ('Rank: {},\nTensor BEFORE all_gather: {}'.format(rank, tensor))
    dist.all_gather(tensors_list, tensor)
    print ('Rank: {},\nTensor AFTER all_gather: {}. tensors_list: {}\n'.format(
        rank, tensor, tensors_list))

    # tensor shouldn't have changed
    assert torch.equal(tensor, _get_tensor(rank, rows, columns)), \
        'Rank {}: Tensor got changed after all_gather.'.format(rank)
    for i in range(dist.get_world_size()):
        assert torch.equal(tensors_list[i], _get_tensor(i, rows, columns)), \
            'Rank {}: tensors lists are not the same after all_gather.'


def _gather(rank, rows, columns):
    dest = 0
    tensor = _get_tensor(rank, rows, columns)
    if rank == dest:
        tensors_list = _get_zeros_tensors_list(rows, columns)
        print ('Rank: {},\nTensor BEFORE gather: {}. tensors_list: {}'.format(
            rank, tensor, tensors_list))
        dist.gather(tensor=tensor, gather_list=tensors_list)
        print ('Rank: {},\nTensor AFTER gather: {}. tensors_list: {}\n'.format(
            rank, tensor, tensors_list))
        for i in range(dist.get_world_size()):
            assert torch.equal(tensors_list[i], _get_tensor(i, rows, columns)), \
                'Rank {}: tensors lists are not the same after gather.'
    else:
        print ('Rank: {},\nTensor BEFORE gather: {}\n'.format(rank, tensor))
        dist.gather(tensor=tensor, dst=dest)
        print ('Rank: {},\nTensor AFTER gather: {}\n'.format(rank, tensor))

    # tensor shouldn't have changed
    assert torch.equal(tensor, _get_tensor(rank, rows, columns)), \
        'Rank {}: Tensor got changed after gather.'.format(rank)


def _scatter(rank, rows, columns):
    source = 0
    tensor = _get_tensor(rank, rows, columns)
    if rank == source:
        tensors_list = _get_zeros_tensors_list(rows, columns)
        print ('Rank: {},\nTensor BEFORE scatter: {}. tensors_list: {}'.format(
            rank, tensor, tensors_list))
        dist.scatter(tensor=tensor, scatter_list=tensors_list)
    else:
        print ('Rank: {},\nTensor BEFORE scatter: {}\n'.format(rank, tensor))
        dist.scatter(tensor=tensor, src=source)
    print ('Rank: {},\nTensor AFTER scatter: {}\n'.format(rank, tensor))

    assert torch.equal(tensor, _get_zeros_tensor(rows, columns)), \
        'Rank {}: Tensor should be all zeroes after scatter.'.format(rank)


def _barrier(rank):
    print ('Rank: {}, Waiting for other processes before the barrier.'.format(rank))
    dist.barrier()
    print ('Rank: {}, Passing the barrier'.format(rank))


def train(master_addr, master_port, current_host, host_rank, num_gpus, hosts, num_cpus, hyperparameters):
    backend = hyperparameters.get('backend')
    rows = hyperparameters.get('rows', 1)
    columns = hyperparameters.get('columns', 1)
    cuda = torch.cuda.is_available()
    number_of_processes = num_gpus if cuda else num_cpus
    world_size = number_of_processes * len(hosts)
    print ('Running \'{}\' backend on {} nodes and {} processes. World size is {}. Using cuda: {}'.format(
        backend, len(hosts), number_of_processes, world_size, cuda
    ))

    processes = []
    for rank in range(number_of_processes):
        process_rank = host_rank * number_of_processes + rank
        p = Process(
            target=init_processes,
            args=(backend, master_addr, master_port, process_rank, world_size, rows, columns, current_host)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return 'success'


def init_processes(backend, master_addr, master_port, rank, world_size, rows, columns, host):
    # Initialize the distributed environment.
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    print ('Init process rank {} on host \'{}\''.format(rank, host))
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    #run(backend, rank, rows, columns)
    _broadcast(rank, rows, columns)


def run(backend, rank, rows, columns):
    # http://pytorch.org/docs/master/distributed.html
    if backend == 'tcp':
        print ('Run operations supported by \'tcp\' backend.')
        _broadcast(rank, rows, columns)
        _all_reduce(rank, rows, columns)
        _barrier(rank)
        _send_recv(rank, rows, columns)
        _reduce(rank, rows, columns)
        _all_gather(rank, rows, columns)
        _gather(rank, rows, columns)
        _scatter(rank, rows, columns)
    elif backend == 'gloo':
        print ('Run operations supported by \'gloo\' backend.')
        _broadcast(rank, rows, columns)
        _all_reduce(rank, rows, columns)
        _barrier(rank)
    elif backend == 'nccl':
        print ('Run operations supported by \'nccl\' backend.')
        _broadcast(rank, rows, columns)
        _all_reduce(rank, rows, columns)
        _reduce(rank, rows, columns)
        _all_gather(rank, rows, columns)
        _gather(rank, rows, columns)
        _scatter(rank, rows, columns)


def save(model, model_dir):
    filename = os.path.join(model_dir, model)
    if not os.path.exists(filename):
        print ("Saving success result")
        with open(filename, 'w') as f:
            f.write(model)


def test_dist():
    master_addr = '127.0.0.1'
    train(
        master_addr=master_addr,
        master_port='29900',
        current_host=master_addr,
        host_rank=0,
        num_gpus=2,
        hosts=['master_addr'],
        num_cpus=32,
        hyperparameters = {
            'backend': 'gloo'
        }
    )


def test_dist2():
    size = 2
    processes = []
    backend = 'gloo'
    master_addr = '127.0.0.1'
    master_port = '29700'
    world_size = 2
    rows = 1
    columns =1
    host = '127.0.0.1'
    for rank in range(size):
        p = Process(target=init_processes, args=(
            backend, master_addr, master_port, rank, world_size, rows, columns, host
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()