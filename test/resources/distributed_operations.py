import logging
import os
import torch
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _get_tensor(rank, rows, columns):
    tensor = torch.ones(rows, columns) * (rank + 1)
    return tensor.cuda() if torch.cuda.is_available() else tensor


def _get_zeros_tensor(rows, columns):
    tensor = torch.zeros(rows, columns)
    return tensor.cuda() if torch.cuda.is_available() else tensor


def _get_zeros_tensors_list(rows, columns):
    return [_get_zeros_tensor(rows, columns) for _ in range(dist.get_world_size())]


def _get_tensors_sum(rows, columns):
    result = 0
    for i in range(dist.get_world_size()):
        result += i + 1
    tensor = torch.ones(rows, columns) * result
    return tensor.cuda() if torch.cuda.is_available() else tensor


def _send_recv(rank, rows, columns):
    source = 0
    tensor = _get_tensor(rank, rows, columns)
    logger.debug('Rank: {},\nTensor BEFORE send_recv: {}'.format(rank, tensor))
    if rank == 0:
        for i in range(1, dist.get_world_size()):
            dist.send(tensor=tensor, dst=i)
    else:
        dist.recv(tensor=tensor, src=source)
    logger.debug('Rank: {},\nTensor AFTER send_recv: {}\n'.format(rank, tensor))

    assert torch.equal(tensor, _get_tensor(source, rows, columns)),\
        'Rank {}: Tensor was not equal to rank {} tensor after send-recv.'.format(rank, source)


def _broadcast(rank, rows, columns):
    source = 0
    tensor = _get_tensor(rank, rows, columns)
    logger.debug('Rank: {},\nTensor BEFORE broadcast: {}'.format(rank, tensor))
    dist.broadcast(tensor, src=source)
    logger.debug('Rank: {},\nTensor AFTER broadcast: {}\n'.format(rank, tensor))

    assert torch.equal(tensor, _get_tensor(source, rows, columns)), \
        'Rank {}: Tensor was not equal to rank {} tensor after broadcast.'.format(rank, source)


def _all_reduce(rank, rows, columns):
    tensor = _get_tensor(rank, rows, columns)
    logger.debug('Rank: {},\nTensor BEFORE all_reduce: {}'.format(rank, tensor))
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    logger.debug('Rank: {},\nTensor AFTER all_reduce: {}\n'.format(rank, tensor))

    assert torch.equal(tensor, _get_tensors_sum(rows, columns)), \
        'Rank {}: Tensor was not equal to SUM of {} tensors after all_reduce.'.format(rank, dist.get_world_size())


def _reduce(rank, rows, columns):
    dest = 0
    tensor = _get_tensor(rank, rows, columns)
    logger.debug('Rank: {},\nTensor BEFORE reduce: {}'.format(rank, tensor))
    # this is inplace operation
    dist.reduce(tensor, op=dist.reduce_op.SUM, dst=dest)
    logger.debug('Rank: {},\nTensor AFTER reduce: {}\n'.format(rank, tensor))

    if rank == dest:
        assert torch.equal(tensor, _get_tensors_sum(rows, columns)), \
            'Rank {}: Tensor was not equal to SUM of {} tensors after reduce.'.format(rank, dist.get_world_size())


def _all_gather(rank, rows, columns):
    tensor = _get_tensor(rank, rows, columns)
    tensors_list = _get_zeros_tensors_list(rows, columns)
    logger.debug('Rank: {},\nTensor BEFORE all_gather: {}'.format(rank, tensor))
    dist.all_gather(tensors_list, tensor)
    logger.debug('Rank: {},\nTensor AFTER all_gather: {}. tensors_list: {}\n'.format(
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
    tensors_list = _get_zeros_tensors_list(rows, columns)
    logger.debug('Rank: {},\nTensor BEFORE gather: {}. tensors_list: {}'.format(
        rank, tensor, tensors_list))
    if rank == dest:
        dist.gather(tensor=tensor, gather_list=tensors_list)
    else:
        dist.gather(tensor=tensor, dst=dest)
    logger.debug('Rank: {},\nTensor AFTER gather: {}. tensors_list: {}\n'.format(
        rank, tensor, tensors_list))

    # tensor shouldn't have changed
    assert torch.equal(tensor, _get_tensor(rank, rows, columns)), \
        'Rank {}: Tensor got changed after gather.'.format(rank)


def _scatter(rank, rows, columns):
    source = 0
    tensor = _get_tensor(rank, rows, columns)
    tensors_list = _get_zeros_tensors_list(rows, columns)
    logger.debug('Rank: {},\nTensor BEFORE scatter: {}. tensors_list: {}'.format(
        rank, tensor, tensors_list))
    if rank == source:
        dist.scatter(tensor=tensor, scatter_list=tensors_list)
    else:
        dist.scatter(tensor=tensor, src=source)
    logger.debug('Rank: {},\nTensor AFTER scatter: {}\n'.format(rank, tensor))

    assert torch.equal(tensor, _get_zeros_tensor(rows, columns)), \
        'Rank {}: Tensor should be all zeroes after scatter.'.format(rank)


def _barrier(rank):
    logger.debug('Rank: {}, Waiting for other processes before the barrier.'.format(rank))
    dist.barrier()
    logger.debug('Rank: {}, Passing the barrier'.format(rank))


def train(rank, world_size, hyperparameters):
    # Initialize the distributed environment.
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'algo-1'
    os.environ['MASTER_PORT'] = '29500'
    backend = hyperparameters.get('backend')
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    logger.info('Running \'{}\' backend on {} nodes. Current host rank is {}. Using cuda: {}'.format(
        backend, dist.get_world_size(), dist.get_rank(), torch.cuda.is_available()))

    rows = hyperparameters.get('rows', 1)
    columns = hyperparameters.get('columns', 1)

    # operations supported by all backends: http://pytorch.org/docs/master/distributed.html
    _broadcast(rank, rows, columns)
    _all_reduce(rank, rows, columns)
    _barrier(rank)

    # operations not supported by 'gloo'
    if backend != 'gloo':
        _send_recv(rank, rows, columns)
        _reduce(rank, rows, columns)
        _all_gather(rank, rows, columns)
        _gather(rank, rows, columns)
        _scatter(rank, rows, columns)
