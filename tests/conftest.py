import torch


def pytest_sessionstart(session):
    # Small CPU fixtures run faster without launching a large thread pool.
    torch.set_num_threads(1)
