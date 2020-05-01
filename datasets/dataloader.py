"""
Reference:
- https://github.com/ildoonet/ai-starthon-2019/blob/master/cls4%2B7%2B8%2B9/utils.py
"""

import torch
from torch.utils.data import DataLoader

from .datasets import RetinaDataset
from .stratified_sampler import StratifiedSampler


def get_dataloader(config, split, transform=None):
    dataset = RetinaDataset(config, split, transform)

    is_train = 'train' == split

    if is_train:
        batch_size = config.TRAIN.BATCH_SIZE
        steps = len(dataset) // batch_size

        sampler = StratifiedSampler(dataset.labels)
        dataloader = FixedSizeDataLoader(dataset, steps=steps, batch_size=batch_size, num_workers=config.NUM_WORKERS,
                                         drop_last=True, sampler=sampler)
        dataloader = PrefetchDataLoader(dataloader, device=torch.device('cuda', 0))

    else:
        dataloader = DataLoader(dataset,
                                 shuffle=False,
                                 batch_size=config.EVAL.BATCH_SIZE,
                                 num_workers=config.NUM_WORKERS,
                                 pin_memory=True)

    return dataloader


class FixedSizeDataLoader:
    def __init__(self, dataset, steps, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False,
                 sampler=None):
        sampler = InfiniteSampler(dataset, shuffle) if sampler is None else sampler
        self.batch_size = batch_size
        batch_size = 1 if batch_size is None else batch_size

        self.steps = steps
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )

    def __len__(self):
        return self.steps

    def __iter__(self):
        if self.steps is not None:
            for _, data in zip(range(self.steps), self.dataloader):
                yield ([t[0] for t in data] if self.batch_size is None else data)
        else:
            for data in self.dataloader:
                yield ([t[0] for t in data] if self.batch_size is None else data)


class InfiniteSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.data_source)
        while True:
            index_list = torch.randperm(n).tolist() if self.shuffle else list(range(n))
            for idx in index_list:
                yield idx

    def __len__(self):
        return len(self.data_source)


class PrefetchDataLoader:
    def __init__(self, dataloader, device):
        self.loader = dataloader
        self.iter = None
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_data =None

    def __len__(self):
        return len(self.loader)

    def async_prefech(self):
        try:
            self.next_data = next(self.iter)
        except StopIteration:
            self.next_data = None
            return

        with torch.cuda.stream(self.stream):
            if isinstance(self.next_data, torch.Tensor):
                self.next_data = self.next_data.to(device=self.device, non_blocking=True)
            elif isinstance(self.next_data, (list, tuple)):
                self.next_data = [t.to(device=self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t for t in self.next_data]

    def __iter__(self):
        self.iter = iter(self.loader)
        self.async_prefech()
        while self.next_data is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            data = self.next_data
            self.async_prefech()
            yield data
