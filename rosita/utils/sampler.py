import math, os, json, torch, datetime, random, copy, shutil, logging, socket
import torch.utils.data as Data
import torch.distributed as dist


class SubsetDistributedSampler(Data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, subset_indices=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        if subset_indices:
            self.subset_indices = subset_indices
        else:
            self.subset_indices = list(range(len(self.dataset)))
        self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.num_samples = int(math.ceil(len(self.subset_indices) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.rest_data_num = self.total_size - len(self.subset_indices)

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.subset_indices), generator=g).tolist()
        else:
            indices = list(range(len(self.subset_indices)))

        # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        indices += indices[:self.rest_data_num]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # return iter(indices)
        return (self.subset_indices[i] for i in indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_shuffle(self, shuffle):
        assert shuffle in [True, False]
        self.shuffle = shuffle
