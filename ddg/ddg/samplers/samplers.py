import torch
from typing import Iterator, Sized
from torch.utils.data import Sampler
from ddg.datasets import DomainFolder
from ddg.utils import SAMPLERS_REGISTRY


__all__ = ['RandomDomainSampler']


@SAMPLERS_REGISTRY.register()
class RandomDomainSampler(Sampler):
    r"""Samples elements randomly from each domain equally and ensure two adjacent samples are definitely from
    different domains.

    Args:
        data_source (DomainFolder): dataset to sample from
        generator (Generator): Generator used in sampling.

    Note: Not for DDP, if you want the random domain sampler to work on DDP,
    you need to write another sampler by inherit DistributedSampler.
    """
    data_source: Sized

    def __init__(self, data_source: DomainFolder, generator=None) -> None:
        super(RandomDomainSampler, self).__init__(data_source=data_source)
        self.data_source = data_source
        self.generator = generator

        self.n_iter = None
        for data in self.data_source.datasets:
            self.n_iter = len(data) if not self.n_iter else min(self.n_iter, len(data))
        self.num_samples = self.n_iter * len(self.data_source.domains)

    def __iter__(self) -> Iterator[int]:
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        perms = []
        for data in self.data_source.datasets:
            perms.append(torch.randperm(len(data), generator=generator))
        last_domain = -1
        for idx in range(self.n_iter):
            domains = torch.randperm(len(self.data_source.domains), generator=generator)
            domains = torch.flip(domains, dims=[0]) if domains[0] == last_domain else domains
            last_domain = domains[-1]
            for i in range(len(self.data_source.domains)):
                yield perms[domains[i]][idx] + (self.data_source.cumulative_sizes[domains[i] - 1] if domains[i] else 0)

    def __len__(self) -> int:
        return self.num_samples
