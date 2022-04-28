import time
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from ddg.trainer import Trainer
from ddg.utils import DATASET_REGISTRY
from ddg.utils import SAMPLERS_REGISTRY

__all__ = ['TrainerDG']


class TrainerDG(Trainer):
    """Base trainer provided by DDG, one can inherit it to implement your own trainer.
    """
    TASK = "DG"

    def build_dataset(self):
        args = self.args
        self.datasets['train'] = DATASET_REGISTRY.get(args.dataset)(root=args.root,
                                                                    domains=set(args.source_domains),
                                                                    splits={'train'},
                                                                    transform=self.transform.train)
        self.datasets['val'] = DATASET_REGISTRY.get(args.dataset)(root=args.root,
                                                                  domains=set(args.source_domains),
                                                                  splits={'val'},
                                                                  transform=self.transform.test)
        self.datasets['test'] = DATASET_REGISTRY.get(args.dataset)(root=args.root,
                                                                   domains=set(args.target_domains),
                                                                   splits={'test'},
                                                                   transform=self.transform.test)
        args.num_classes = self.datasets['train'].num_classes

    def build_data_loader(self):
        args = self.args
        if args.sampler is None:
            sampler = DistributedSampler(self.datasets['train']) if self.args.distributed else None
            args.sampler = 'DistributedSampler'
        elif args.sampler in SAMPLERS_REGISTRY:
            sampler = SAMPLERS_REGISTRY.get(args.sampler)(self.datasets['train'])
        else:
            raise NotImplementedError(f"Sampler name {args.sampler} is not implemented yet!")
        self.samplers['train'] = sampler
        self.data_loaders['train'] = DataLoader(self.datasets['train'], batch_size=args.batch_size,
                                                shuffle=(sampler is None),
                                                num_workers=args.workers, pin_memory=True, sampler=sampler)
        self.data_loaders['val'] = DataLoader(self.datasets['val'], batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
        self.data_loaders['test'] = DataLoader(self.datasets['test'], batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True)

    def run_epoch(self):
        args = self.args
        progress = self.progress['train']
        end = time.time()
        for i, data in enumerate(self.data_loaders['train']):
            # measure data loading time
            data_time = time.time() - end

            loss, acc1, acc5, batch_size = self.model_forward_backward(data)

            batch_time = time.time() - end
            self.meters_update(batch_time=batch_time,
                               data_time=data_time,
                               losses=loss.item(),
                               top1=acc1[0],
                               top5=acc5[0],
                               batch_size=batch_size)

            # measure elapsed time
            end = time.time()

            if i % args.log_freq == 0:
                progress.display(i)

    def model_forward_backward(self, batch):
        images, target, _ = self.parse_batch_train(batch)
        output = self.model_inference(images)
        loss = self.criterion(output, target)
        self.optimizer_step(loss)
        acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
        return loss, acc1, acc5, images.size(0)

    def parse_batch_train(self, batch):
        images, target, domain = batch
        args = self.args
        if args.gpu is not None:
            images = images.cuda(device=args.gpu, non_blocking=True)
            target = target.cuda(device=args.gpu, non_blocking=True)
            domain = domain.cuda(device=args.gpu, non_blocking=True)
        return images, target, domain


if __name__ == '__main__':
    TrainerDG().run()
