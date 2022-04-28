import os
import time
import shutil
import random
import inspect
import datetime
import argparse
import warnings
from typing import Dict
from tabulate import tabulate
from collections import OrderedDict
from logger_tt import setup_logging, logger
import torch
from torch import nn
from torch import optim
from torch import distributed
from torch.backends import cudnn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.utils.collect_env import get_pretty_env_info
from ddg.utils import DictAction
from ddg.utils import AverageMeter
from ddg.utils import ProgressMeter
from ddg.utils import MODELS_REGISTRY
from ddg.utils import TRANSFORMS_REGISTRY


__all__ = ['Trainer']


def prepare_parameters(task, trainer):
    parser = argparse.ArgumentParser(description=f'{task} trainer named {trainer} provided '
                                                 f'by DDG')
    parser.add_argument('--root', metavar='DIR', default='./data',
                        help='path to dataset')
    parser.add_argument('--dataset', metavar='dataset', help='dataset')
    parser.add_argument('--transform', type=str, help='data transform')
    parser.add_argument('--input-size', type=int, default=224, help='input size')
    parser.add_argument('--source-domains', type=str, nargs='+',
                        help=f'source domains for {task}')
    parser.add_argument('--target-domains', type=str, nargs='+',
                        help=f'target domains for {task}')
    parser.add_argument('--models', type=str, nargs='+', metavar='KEY=VALUE',
                        default=OrderedDict({'model': 'resnet18'}), action=DictAction,
                        help='get a number of key-value pairs (model_name=model_arch), models will be '
                             'initialized in input order, pay attention to their dependencies. '
                             'Note: All arguments required by models can be passed into args by overriding '
                             'self.add_extra_args() method, and the same is true for parameters exchange '
                             'between models')
    parser.add_argument('--workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 6)')
    parser.add_argument('--sampler', default=None, type=str, help='sampler for train split')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 64), this is the total batch size of all GPUs '
                             'when using Distributed Data Parallel')
    parser.add_argument('--optimizer', type=str, nargs='+', metavar='KEY=VALUE',
                        default=OrderedDict({'name': 'SGD',
                                             'lr': 0.1,
                                             'momentum': 0.9,
                                             'weight_decay': 1e-4}),
                        action=DictAction,
                        help='optimizer, support almost ALL optimizer provided by PyTorch now, this option get a '
                             'number of key-value pairs, you must include the key named "name" to specify which '
                             'optimizer to use, all other key-value pairs will be passed to the optimizer if '
                             'required by it, otherwise discarded. Note: value must be numbers.')
    parser.add_argument('--scheduler', type=str, nargs='+', metavar='KEY=VALUE',
                        default=OrderedDict({'name': 'StepLR',
                                             'step_size': 30,
                                             'gamma': 0.1}),
                        action=DictAction,
                        help='scheduler, support almost ALL optimizer provided by PyTorch now, this option get '
                             'a number of key-value pairs, you must include the key named "name" to specify which'
                             ' scheduler to use, all other key-value pairs will be passed to the scheduler if '
                             'required by it, otherwise discarded. Note: value must be numbers.')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='label smoothing for CrossEntropyLoss')
    parser.add_argument('--pretrained', default=False, action='store_true',
                        help='load pre-trained model, Note: this is different from "resume", which determines whether '
                             'to load pretrained weights from Internet when the model is initialized, while "resume" '
                             'is used to load previously saved checkpoints')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none), training process will start from this '
                             'checkpoint, can work with "evaluate" to perform evaluation on previously saved '
                             'checkpoints')
    parser.add_argument('--checkpoint-freq', default=10, type=int,
                        metavar='N', help='checkpoint frequency (default: 10)')
    parser.add_argument('--log-freq', default=10, type=int,
                        metavar='N', help='log frequency (default: 10)')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--val-split', type=str, default='test', choices={'val', 'test'},
                        help='Use which split to do evaluate on current model after every epoch.')
    parser.add_argument('--final-model', type=str, default=None, choices={'last_step', 'best_val'},
                        help='Use which model to do evaluate on test split after training, default no test after train')
    parser.add_argument('--distributed', action='store_true',
                        help='distributed train by using pytorch Distributed Data Parallel')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        choices={'nccl', 'gloo', 'mpi'},
                        help='distributed backend for Distributed Data Parallel, Note: early released NVIDIA '
                             'GPU do not support "nccl", use "gloo" instead')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use, only works when no --distributed is set and '
                             'torch.cuda.is_available() is True')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--save-dir', type=str, default='debug_output')
    return parser


class Trainer:
    """Base trainer provided by DDG.
    """

    TASK = "Base"

    def __init__(self):
        super().__init__()

        self.parser = prepare_parameters(task=self.TASK, trainer=self.__class__.__name__)
        self.add_extra_args()
        args = self.parser.parse_args()
        log_file = 'log.txt'
        if args.distributed:
            args.local_rank = int(os.environ["LOCAL_RANK"])
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.dist_url = f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}'
            args.gpu = args.local_rank
            args.batch_size = int(args.batch_size / args.world_size)
            log_file = 'rank_' + str(args.rank) + '_' + log_file
        args.log_file = log_file
        self.args = args
        setup_logging(log_path=os.path.join(args.save_dir, args.log_file))

        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')

        if not torch.cuda.is_available():
            logger.warning('cuda is not available, using CPU, this will be slow.')
            self.args.gpu = None
            if args.distributed:
                raise NotImplementedError('cuda is not available, but distributed is set, we not support cpu '
                                          'distributed training.')
        else:
            if torch.cuda.device_count() < (args.gpu + 1):
                raise ValueError(f'No device {args.gpu} configured in this machine!')

        cudnn.benchmark = True

        self.init_process_group()

        self.transform = None
        self.build_transform()

        self.datasets: OrderedDict = OrderedDict()
        self.build_dataset()

        self.samplers: OrderedDict = OrderedDict()
        self.data_loaders: OrderedDict = OrderedDict()
        self.build_data_loader()

        self.models: OrderedDict = OrderedDict()
        self.optimizer = None
        self.scheduler = None
        self.build_model()
        self.model_cuda()
        self.build_optimizer()
        self.build_scheduler()

        self.criterion = None
        self.build_criterion()

        self.writer = None
        self.start_epoch: int = 0
        self.epoch: int = 0
        self.max_epoch = self.args.epochs
        self.resume_train_state()

        self.meters: Dict = {}
        self.progress: Dict = {}
        self.best_acc1 = 0
        self.log_args_and_env()

    def add_extra_args(self):
        """ One can add additional parameters by overriding this method in the inherited trainers.

        Examples::
        def add_extra_args(self):
            super(DDG, self).add_extra_args()
            parser = self.parser
            parser.add_argument('--test-arg', type=str, default='arg for test')
            self.parser = parser
        """

    def log_args_and_env(self):
        args_dict = self.args.__dict__
        args_table = []
        for key in args_dict.keys():
            args_table.append((key, args_dict[key]))
        table_headers = ["Args", "Value"]
        table = tabulate(
            args_table, headers=table_headers, tablefmt="fancy_grid"
        )
        logger.info("\nAll args of {}:\n".format(self.__class__.__name__) + table)

        env_info = '\n' + 'env'.center(60, '*') + '\n' + get_pretty_env_info() + '\n'
        env_info += ''.center(60, '*')
        logger.info(env_info)

    def init_process_group(self):
        if self.args.distributed:
            distributed.init_process_group(backend=self.args.dist_backend,
                                           init_method=self.args.dist_url,
                                           world_size=self.args.world_size,
                                           rank=self.args.rank)

    def build_transform(self):
        self.transform = TRANSFORMS_REGISTRY.get(self.args.transform)(args=self.args)

    def build_dataset(self):
        raise NotImplementedError

    def build_data_loader(self):
        raise NotImplementedError

    def build_model(self):
        """
        args.num_classes already set when loading data, you can use it for free. All other parameters that
        need to interact between models need to be set in args by yourself. For example, if model_2 use
        model_1`s output as input, we can set args.model_1_output_dim=N after init model_1, then when init
        model_2, we can let input_size=args.model_1_output_dim.

        """
        args = self.args
        from_name = None
        for model in args.models:
            args.__dict__[model] = {}
            self.models[model] = MODELS_REGISTRY.get(args.models[model])(model, args, from_name)
            from_name = model

    def model_cuda(self):
        args = self.args
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            for model in self.models:
                self.models[model].cuda(args.gpu)
        if args.distributed:
            for model in self.models:
                self.models[model] = DistributedDataParallel(self.models[model],
                                                             device_ids=[args.gpu],
                                                             output_device=args.gpu)

    def build_optimizer(self):
        args = self.args
        parameters = []
        for model in self.models:
            parameters.append({'params': self.models[model].parameters()})
        optimizer = args.optimizer['name']
        params = {}
        if not (hasattr(optim, optimizer) and callable(getattr(optim, optimizer))):
            raise NotImplementedError(f'optimizer named {optimizer} not implemented yet!')
        accepted_params = inspect.getfullargspec(getattr(optim, optimizer)).args[2:]
        for key in args.optimizer:
            if key in accepted_params:
                params[key] = args.optimizer[key]
        self.optimizer = getattr(optim, optimizer)(parameters, **params)

    def build_scheduler(self):
        args = self.args
        scheduler = args.scheduler['name']
        params = {}
        if not (hasattr(lr_scheduler, scheduler) and callable(getattr(lr_scheduler, scheduler))):
            raise NotImplementedError(f'lr scheduler named {scheduler} not implemented yet!')
        accepted_params = inspect.getfullargspec(getattr(lr_scheduler, scheduler)).args[2:]
        for key in args.scheduler:
            if key in accepted_params:
                params[key] = args.scheduler[key]
        if 'T_max' in accepted_params:
            params['T_max'] = args.epochs
        self.scheduler = getattr(lr_scheduler, scheduler)(self.optimizer, **params)

    def build_criterion(self):
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing).cuda(self.args.gpu)

    def run(self):
        args = self.args
        self.writer_init()
        self.meters['batch_time'] = AverageMeter('Time', ':6.3f')
        self.meters['data_time'] = AverageMeter('Data', ':6.3f')
        self.meters['losses'] = AverageMeter('Loss', ':.4e')
        self.meters['top1'] = AverageMeter('Acc@1', ':6.2f')
        self.meters['top5'] = AverageMeter('Acc@5', ':6.2f')
        time_start = datetime.datetime.now()
        if args.evaluate:
            logger.info('Evaluate started...')
            self.evaluate()
        else:
            logger.info('Train started...')
            self.train()
        self.writer_close()
        elapsed = datetime.datetime.now() - time_start
        logger.info(f"Elapsed: {elapsed}")

    def train(self):
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def before_train(self):
        pass

    def after_train(self):
        logger.info('Finished training...')
        args = self.args
        if args.final_model is not None:
            if args.final_model == 'best_val':
                self.resume_train_state(resume_path=os.path.join(args.save_dir, 'checkpoint_best.pth.tar'))
            self.evaluate()
        logger.info('Finished...')

    def before_epoch(self):
        if self.args.distributed:
            for sampler in self.samplers:
                if hasattr(self.samplers[sampler], 'set_epoch'):
                    self.samplers[sampler].set_epoch(self.epoch)
        self.models_train()
        self.meters_reset()
        self.setup_progress()
        self.progress['train'] = ProgressMeter(
            len(self.data_loaders['train']),
            [self.meters[key] for key in self.meters],
            prefix="Epoch: [{}]".format(self.epoch))

    def setup_progress(self):
        self.progress['train'] = ProgressMeter(
            len(self.data_loaders['train']),
            [self.meters[key] for key in self.meters],
            prefix="Epoch: [{}]".format(self.epoch))

    def after_epoch(self):
        args = self.args
        train_losses = self.meters['losses'].avg
        train_acc1 = self.meters['top1'].avg
        train_acc5 = self.meters['top5'].avg
        self.evaluate(split=args.val_split)
        losses = self.meters['losses'].avg
        acc1 = self.meters['top1'].avg
        acc5 = self.meters['top5'].avg
        is_best = acc1 > self.best_acc1
        self.best_acc1 = max(acc1, self.best_acc1)
        if not args.distributed or (args.distributed and args.rank == 0):
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalars('learning_rate', {'lr': lr}, self.epoch + 1)
            self.writer.add_scalars('losses', {'train': train_losses, 'validation': losses}, self.epoch + 1)
            self.writer.add_scalars('acc@1', {'train': train_acc1, 'validation': acc1}, self.epoch + 1)
            self.writer.add_scalars('acc@5', {'train': train_acc5, 'validation': acc5}, self.epoch + 1)
            self.save_checkpoint(is_best=is_best)

        self.scheduler_step()

    def run_epoch(self):
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, split='test'):

        def run_evaluate(loader):
            args = self.args
            progress = self.progress['evaluate']
            end = time.time()
            for i, data in enumerate(loader):
                images, target = self.parse_batch_test(data)

                # measure data loading time
                data_time = time.time() - end

                # compute train output
                output = self.model_inference(images)
                loss = self.criterion(output, target)
                # measure train accuracy and record loss
                acc1, acc5 = self.accuracy(output, target, topk=(1, 5))

                batch_time = time.time() - end

                self.meters_update(batch_time=batch_time,
                                   data_time=data_time,
                                   losses=loss.item(),
                                   top1=acc1[0],
                                   top5=acc5[0],
                                   batch_size=images.size(0))

                # measure elapsed time
                end = time.time()

                if i % args.log_freq == 0:
                    progress.display(i)

            progress.display_summary()

        self.models_eval()
        self.meters_reset()
        data_loader = self.data_loaders[split]
        logger.info(f'Do evaluation on {split} set')
        self.progress['evaluate'] = ProgressMeter(
            len(data_loader),
            [self.meters[key] for key in self.meters],
            prefix="Evaluate: [{}]".format(self.epoch))
        run_evaluate(data_loader)

    def model_forward_backward(self, batch):
        raise NotImplementedError

    def model_inference(self, images):
        # For the case where the data flows linearly through all the models,
        # in other cases, you need to rewrite the function yourself.
        out = images
        for model in self.args.models:
            out = self.models[model](out)
        return out

    def optimizer_step(self, loss):
        # compute gradient and do optimizer step
        if not torch.isfinite(loss).all():
            raise FloatingPointError('Loss is infinite or NaN!')
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def scheduler_step(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def models_train(self):
        for model in self.models:
            self.models[model].train()

    def models_eval(self):
        for model in self.models:
            self.models[model].eval()

    def writer_init(self):
        if not self.args.distributed or (self.args.distributed and self.args.rank == 0):
            self.writer_close()
            self.writer = SummaryWriter(log_dir=os.path.join(self.args.save_dir, 'tensorboard'), purge_step=self.epoch)

    def writer_close(self):
        if self.writer is not None:
            self.writer.close()

    def meters_reset(self):
        self.meters['batch_time'].reset()
        self.meters['data_time'].reset()
        self.meters['losses'].reset()
        self.meters['top1'].reset()
        self.meters['top5'].reset()

    def meters_update(self, batch_time, data_time, losses, top1, top5, batch_size):
        self.meters['batch_time'].update(data_time)
        self.meters['data_time'].update(batch_time)
        self.meters['losses'].update(losses, batch_size)
        self.meters['top1'].update(top1, batch_size)
        self.meters['top5'].update(top5, batch_size)

    def resume_train_state(self, resume_path=None):
        args = self.args
        if resume_path is None:
            resume_path = args.resume
        if resume_path is None:
            if os.path.isfile(os.path.join(args.save_dir, 'checkpoint.pth.tar')):
                resume_path = os.path.join(args.save_dir, 'checkpoint.pth.tar')
        if resume_path is not None and os.path.isfile(resume_path):
            logger.info(f'=> loading checkpoint "{resume_path}"')
            if args.gpu is None:
                checkpoint = torch.load(resume_path, map_location='cpu')
            else:
                # Map model to be loaded to specified single gpu.
                checkpoint = torch.load(resume_path, map_location=f'cuda:{args.gpu}')
            self.start_epoch = checkpoint['epoch']
            self.best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                self.best_acc1 = self.best_acc1.to(args.gpu)
            for model in self.models:
                self.models[model].load_state_dict(checkpoint[model])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info(f'=> loaded checkpoint "{resume_path}" (epoch {checkpoint["epoch"]})')
        elif args.resume is not None:
            logger.info(f'=> no checkpoint found at "{resume_path}"')

    def save_checkpoint(self, is_best, filename='checkpoint.pth.tar'):
        args = self.args
        if not args.distributed or (args.distributed and args.rank == 0):
            state = {
                'epoch': self.epoch + 1,
                'best_acc1': self.best_acc1,
                'optimizer': self.optimizer.state_dict()
            }
            if self.scheduler is not None:
                state['scheduler'] = self.scheduler.state_dict()
            for model in self.models:
                state[model] = self.models[model].state_dict()
            torch.save(state, os.path.join(args.save_dir, filename))
            if is_best:
                shutil.copyfile(os.path.join(args.save_dir, filename),
                                os.path.join(args.save_dir, 'checkpoint_best.pth.tar'))
            if (self.epoch + 1) % args.checkpoint_freq == 0:
                shutil.copyfile(os.path.join(args.save_dir, filename),
                                os.path.join(args.save_dir, f'checkpoint_{self.epoch + 1}.pth.tar'))

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def parse_batch_train(self, batch):
        raise NotImplementedError

    def parse_batch_test(self, batch):
        images, target, _ = batch
        args = self.args
        if args.gpu is not None:
            images = images.cuda(device=args.gpu, non_blocking=True)
            target = target.cuda(device=args.gpu, non_blocking=True)
        return images, target
