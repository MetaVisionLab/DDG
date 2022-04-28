import torch
from ddg.trainer import TrainerDG

__all__ = ['DomainMix']


class DomainMix(TrainerDG):
    """DomainMix trainer provided by DDG.
    """

    def __init__(self):
        super(DomainMix, self).__init__()
        self.mix_type = self.args.domain_mix_type
        self.alpha = self.args.domain_mix_alpha
        self.beta = self.args.domain_mix_beta
        self.dist_beta = torch.distributions.Beta(self.alpha, self.beta)

    def add_extra_args(self):
        super(DomainMix, self).add_extra_args()
        parse = self.parser
        parse.add_argument('--domain-mix-type', type=str, default='crossdomain', choices={'random', 'crossdomain'},
                           help='Mix type for DomainMix.')
        parse.add_argument('--domain-mix-alpha', type=float, default=1.0, help='alpha for DomainMix.')
        parse.add_argument('--domain-mix-beta', type=float, default=1.0, help='beta for DomainMix.')

    def model_forward_backward(self, batch):
        images, target, label_a, label_b, lam = self.parse_batch_train(batch)
        output = self.model_inference(images)
        loss = lam * self.criterion(output, label_a) + (1 - lam) * self.criterion(output, label_b)
        self.optimizer_step(loss)
        acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
        return loss, acc1, acc5, images.size(0)

    def parse_batch_train(self, batch):
        images, target, domain = super(DomainMix, self).parse_batch_train(batch)
        images, target_a, target_b, lam = self.domain_mix(images, target, domain)
        return images, target, target_a, target_b, lam

    def domain_mix(self, x, target, domain):
        lam = (self.dist_beta.rsample((1,)) if self.alpha > 0 else torch.tensor(1)).to(x.device)

        # random shuffle
        perm = torch.randperm(x.size(0), dtype=torch.int64, device=x.device)
        if self.mix_type == 'crossdomain':
            domain_list = torch.unique(domain)
            if len(domain_list) > 1:
                for idx in domain_list:
                    cnt_a = torch.sum(domain == idx)
                    idx_b = (domain != idx).nonzero().squeeze(-1)
                    cnt_b = idx_b.shape[0]
                    perm_b = torch.ones(cnt_b).multinomial(num_samples=cnt_a, replacement=bool(cnt_a > cnt_b))
                    perm[domain == idx] = idx_b[perm_b]
        elif self.mix_type != 'random':
            raise NotImplementedError(f"Chooses {'random', 'crossdomain'}, but got {self.mix_type}.")
        mixed_x = lam * x + (1 - lam) * x[perm, :]
        target_a, target_b = target, target[perm]
        return mixed_x, target_a, target_b, lam


if __name__ == '__main__':
    DomainMix().run()
