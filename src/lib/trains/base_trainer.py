from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import math
import random
import multiprocessing

from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter


class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        # TODO: �Ż�batch��FPN�ṹ
        outputs = self.model(batch[0]['input'])
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats


class BaseTrainer(object):
    def __init__(
            self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModleWithLoss(model, self.loss)

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
            if epoch < self.opt.start_wh:
                self.opt.wh_weight = 0.
            if epoch < self.opt.start_a:
                self.opt.a_weight = 0.
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            # InputSize.set_size(32 * random.randint(12, 20))
            if self.opt.warmup:
                iters = (epoch - 1) * num_iters + iter_id
                lr = self.get_lr(epoch)
                if iters <= self.opt.warmup_iters:
                    lr = self.get_warmup_lr(iters, [lr])[0]
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)
            for i in range(len(batch)):
                for k in batch[i]:
                    if k != 'meta':
                        batch[i][k] = batch[i][k].to(device=opt.device, non_blocking=True)

            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                try:
                    avg_loss_stats[l].update(
                        loss_stats[l].mean().item(), batch[0]['input'].size(0))
                except AttributeError:
                    print(l, avg_loss_stats[l])
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
            else:
                bar.next()

            if self.opt.related_hm:
                self.opt.wh_weight = 1. / math.exp(avg_loss_stats['hm_loss'].avg) + 1e-4
                self.opt.a_weight = 10. / math.exp(avg_loss_stats['hm_loss'].avg) + 1e-3
                self.opt.wh_weight = self.opt.wh_weight if self.opt.wh_weight < 10. else 10.
                self.opt.a_weight = self.opt.a_weight if self.opt.a_weight < 100. else 100.

            # if opt.debug > 0:
                # self.debug_for_polygon(batch, output, iter_id)
            # self.debug_for_loss(loss_stats, epoch*iter_id)

            if opt.test:
                self.save_result(output, batch, results)
            del output, loss, loss_stats

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def debug_for_polygon(self, batch, output, iter_id):
        raise NotImplementedError

    def debug_for_loss(self, loss_stats, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)

    def get_lr(self, progress, gamma=0.1):

        if isinstance(self.opt.lr_step, int):
            return self.opt.lr * (gamma**(progress // self.opt.lr_step))

        exp = len(self.opt.lr_step)
        for i, s in enumerate(self.opt.lr_step):
            if progress < s:
                exp = i
                break
        return self.opt.lr * gamma**exp

    def get_warmup_lr(self, cur_iters, regular_lr):
        warmup_ratio = 1.0 / 3
        warmup_method = 'linear'
        if warmup_method == 'constant':
            warmup_lr = [_lr * warmup_ratio for _lr in regular_lr]
        elif warmup_method == 'linear':
            k = (1 - cur_iters / self.opt.warmup_iters) * (1 - warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
        elif warmup_method == 'exp':
            k = warmup_ratio**(1 - cur_iters / self.opt.warmup_iters)
            warmup_lr = [_lr * k for _lr in regular_lr]
        return warmup_lr
