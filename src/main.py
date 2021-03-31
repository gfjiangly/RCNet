from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    # torch.backends.cudnn.benchmark = False
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt.fpn)

    # from thop import profile
    # input = torch.randn(1, 3, 512, 512)
    # flops, params = profile(model, (1, 3, 512, 512), device='cuda')
    # from thop.utils import clever_format
    # flops = clever_format(flops, "%.3f")
    # params = clever_format(params, "%.3f")

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print('Setting up data...')
    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    if opt.test:
        _, preds = trainer.val(0, val_loader)
        val_loader.dataset.run_eval(preds, opt.save_dir)
        return

    train_dataset = Dataset(opt, 'trainval')
    if opt.weighted:
        import _pickle as pickle
        weights = pickle.load(open('/CenterNet/data/log_weights.pkl', 'rb'))
        # seq_sample = torch.utils.data.sampler.SequentialSampler(train_dataset)
        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset,
        #     shuffle=False,
        #     # num_workers=opt.num_workers,
        #     num_workers=0,
        #     pin_memory=False,
        #     # drop_last=True,
        #     sampler=seq_sample
        # )
        # for inds in train_loader:
        #     data = train_dataset[inds[0]['inds']]
        #     print(data)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            # shuffle=False,
            num_workers=opt.num_workers,
            # num_workers=0,
            pin_memory=True,
            # drop_last=True,
            sampler=sampler
        )
        # for i in range(100):
        #     print(torch.multinomial(torch.tensor(weights, dtype=torch.double), 16, replacement=True))
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True,
            # collate_fn=collate_fn
        )

    print('Starting training...')
    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                           epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
