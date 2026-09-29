import argparse
import math
import os
import pprint
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

from loader.dataloader import make_auto_loader
from model import CFM, DiT
from model.model_utils import get_tokenizer
from utils.logger import get_logger


def make_dataloader(opt):
    dataloader_setting = {
        **opt["datasets"]["dataloader_setting"],
        "mel_spec_kwargs": opt["model"]["mel_spec"],
        "tokenizer": opt["model"]["tokenizer"],
    }
    train_sampler, train_loader = make_auto_loader(
        **opt["datasets"]["train"],
        **dataloader_setting,
    )
    val_sampler, val_loader = make_auto_loader(
        **opt["datasets"]["val"],
        **dataloader_setting,
    )
    return train_sampler, train_loader, val_sampler, val_loader


def save_checkpoint(
    checkpoint_dir,
    nnet,
    optimizer,
    scheduler,
    epoch,
    best_loss,
    step=None,
    save_period=-1,
    best=True,
    logger=None,
):
    """
    Save checkpoint (epoch, model, optimizer, best_loss)
    """
    cpt = {
        "epoch": epoch,
        "model_state_dict": nnet.module.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),  
        "best_loss": best_loss,
    }
    cpt_name = "{0}.pt.tar".format("best" if best else "last")
    torch.save(cpt, checkpoint_dir / cpt_name)
    if logger is not None:
        logger.info(f"save checkpoint {cpt_name}")
    if step is not None:
        torch.save(cpt, checkpoint_dir / f"{epoch}_{step}.pt.tar")
    elif save_period > 0 and epoch % save_period == 0:
        torch.save(cpt, checkpoint_dir / f"{epoch}.pt.tar")
    
    


def load_obj(obj, device):
    """
    Offload tensor object in obj to cuda device
    """

    def cuda(obj):
        return (
            obj.to(device, non_blocking=True) if isinstance(obj, torch.Tensor) else obj
        )

    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)


def reduce_weighted_mean(value, weight):
    weight = weight.to(device=value.device, dtype=value.dtype)
    stats = torch.stack((value.detach() * weight, weight))
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return stats[0] / stats[1], stats[1]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [str(datetime.now()) + "\t"]
        entries += [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def get_learning_rate(optimizer):
    """Get learning rate"""
    return optimizer.param_groups[0]["lr"]


def train_one_epoch(
    train_loader,
    nnet,
    optimizer,
    scheduler,
    epoch,
    local_rank,
    conf,
    device,
    logger,
):
    if local_rank == 0:
        lr = get_learning_rate(optimizer)
        logger.info("set train mode, lr: {:.3e}".format(lr))
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
        logger=logger,
    )

    nnet.train()
    grad_accumulation_steps = max(1, int(conf["optim"]["grad_accumulation_steps"]))
    optimizer.zero_grad(set_to_none=True)

    end = time.time()
    for i, egs in enumerate(train_loader):

        
        egs = load_obj(egs, device)
        data_time.update(time.time() - end)
        noisy = egs["noisy_mel"].transpose(-1,-2)
        label = egs["label_mel"].transpose(-1,-2)
        text = egs["text"]
    
        loss, _, _ = nnet(
            inp=noisy,
            clean=label,
            text=text,
            mel_lengths=egs["label_mel_lengths"],
        )
        loss_weight = egs["label_mel_lengths"].sum() * label.size(-1)
        reduced_loss, reduced_weight = reduce_weighted_mean(loss, loss_weight)
        losses.update(reduced_loss.item(), reduced_weight.item())

        (loss / grad_accumulation_steps).backward()

        should_step = (i + 1) % grad_accumulation_steps == 0 or (i + 1) == len(train_loader)
        if should_step:
            clip_grad_norm_(nnet.parameters(), conf["optim"]["gradient_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        batch_time.update(time.time() - end)
        end = time.time()

        if i % conf["logger"]["print_freq"] == 0 and local_rank == 0:
            progress.display(i)
            
        

    if local_rank == 0:
        progress.display(len(train_loader))


def validate_one_epoch(
    val_loader,
    nnet,
    local_rank,
    conf,
    device,
    logger,
):
    if local_rank == 0:
        logger.info("set validate mode")
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix="Validation: ",
        logger=logger,
    )

    nnet.eval()

    with torch.no_grad():
        end = time.time()
        for i, egs in enumerate(val_loader):
            
            
            egs = load_obj(egs, device)

            noisy = egs["noisy_mel"].transpose(-1,-2)
            label = egs["label_mel"].transpose(-1,-2)
            text = egs["text"]
            
            loss, _, _ = nnet(
                inp=noisy,
                text=text,
                clean=label,
                mel_lengths=egs["label_mel_lengths"],
            )

            loss_weight = egs["label_mel_lengths"].sum() * label.size(-1)
            reduced_loss, reduced_weight = reduce_weighted_mean(loss, loss_weight)

            losses.update(reduced_loss.item(), reduced_weight.item())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % conf["logger"]["print_freq"] == 0 and local_rank == 0:
                progress.display(i)
    if local_rank == 0:
        progress.display(len(val_loader))

    return losses.avg


def main_worker(local_rank, args):
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    cudnn.benchmark = True
    




    with open(args.conf, "r") as f:
        conf = yaml.safe_load(f)

    random.seed(conf['train']['seed'])
    np.random.seed(conf['train']['seed'])
    torch.cuda.manual_seed_all(conf['train']['seed'])
    checkpoint_dir = Path(conf["train"]["checkpoint"])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    logger = get_logger(
        name=(
            (checkpoint_dir / "trainer.log").as_posix()
            if conf["logger"]["path"] is None
            else conf["logger"]["path"]
        ),
        file=local_rank == 0,
    )
    logger.disabled = local_rank != 0
    if local_rank == 0:
        logger.info("Arguments in args:\n{}".format(pprint.pformat(vars(args))))
        logger.info("Arguments in yaml:\n{}".format(pprint.pformat(conf)))
        with open(checkpoint_dir / "train.yaml", "w") as f:
            yaml.dump(conf, f)

    model_cls = DiT
    
    ## 
    tokenizer = conf['model']['tokenizer']

    ##
    tokenizer_path = conf['model']['tokenizer_path']
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)
    
    
    nnet = CFM(
        transformer=model_cls(**conf['model']['arch'], text_num_embeds=vocab_size,mel_dim=conf['model']['mel_spec']['n_mel_channels']),
        audio_drop_prob=conf['model']['audio_drop_prob'],cond_drop_prob=conf['model']['cond_drop_prob'],
        mel_spec_kwargs=conf['model']['mel_spec'],vocab_char_map=vocab_char_map
    )

    
    if local_rank == 0:
        num_params = sum([param.nelement() for param in nnet.parameters()]) / 10.0**6
        logger.info("model summary:\n{}".format(nnet))
        logger.info(f"#param: {num_params:.2f}M")
  
    start_epoch = 0
    end_epoch = conf["train"]["epoch"]

    cpt = None
    if conf["train"]["resume"]:
        if not Path(conf["train"]["resume"]).exists():
            raise FileNotFoundError(
                f"Could not find resume checkpoint: {conf['train']['resume']}")
        

        else:
            cpt = torch.load(conf["train"]["resume"], map_location="cpu", weights_only=True)
            if conf["train"]["rm_stft"]:
                for i in list(cpt["model_state_dict"]):
                    if i[: len("stft")] == "stft" or i[: len("istft")] == "istft":
                        del cpt["model_state_dict"][i]
            
            start_epoch = cpt["epoch"] + 1
            
            if local_rank == 0:
                logger.info(
                    f"resume from checkpoint {conf['train']['resume']}: epoch {start_epoch:d}"
                )
            nnet.load_state_dict(
                cpt["model_state_dict"], strict=conf["train"]["strict"]
            )

    nnet = nnet.to(device)

    nnet = DistributedDataParallel(
        nnet,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )
    optimizer = AdamW(nnet.parameters(), lr=conf['optim']['lr'])

    train_sampler, train_loader, val_sampler, val_loader = make_dataloader(conf)

    grad_accumulation_steps = max(1, int(conf['optim']['grad_accumulation_steps']))
    updates_per_epoch = math.ceil(len(train_loader) / grad_accumulation_steps)
    scheduler_epochs = end_epoch
    if conf["train"]["resume"] and conf["train"]["reset_lr"]:
        scheduler_epochs = end_epoch - start_epoch
    total_steps = updates_per_epoch * scheduler_epochs
    warmup_steps = int(conf['optim']['warm_up_step'])
    if total_steps <= warmup_steps:
        raise ValueError(
            f"warm_up_step ({warmup_steps}) must be smaller than total optimizer steps ({total_steps})"
        )
    decay_steps = total_steps - warmup_steps
    logger.info(f"warmup_steps:{warmup_steps}")
    logger.info(f"decay_steps:{decay_steps}")
    
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
    scheduler = SequentialLR(
        optimizer,schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps]
    )
    
    if conf["train"]["resume"] and not conf["train"]["reset_lr"]:
        if not Path(conf["train"]["resume"]).exists():
            raise FileNotFoundError(
                f"Could not find resume checkpoint: {conf['train']['resume']}"
            )
        cpt = torch.load(conf["train"]["resume"], map_location=device, weights_only=True)
        
        optimizer.load_state_dict(cpt["optim_state_dict"])
        scheduler.load_state_dict(cpt["scheduler_state_dict"])
    
    best_loss = 10000
    if cpt is not None and conf["train"].get("reload_best_loss", True):
        best_loss = cpt.get("best_loss", best_loss)
    no_impr = 0

    for epoch in range(start_epoch, end_epoch):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        logger.info(f"epoch:{epoch} train")
        train_one_epoch(
            train_loader,
            nnet,
            optimizer,
            scheduler,
            epoch,
            local_rank,
            conf,
            device,
            logger,
        )
        logger.info(f"epoch:{epoch} val")
        cv_loss = validate_one_epoch(
            val_loader,
            nnet,
            local_rank,
            conf,
            device,
            logger,
        )
        logger.info(f"epoch:{epoch} val done")
        if cv_loss < best_loss:
            best_loss = cv_loss
            no_impr = 0
            
            if local_rank == 0:
                save_checkpoint(
                    checkpoint_dir,
                    nnet,
                    optimizer,
                    scheduler,
                    epoch,
                    best_loss,
                    save_period=-1,
                    best=True,
                    logger=logger,
                )
            logger.info(f"epoch:{epoch} save best")
        else:
            no_impr += 1
            if local_rank == 0:
                logger.info(f"| no impr, best = {best_loss:.4f}")
                

        logger.info(f"epoch:{epoch} save")
        if local_rank == 0:
            save_checkpoint(
                checkpoint_dir,
                nnet,
                optimizer,
                scheduler,
                epoch,
                best_loss,
                save_period=conf["train"]["save_period"],
                best=False,
                logger=logger,
            )

        if no_impr >= conf["train"]["early_stop"]:
            if local_rank == 0:
                logger.info(f"stop training cause no impr for {no_impr:d} epochs")
            break
        logger.info(f"epoch:{epoch} done")

    dist.destroy_process_group()


def run(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    main_worker(local_rank, args)


if __name__ == "__main__":
    # os.environ["NCCL_SOCKET_IFNAME"] = "en,eth,em,bond"

    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument(
        "-conf", type=str, required=True, help="Yaml configuration file for training"
    )
    args = parser.parse_args()
    run(args)
