import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from models import Uni_Sign
import utils as utils
from datasets import S2T_Dataset, S2T_Dataset_YTASL
#S2T_Dataset_YTASL_h5
import os
import time
import argparse, json, datetime
from pathlib import Path
import math
import sys
import random
from timm.optim import create_optimizer
from models import get_requires_grad_dict
from SLRT_metrics import translation_performance, islr_performance, wer_list
from transformers import get_scheduler
from config import *
import wandb
import numpy as np


def _resolve_ds_checkpoint_load_args(checkpoint_path):
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    if not ckpt_path.is_dir():
        raise ValueError(
            f"DeepSpeed checkpoint path must be a directory (got file): {checkpoint_path}"
        )
    if ckpt_path.name.startswith("checkpoint_"):
        return str(ckpt_path.parent), ckpt_path.name, ckpt_path
    return str(ckpt_path), None, _resolve_latest_tag_dir(ckpt_path)


def _resolve_latest_tag_dir(output_dir_path):
    latest_path = output_dir_path / "latest"
    if not latest_path.exists():
        return None
    tag = latest_path.read_text().strip()
    if not tag:
        return None
    return output_dir_path / tag


def _print_ds_checkpoint_file_hints(tag_dir):
    if tag_dir is None:
        print("DeepSpeed checkpoint hint: could not resolve specific tag directory (no readable 'latest').")
        return
    if not tag_dir.exists() or not tag_dir.is_dir():
        print(f"DeepSpeed checkpoint hint: expected tag directory does not exist: {tag_dir}")
        return

    model_state = tag_dir / "mp_rank_00_model_states.pt"
    optim_states = list(tag_dir.glob("*_optim_states.pt"))
    if not model_state.exists():
        print(f"DeepSpeed checkpoint hint: missing expected file: {model_state}")
    if len(optim_states) == 0:
        print(f"DeepSpeed checkpoint hint: missing expected optimizer shard file in {tag_dir} (pattern '*_optim_states.pt').")


def load_weights_from_torch_checkpoint(model_without_ddp, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location='cpu')['model']
    ret = model_without_ddp.load_state_dict(state_dict, strict=True)
    print('Missing keys: \n', '\n'.join(ret.missing_keys))
    print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))


def get_system_job_id():
    env = os.environ
    slurm_job_id = env.get("SLURM_JOB_ID") or env.get("SLURM_JOBID")
    if slurm_job_id:
        return f"slurm{slurm_job_id}"
    pbs_job_id = env.get("PBS_JOBID") or env.get("PBS_JOB_ID")
    if pbs_job_id:
        return f"pbs{pbs_job_id}"
    return None

def main(args):
    utils.init_distributed_mode_ds(args)

    print(args)
    utils.set_seed(args.seed)

    if args.finetune and args.resume:
        raise ValueError("Use only one of --finetune (weights-only) or --resume (full state).")

    wandb_run_id = None
    wandb_run_name = None

    print(f"Creating dataset:")

    if args.dataset == "YTASL":
        train_data = S2T_Dataset_YTASL(path=train_label_paths[args.dataset],
                                 args=args, phase='train')
    else:
        train_data = S2T_Dataset(path=train_label_paths[args.dataset],
                                 args=args, phase='train')
    print(train_data)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  collate_fn=train_data.collate_fn,
                                  sampler=train_sampler,
                                  pin_memory=args.pin_mem,
                                  drop_last=True)

    # metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # header = 'Epoch: [{}/{}]'.format(1, args.epochs)
    # print_freq = 10
    # model = Uni_Sign(
    #     args=args
    # )
    # # model.cuda()
    # model.train()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         param.data = param.data.to(torch.float32)
    # src_input, tgt_input = next(iter(train_dataloader))
    # # src_input, tgt_input = src_input.cuda(), tgt_input.cuda()
    # out = model(src_input, tgt_input)
    #
    # for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)):
    #     print(step)
    #
    #     if args.task == "CSLR":
    #         tgt_input['gt_sentence'] = tgt_input['gt_gloss']
    #
    #     for key in src_input.keys():
    #         if isinstance(src_input[key], torch.Tensor):
    #             # src_input[key] = src_input[key].cuda()
    #             src_input[key] = src_input[key]
    #             # src_input[key] = src_input[key].to(torch.bfloat16).cuda()
    #
    #     stack_out = model(src_input, tgt_input)
    #     print(stack_out)
    #     break



    if args.dataset == "YTASL":
        dev_data = S2T_Dataset_YTASL(path=dev_label_paths[args.dataset],
                                 args=args, phase='dev')
    else:
        dev_data = S2T_Dataset(path=dev_label_paths[args.dataset],
                                 args=args, phase='dev')
    # dev_data = S2T_Dataset(path=dev_label_paths[args.dataset],
    #                        args=args, phase='dev')
    print(dev_data)
    if args.distributed:
        dev_sampler = torch.utils.data.distributed.DistributedSampler(
            dev_data,
            shuffle=False,
            drop_last=False
        )
    else:
        dev_sampler = torch.utils.data.SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=dev_data.collate_fn,
                                sampler=dev_sampler,
                                pin_memory=args.pin_mem)

    if args.dataset == "YTASL":
        test_data = S2T_Dataset_YTASL(path=test_label_paths[args.dataset],
                                 args=args, phase='test')
    else:
        test_data = S2T_Dataset(path=test_label_paths[args.dataset],
                                 args=args, phase='test')
    # test_data = S2T_Dataset(path=test_label_paths[args.dataset],
    #                         args=args, phase='test')
    print(test_data)
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_data,
            shuffle=False,
            drop_last=False
        )
    else:
        test_sampler = torch.utils.data.SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=test_data.collate_fn,
                                 sampler=test_sampler,
                                 pin_memory=args.pin_mem)

    print(f"Creating model:")
    model = Uni_Sign(
        args=args
    )
    model.cuda()
    model.train()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    finetune_from_ds_dir = False
    if args.finetune != '':
        print('***********************************')
        print('Load Checkpoint...')
        print('Weights-only finetune (optimizer/scheduler will reset)')
        print('***********************************')
        finetune_path = Path(args.finetune)
        if not finetune_path.exists():
            raise FileNotFoundError(f"Finetune checkpoint not found: {args.finetune}")
        if finetune_path.is_file():
            load_weights_from_torch_checkpoint(model, args.finetune)
        elif finetune_path.is_dir():
            finetune_from_ds_dir = True
        else:
            raise ValueError(f"Unsupported --finetune path: {args.finetune}")

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')

    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_epochs * len(train_dataloader) / args.gradient_accumulation_steps),
        num_training_steps=int(args.epochs * len(train_dataloader) / args.gradient_accumulation_steps),
    )

    for param in model.parameters(): param.data = param.data.contiguous()
    model, optimizer, lr_scheduler = utils.init_deepspeed(args, model, optimizer, lr_scheduler)
    model_without_ddp = model.module.module
    # print(model_without_ddp)
    print(optimizer)

    output_dir = Path(args.output_dir)

    start_time = time.time()
    max_accuracy = 0
    if args.task == "CSLR":
        max_accuracy = 1000
    start_epoch = 0
    client_state = {}

    if args.resume:
        print('***********************************')
        print('Resume Checkpoint (DeepSpeed)...')
        print('***********************************')
        load_dir, load_tag, tag_dir = _resolve_ds_checkpoint_load_args(args.resume)
        if utils.is_main_process():
            _print_ds_checkpoint_file_hints(tag_dir)

        load_path, client_state = model.load_checkpoint(
            load_dir,
            tag=load_tag,
            load_module_strict=True,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        if load_path is None:
            raise RuntimeError(f"Failed to load DeepSpeed checkpoint from: {args.resume}")
        print(f"Loaded DeepSpeed checkpoint: {load_path}")

        start_epoch = client_state.get('epoch', -1) + 1
        max_accuracy = client_state.get('max_accuracy', max_accuracy)
        if not wandb_run_id:
            wandb_run_id = client_state.get('wandb_run_id')
        wandb_run_name = client_state.get('wandb_run_name')
        if 'rng_state' in client_state:
            torch.set_rng_state(client_state['rng_state'])
        if 'cuda_rng_state' in client_state:
            torch.cuda.set_rng_state_all(client_state['cuda_rng_state'])
        if 'numpy_rng_state' in client_state:
            np.random.set_state(client_state['numpy_rng_state'])
        if 'random_rng_state' in client_state:
            random.setstate(client_state['random_rng_state'])
        if start_epoch >= args.epochs:
            print(f"Resume epoch {start_epoch} >= total epochs {args.epochs}; nothing to do.")
            return

    if finetune_from_ds_dir:
        load_dir, load_tag, tag_dir = _resolve_ds_checkpoint_load_args(args.finetune)
        if utils.is_main_process():
            _print_ds_checkpoint_file_hints(tag_dir)
        try:
            load_path, _ = model.load_checkpoint(
                load_dir,
                tag=load_tag,
                load_module_strict=True,
                load_optimizer_states=False,
                load_lr_scheduler_states=False,
                load_module_only=True,
            )
        except TypeError:
            load_path, _ = model.load_checkpoint(
                load_dir,
                tag=load_tag,
                load_module_strict=True,
                load_optimizer_states=False,
                load_lr_scheduler_states=False,
            )
        if load_path is None:
            raise RuntimeError(f"Failed to load DeepSpeed weights-only checkpoint from: {args.finetune}")
        print(f"Loaded DeepSpeed weights-only checkpoint: {load_path}")

    # Only main process logs to wandb
    if utils.is_main_process() and args.wandb:
        init_kwargs = {}
        if wandb_run_id:
            init_kwargs["id"] = wandb_run_id
            init_kwargs["resume"] = "allow"
        base_run_name = wandb_run_name or f"{os.path.basename(args.output_dir)}-{args.dataset}_{args.task}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        system_job_id = get_system_job_id()
        args.system_job_id = system_job_id
        if system_job_id and system_job_id not in base_run_name:
            base_run_name = f"{base_run_name}-{system_job_id}"
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "default_project"),
            entity=os.environ.get("WANDB_ENTITY", None),
            config=vars(args),
            name=base_run_name,
            **init_kwargs
        )

    if args.eval:
        # Run eval on all ranks to keep DeepSpeed/NCCL collectives aligned.
        if args.task != "ISLR":
            if utils.is_main_process():
                print("📄 dev result")
            evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev')
        if utils.is_main_process():
            print("📄 test result")
        evaluate(args, test_dataloader, model, model_without_ddp, phase='test')

        return
    print(f"Start training for {args.epochs} epochs")

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(args, model, train_dataloader, optimizer, epoch)

        if args.output_dir:
            ds_client_state = {
                'epoch': epoch,
                'max_accuracy': max_accuracy,
                'wandb_run_id': wandb.run.id if args.wandb and utils.is_main_process() and wandb.run else None,
                'wandb_run_name': wandb.run.name if args.wandb and utils.is_main_process() and wandb.run else None,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'numpy_rng_state': np.random.get_state(),
                'random_rng_state': random.getstate(),
                'global_step': (epoch + 1) * len(train_dataloader),
            }
            model.save_checkpoint(str(output_dir), tag=f'checkpoint_{epoch}', client_state=ds_client_state)
            if args.distributed and torch.distributed.is_initialized():
                # Keep all ranks aligned before rank-0-only evaluation/logging.
                torch.distributed.barrier()

        # Evaluate on all ranks so DeepSpeed collective ops remain matched.
        dev_stats = evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev')
        # evaluate(args, test_dataloader, model, model_without_ddp, phase='test')

        if utils.is_main_process():
            if args.task == "SLT":
                if max_accuracy < dev_stats["bleu4"]:
                    max_accuracy = dev_stats["bleu4"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': get_requires_grad_dict(model_without_ddp),
                            }, checkpoint_path)

                print(f"BLEU-4 of the network on the {len(dev_dataloader)} dev videos: {dev_stats['bleu4']:.2f}")
                print(f'Max BLEU-4: {max_accuracy:.2f}%')

            elif args.task == "ISLR":
                if max_accuracy < dev_stats["top1_acc_pi"]:
                    max_accuracy = dev_stats["top1_acc_pi"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': get_requires_grad_dict(model_without_ddp),
                            }, checkpoint_path)

                print(
                    f"PI accuracy of the network on the {len(dev_dataloader)} dev videos: {dev_stats['top1_acc_pi']:.2f}")
                print(f'Max PI accuracy: {max_accuracy:.2f}%')

            elif args.task == "CSLR":
                if max_accuracy > dev_stats["wer"]:
                    max_accuracy = dev_stats["wer"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': get_requires_grad_dict(model_without_ddp),
                            }, checkpoint_path)

                print(f"WER of the network on the {len(dev_dataloader)} dev videos: {dev_stats['wer']:.2f}")
                print(f'Min WER: {max_accuracy:.2f}%')

            log_stats = {**{f'train/{k}': v for k, v in train_stats.items()},
                         **{f'dev/{k}': v for k, v in dev_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            epoch_elapsed = time.time() - epoch_start_time
            log_stats['train/epoch_elapsed_sec'] = epoch_elapsed
            if args.wandb:
                wandb.log(log_stats, step=(epoch +1 ) * len(train_dataloader))

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.distributed and torch.distributed.is_initialized():
            # Non-zero ranks skip eval; wait for rank 0 to finish before next epoch collectives.
            torch.distributed.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if utils.is_main_process() and args.wandb:
        wandb.finish()


def train_one_epoch(args, model, data_loader, optimizer, epoch):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10
    optimizer.zero_grad()

    target_dtype = None
    if model.bfloat16_enabled():
        target_dtype = torch.bfloat16

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        start_time = time.time()
        if target_dtype != None:
            for key in src_input.keys():
                if isinstance(src_input[key], torch.Tensor):
                    src_input[key] = src_input[key].to(target_dtype).cuda()

        if args.task == "CSLR":
            tgt_input['gt_sentence'] = tgt_input['gt_gloss']
        stack_out = model(src_input, tgt_input)

        total_loss = stack_out['loss']
        model.backward(total_loss)
        model.step()

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        global_step = epoch * len(data_loader) + step
        if utils.is_main_process() and args.wandb and global_step % args.log_step == 0:
            elapsed_time = time.time() - start_time
            log_dict = {
                f"train/{name}": meter.global_avg
                for name, meter in metric_logger.meters.items()
            }
            log_dict['train/loss_raw'] = loss_value
            log_dict['train/iter_time'] = elapsed_time
            wandb.log(log_dict, step=global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(args, data_loader, model, model_without_ddp, phase):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    target_dtype = None
    if model.bfloat16_enabled():
        target_dtype = torch.bfloat16

    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []

        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            if target_dtype != None:
                for key in src_input.keys():
                    if isinstance(src_input[key], torch.Tensor):
                        src_input[key] = src_input[key].to(target_dtype).cuda()

            if args.task == "CSLR":
                tgt_input['gt_sentence'] = tgt_input['gt_gloss']
            stack_out = model(src_input, tgt_input)

            total_loss = stack_out['loss']
            metric_logger.update(loss=total_loss.item())

            output = model_without_ddp.generate(stack_out,
                                                max_new_tokens=100,
                                                num_beams=4,
                                                )

            for i in range(len(output)):
                tgt_pres.append(output[i])
                tgt_refs.append(tgt_input['gt_sentence'][i])

    tokenizer = model_without_ddp.mt5_tokenizer
    padding_value = tokenizer.eos_token_id

    pad_tensor = torch.ones(150 - len(tgt_pres[0])).cuda() * padding_value
    tgt_pres[0] = torch.cat((tgt_pres[0], pad_tensor.long()), dim=0)

    tgt_pres = pad_sequence(tgt_pres, batch_first=True, padding_value=padding_value)
    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)

    # fix mt5 tokenizer bug
    if args.dataset == 'CSL_Daily' and args.task == "SLT":
        tgt_pres = [' '.join(list(r.replace(" ", '').replace("\n", ''))) for r in tgt_pres]
        tgt_refs = [' '.join(list(r.replace("，", ',').replace("？", "?").replace(" ", ''))) for r in tgt_refs]

    if args.task == "SLT":
        bleu_dict, rouge_score = translation_performance(tgt_refs, tgt_pres)
        for k, v in bleu_dict.items():
            metric_logger.meters[k].update(v)
        metric_logger.meters['rouge'].update(rouge_score)

    elif args.task == "ISLR":
        top1_acc_pi, top1_acc_pc = islr_performance(tgt_refs, tgt_pres)
        metric_logger.meters['top1_acc_pi'].update(top1_acc_pi)
        metric_logger.meters['top1_acc_pc'].update(top1_acc_pc)

    elif args.task == "CSLR":
        wer_results = wer_list(hypotheses=tgt_pres, references=tgt_refs)
        print(wer_results)
        for k, v in wer_results.items():
            metric_logger.meters[k].update(v)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        with open(args.output_dir + f'/{phase}_tmp_pres.txt', 'w') as f:
            for i in range(len(tgt_pres)):
                f.write(tgt_pres[i] + '\n')
        with open(args.output_dir + f'/{phase}_tmp_refs.txt', 'w') as f:
            for i in range(len(tgt_refs)):
                f.write(tgt_refs[i] + '\n')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
