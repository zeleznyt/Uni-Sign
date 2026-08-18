import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, WeightedRandomSampler
from models import Uni_Sign
import utils as utils
from datasets import Combined_Dataset, S2T_Dataset, S2T_Dataset_YTASL, S2T_Dataset_Isharah
#S2T_Dataset_YTASL_h5
import os
import time
import argparse, json, datetime
import shutil
from pathlib import Path
import math
import sys
import random
from timm.optim import create_optimizer
from models import get_requires_grad_dict
from SLRT_metrics import translation_performance, islr_performance, wer_list
from transformers import get_scheduler
from config import *
from data_config import (
    format_data_setup_report,
    get_required_split_specs,
    load_data_config,
    normalize_split_specs,
    preflight_data_config,
    spec_name,
    spec_pose_roots,
    spec_rgb_config,
)
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


def _cleanup_old_ds_checkpoints(output_dir, keep_tag):
    for ckpt_dir in output_dir.glob("checkpoint_*"):
        if not ckpt_dir.is_dir():
            continue
        if ckpt_dir.name == keep_tag:
            continue
        shutil.rmtree(ckpt_dir, ignore_errors=True)


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


def get_optimizer_step(model, fallback):
    global_steps = getattr(model, "global_steps", None)
    if callable(global_steps):
        global_steps = global_steps()
    if global_steps is None:
        return int(fallback)
    if isinstance(global_steps, torch.Tensor):
        global_steps = global_steps.item()
    return int(global_steps)


def get_optimizer_steps_per_epoch(data_loader, gradient_accumulation_steps):
    return math.ceil(len(data_loader) / gradient_accumulation_steps)


def make_train_wandb_stats(train_stats):
    log_stats = {}
    for key, value in train_stats.items():
        if key == "lr":
            log_stats["train/lr_epoch_avg"] = value
        elif key == "loss":
            log_stats["train/loss_epoch_avg"] = value
        else:
            log_stats[f"train/{key}_epoch_avg"] = value
    return log_stats


def apply_data_config_to_args(args, data_config):
    args.layout = data_config["layout"]
    args.graph = data_config["graph"]
    args.target_language = data_config["target_language"]
    args.dataset = f"config:{Path(args.data_config).stem}"
    if "normalization" in data_config:
        args.normalization = data_config["normalization"]

    if any(spec_rgb_config(spec) is not None for split in ("train", "dev", "test") for spec in normalize_split_specs(data_config, split)):
        args.rgb_support = False


def build_dataset_from_spec(spec, args, phase):
    loader = spec.get("loader")
    dataset_name = spec_name(spec)
    annotation_path = spec.get("annotation_path")
    pose_roots = spec_pose_roots(spec)
    rgb = spec_rgb_config(spec)

    if loader == "ytasl_json":
        return S2T_Dataset_YTASL(
            path=annotation_path,
            args=args,
            phase=phase,
            pose_roots=pose_roots,
            rgb=rgb,
            dataset_name=dataset_name,
            loader=loader,
        )
    if loader == "isharah_json":
        return S2T_Dataset_Isharah(
            path=annotation_path,
            args=args,
            phase=phase,
            pose_roots=pose_roots,
            rgb=rgb,
            dataset_name=dataset_name,
            loader=loader,
        )
    raise NotImplementedError(f"Data config loader '{loader}' is not implemented.")


def build_split_dataset(specs, args, phase):
    if len(specs) == 0:
        return None

    datasets = [build_dataset_from_spec(spec, args, phase) for spec in specs]
    if len(datasets) == 1:
        return datasets[0]

    return Combined_Dataset(
        datasets=datasets,
        names=[spec_name(spec) for spec in specs],
        weights=[spec.get("weight", 1.0) for spec in specs],
        phase=phase,
    )


def get_dataset_setup_summaries(dataset):
    if dataset is None:
        return []
    if hasattr(dataset, "get_setup_summaries"):
        return dataset.get_setup_summaries()
    if hasattr(dataset, "get_setup_summary"):
        return [dataset.get_setup_summary()]
    return []


def make_legacy_dataset(args, phase):
    if phase == "train":
        label_paths = train_label_paths
    elif phase == "dev":
        label_paths = dev_label_paths
    elif phase == "test":
        label_paths = test_label_paths
    else:
        raise ValueError(f"Unknown phase: {phase}")

    if args.dataset == "YTASL":
        return S2T_Dataset_YTASL(path=label_paths[args.dataset], args=args, phase=phase)
    if args.dataset == "Isharah":
        return S2T_Dataset_Isharah(path=label_paths[args.dataset], args=args, phase=phase)
    return S2T_Dataset(path=label_paths[args.dataset], args=args, phase=phase)


def build_datasets(args):
    data_setup_text = None
    if not args.data_config:
        if utils.is_main_process():
            print("WARNING: using legacy --dataset path configuration. Prefer --data_config for new dataset runs.")
        if args.dataset in ("YTASL", "Isharah") and args.rgb_support:
            print(f"WARNING: RGB is not implemented for legacy {args.dataset} JSON loading; continuing pose-only.")
            args.rgb_support = False
        train_data = make_legacy_dataset(args, "train")
        dev_data = make_legacy_dataset(args, "dev")
        test_data = make_legacy_dataset(args, "test")
        return train_data, dev_data, test_data, data_setup_text

    data_config = load_data_config(args.data_config)
    apply_data_config_to_args(args, data_config)
    preflight_report = preflight_data_config(data_config)
    if preflight_report["errors"]:
        print(format_data_setup_report(preflight_report))
        raise FileNotFoundError("Data config contains missing required paths; see setup report above.")

    train_specs = get_required_split_specs(data_config, "train")
    train_data = build_split_dataset(train_specs, args, "train")
    dev_data = build_split_dataset(normalize_split_specs(data_config, "dev"), args, "dev")
    test_data = build_split_dataset(normalize_split_specs(data_config, "test"), args, "test")

    summaries = {
        "train": get_dataset_setup_summaries(train_data),
        "dev": get_dataset_setup_summaries(dev_data),
        "test": get_dataset_setup_summaries(test_data),
    }
    data_setup_text = format_data_setup_report(preflight_report, summaries)
    return train_data, dev_data, test_data, data_setup_text


def make_train_sampler(args, train_data):
    if args.distributed:
        if hasattr(train_data, "sample_weights"):
            weights = train_data.sample_weights()
            if any(weight != 1.0 for weight in weights):
                print("WARNING: dataset balancing weights are ignored with DistributedSampler.")
        return torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)

    if hasattr(train_data, "sample_weights"):
        weights = train_data.sample_weights()
        if any(weight != 1.0 for weight in weights):
            return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return torch.utils.data.RandomSampler(train_data)


def make_eval_dataloader(args, dataset, eval_num_workers):
    if dataset is None:
        return None
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=False,
            drop_last=False
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=eval_num_workers,
        collate_fn=dataset.collate_fn,
        sampler=sampler,
        pin_memory=args.pin_mem,
    )


def main(args):
    utils.init_distributed_mode_ds(args)

    print(args)
    utils.set_seed(args.seed)

    if args.finetune and args.resume:
        raise ValueError("Use only one of --finetune (weights-only) or --resume (full state).")

    wandb_run_id = None
    wandb_run_name = None

    print(f"Creating dataset:")

    train_data, dev_data, test_data, data_setup_text = build_datasets(args)
    if len(train_data) == 0:
        raise ValueError("Train split has zero usable samples after matching annotations to pose files.")
    if dev_data is not None and len(dev_data) == 0:
        print("WARNING: dev split has zero usable samples after matching annotations to pose files; treating it as missing.")
        dev_data = None
    if test_data is not None and len(test_data) == 0:
        print("WARNING: test split has zero usable samples after matching annotations to pose files; treating it as missing.")
        test_data = None
    print(train_data)
    train_sampler = make_train_sampler(args, train_data)
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



    print(dev_data)
    eval_num_workers = 0 if args.zero_workers_for_eval else args.num_workers
    dev_dataloader = make_eval_dataloader(args, dev_data, eval_num_workers)

    print(test_data)
    test_dataloader = make_eval_dataloader(args, test_data, eval_num_workers)

    if not args.eval and dev_dataloader is None:
        raise ValueError("Training requires a dev split for checkpoint selection; data config 'dev' is null/missing.")

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
        wandb.define_metric("train/optimizer_step")
        wandb.define_metric("train/micro_step")
        wandb.define_metric("train/*", step_metric="train/optimizer_step")
        wandb.define_metric("dev/*", step_metric="train/optimizer_step")
        wandb.define_metric("test/*", step_metric="train/optimizer_step")

    if data_setup_text and utils.is_main_process():
        print(data_setup_text)
        if args.wandb and wandb.run:
            wandb.config.update({"data_setup": data_setup_text}, allow_val_change=True)

    if args.eval:
        if dev_dataloader is None and test_dataloader is None:
            raise ValueError("Eval requested, but both dev and test splits are null/missing.")
        # Run eval on all ranks to keep DeepSpeed/NCCL collectives aligned.
        if dev_dataloader is not None:
            if utils.is_main_process():
                print("📄 dev result")
            evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev')
        elif utils.is_main_process():
            print("WARNING: dev split is null/missing; skipping dev evaluation.")
        if test_dataloader is not None:
            if utils.is_main_process():
                print("📄 test result")
            evaluate(args, test_dataloader, model, model_without_ddp, phase='test')
        elif utils.is_main_process():
            print("WARNING: test split is null/missing; skipping test evaluation.")

        return
    print(f"Start training for {args.epochs} epochs")
    optimizer_steps_per_epoch = get_optimizer_steps_per_epoch(train_dataloader, args.gradient_accumulation_steps)
    if utils.is_main_process() and args.wandb and start_epoch == 0:
        wandb.log({
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/optimizer_step": 0,
            "train/micro_step": 0,
            "epoch": 0,
        }, step=0)

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(args, model, train_dataloader, optimizer, epoch)

        if args.output_dir:
            current_tag = f'checkpoint_{epoch}'
            ds_client_state = {
                'epoch': epoch,
                'max_accuracy': max_accuracy,
                'wandb_run_id': wandb.run.id if args.wandb and utils.is_main_process() and wandb.run else None,
                'wandb_run_name': wandb.run.name if args.wandb and utils.is_main_process() and wandb.run else None,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'numpy_rng_state': np.random.get_state(),
                'random_rng_state': random.getstate(),
                'global_step': get_optimizer_step(model, (epoch + 1) * optimizer_steps_per_epoch),
                'optimizer_step': get_optimizer_step(model, (epoch + 1) * optimizer_steps_per_epoch),
                'micro_step': (epoch + 1) * len(train_dataloader),
            }
            model.save_checkpoint(str(output_dir), tag=current_tag, client_state=ds_client_state)
            if args.distributed and torch.distributed.is_initialized():
                # Keep all ranks aligned before rank-0-only evaluation/logging.
                torch.distributed.barrier()
            if utils.is_main_process():
                _cleanup_old_ds_checkpoints(output_dir, keep_tag=current_tag)
            if args.distributed and torch.distributed.is_initialized():
                # Keep ranks aligned after rank-0 cleanup before collectives in eval.
                torch.distributed.barrier()

        # Evaluate on all ranks so DeepSpeed collective ops remain matched.
        dev_stats = evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev')
        # evaluate(args, test_dataloader, model, model_without_ddp, phase='test', eval_header='Test evaluation:')

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

                print(f"BLEU-4 of the network on the {len(dev_dataloader.dataset)} dev videos: {dev_stats['bleu4']:.2f}")
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
                    f"PI accuracy of the network on the {len(dev_dataloader.dataset)} dev videos: {dev_stats['top1_acc_pi']:.2f}")
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

            optimizer_step = get_optimizer_step(model, (epoch + 1) * optimizer_steps_per_epoch)
            micro_step = (epoch + 1) * len(train_dataloader)
            log_stats = {**make_train_wandb_stats(train_stats),
                         **{f'dev/{k}': v for k, v in dev_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters,
                         'train/optimizer_step': optimizer_step,
                         'train/micro_step': micro_step,
                         'train/lr': optimizer.param_groups[0]["lr"]}
            epoch_elapsed = time.time() - epoch_start_time
            log_stats['train/epoch_elapsed_sec'] = epoch_elapsed
            if args.wandb:
                wandb.log(log_stats, step=optimizer_step)

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
    optimizer_steps_per_epoch = get_optimizer_steps_per_epoch(data_loader, args.gradient_accumulation_steps)

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
        if hasattr(model, "is_gradient_accumulation_boundary"):
            is_optimizer_update = model.is_gradient_accumulation_boundary()
        else:
            is_optimizer_update = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(data_loader)
        model.step()

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        micro_step = epoch * len(data_loader) + step + 1
        optimizer_step_fallback = epoch * optimizer_steps_per_epoch + math.ceil((step + 1) / args.gradient_accumulation_steps)
        optimizer_step = get_optimizer_step(model, optimizer_step_fallback)
        should_log_step = is_optimizer_update and (optimizer_step == 1 or optimizer_step % args.log_step == 0)
        if utils.is_main_process() and args.wandb and should_log_step:
            elapsed_time = time.time() - start_time
            log_dict = {
                "train/loss": loss_value,
                "train/loss_raw": loss_value,
                "train/loss_epoch_avg": metric_logger.loss.global_avg,
                "train/lr": metric_logger.lr.value,
                "train/lr_epoch_avg": metric_logger.lr.global_avg,
                "train/iter_time": elapsed_time,
                "train/optimizer_step": optimizer_step,
                "train/micro_step": micro_step,
                "train/epoch_batch": step + 1,
            }
            wandb.log(log_dict, step=optimizer_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(args, data_loader, model, model_without_ddp, phase, eval_header='Evaluation:'):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    target_dtype = None
    if model.bfloat16_enabled():
        target_dtype = torch.bfloat16

    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []
        sample_names = []

        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, eval_header)):
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
                sample_names.append(src_input['name_batch'][i])

    tokenizer = model_without_ddp.mt5_tokenizer
    padding_value = tokenizer.eos_token_id

    pad_tensor = torch.ones(150 - len(tgt_pres[0])).cuda() * padding_value
    tgt_pres[0] = torch.cat((tgt_pres[0], pad_tensor.long()), dim=0)

    tgt_pres = pad_sequence(tgt_pres, batch_first=True, padding_value=padding_value)
    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)

    if args.distributed and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        world_size = torch.distributed.get_world_size()
        gathered_pres = [None for _ in range(world_size)]
        gathered_refs = [None for _ in range(world_size)]
        gathered_names = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered_pres, tgt_pres)
        torch.distributed.all_gather_object(gathered_refs, tgt_refs)
        torch.distributed.all_gather_object(gathered_names, sample_names)

        tgt_pres = [x for rank_list in gathered_pres for x in rank_list]
        tgt_refs = [x for rank_list in gathered_refs for x in rank_list]
        sample_names = [x for rank_list in gathered_names for x in rank_list]

        # DistributedSampler with drop_last=False may pad by repeating some samples.
        unique = {}
        for name, pred, ref in zip(sample_names, tgt_pres, tgt_refs):
            if name not in unique:
                unique[name] = (pred, ref)
        sorted_names = sorted(unique.keys())
        tgt_pres = [unique[name][0] for name in sorted_names]
        tgt_refs = [unique[name][1] for name in sorted_names]
        sample_names = sorted_names

    # fix mt5 tokenizer bug
    if args.dataset == 'CSL_Daily' and args.task == "SLT" and args.original_metric_implementation:
        tgt_pres = [' '.join(list(r.replace(" ", '').replace("\n", ''))) for r in tgt_pres]
        tgt_refs = [' '.join(list(r.replace("，", ',').replace("？", "?").replace(" ", ''))) for r in tgt_refs]

    if utils.is_main_process():
        preview_n = min(5, len(tgt_pres), len(tgt_refs))
        if preview_n > 0:
            print(f"[eval-samples] showing {preview_n} ref/pred pairs")
            for i in range(preview_n):
                print(f"[sample {i+1}] REF: {tgt_refs[i]}")
                print(f"[sample {i+1}] PRED: {tgt_pres[i]}")

    if args.task == "SLT":
        bleu_dict, rouge_score = translation_performance(
            tgt_refs,
            tgt_pres,
            original_metric_implementation=args.original_metric_implementation,
            bleu_effective_order=args.bleu_effective_order,
        )
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

    result_metrics = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if utils.is_main_process() and args.eval:
        qualitative_limit = min(100, len(tgt_pres), len(tgt_refs), len(sample_names))
        qualitative_results = [
            {
                "name": sample_names[i],
                "prediction": tgt_pres[i],
                "reference": tgt_refs[i],
            }
            for i in range(qualitative_limit)
        ]
        result_payload = {
            "metrics": result_metrics,
            "predictions": qualitative_results,
        }
        with open(args.output_dir + f'/{phase}_results.json', 'w') as f:
            json.dump(result_payload, f, ensure_ascii=False, indent=4)

    return result_metrics


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
