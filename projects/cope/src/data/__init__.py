import argparse
import logging

from torch.utils.data import DataLoader, DistributedSampler

from . import simple
from .data_collator import DataCollatorForDecoder


def add_args(parser: argparse.ArgumentParser):
    parser = parser.add_argument_group("data")
    parser.add_argument(
        "--task",
        choices=[
            "simple",
        ],
        default="simple",  # todo remove flag
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--train-file", default="train.jsonl", type=str)
    parser.add_argument("--val-file", default="valid.jsonl", type=str)
    parser.add_argument("--test-file", default="test.jsonl", type=str)
    parser.add_argument(
        "--fixed-len",
        type=int,
        default=-1,
        help="cut and pad data samples into a fixed length. reduces memory fragmentation",
    )


def get_data(cfg, tokenizer):
    # Convert text to numbers
    train_data, val_data, test_data = simple.get_data(cfg)
    tokenizer.build_vocab(train_data, val_data, test_data)

    cfg.nvocab = tokenizer.vocab_size
    logging.info(f"nvocab = {cfg.nvocab}")

    return train_data, val_data, test_data, tokenizer


def get_loader(cfg, data, tokenizer, eval=False):
    """Get data loader and sampler for training data."""
    collator = DataCollatorForDecoder(cfg, tokenizer, cfg.fixed_len)

    if cfg.distributed:
        sampler = DistributedSampler(
            data,
            num_replicas=cfg.world_size,
            rank=cfg.rank,
            shuffle=not eval,
            seed=cfg.seed,  # must be the same for all workers
            drop_last=True,
        )
        assert cfg.batch_sz % cfg.world_size == 0
        if eval and len(data) % cfg.world_size != 0:
            logging.warning(
                "eval data size is not divisible by ngpus, so some samples will be omitted!"
            )
        new_batch_sz = cfg.batch_sz // cfg.world_size
        loader = DataLoader(
            data,
            batch_size=new_batch_sz,
            sampler=sampler,
            pin_memory=True,
            collate_fn=collator,
        )
    else:
        loader = DataLoader(
            data,
            batch_size=cfg.batch_sz,
            shuffle=not eval,
            drop_last=False,
            collate_fn=collator,
        )

    return loader
