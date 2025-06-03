"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from apps.main.generate import batch_prompts, pack_prompts, sample_tokens
from lingua.args import dataclass_from_dict
from lingua.checkpoint import CONSOLIDATE_NAME
from lingua.distributed import get_local_rank
from lingua.tokenizer import Tokenizer
from lingua.transformer import (
    causal_mask,
    generate_doc_mask_mod,
    lengths_to_local_ids,
    lengths_to_start_ids,
)
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_mask
from tqdm import tqdm

from projects.mta.mta_transformer import Attention
from projects.mta.tokenizer import build_tokenizer
from projects.mta.transformer import LMTransformer, LMTransformerArgs


class KVCache(nn.Module):
    # in MTA, we also cache queries
    def __init__(self, bsz, seqlen, n_heads, head_dim, dtype, device):
        super().__init__()
        shape = (bsz, seqlen, n_heads, head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=dtype, device=device))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=dtype, device=device))
        self.register_buffer("q_cache", torch.zeros(shape, dtype=dtype, device=device))
        self.offset = 0

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.q_cache.zero_()
        self.offset = 0

    def update(self, k_val, v_val, q_val, tok_idx):
        # input_pos: [B], k_val: [B, S, H, D]
        self.k_cache.index_copy_(1, self.offset + tok_idx, k_val)
        self.v_cache.index_copy_(1, self.offset + tok_idx, v_val)
        self.q_cache.index_copy_(1, self.offset + tok_idx, q_val)
        return (
            self.k_cache,
            self.v_cache,
            self.q_cache,
        )


@dataclass
class PackedCausalTransformerGeneratorArgs:
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    max_gen_len: int = 512  # Maximum number of tokens to generate
    max_tokens: int = 2048  # Maximum number of tokens that can go through the model
    max_prompt_len: Optional[int] = None
    until: List[str] = field(default_factory=list)
    compile_prefilling: bool = False
    reduce_generation_overhead: bool = False
    show_progress: bool = False
    dtype: Optional[str] = "bf16"
    device: Optional[str] = "cuda"
    dump_dir: Optional[str] = "/home/olggol/tmp/830M_head_conv_2_mta_4l"


class PackedCausalTransformerGenerator:
    def __init__(
        self,
        cfg: PackedCausalTransformerGeneratorArgs,
        model: nn.Module,
        tokenizer: Tokenizer,
    ):
        """
        This class wraps a causal transformer model with its corresponding tokenizer
        and provides an efficient way to pack prompts together and do generation on
        the packed sequence.

        For example, if we had the prompts "Hello, I am a " and "Initiating calibration "
        Then this class will concatenate those sequence (pack them together)
        "Hello, I am a Initiating calibration"
        And make the necessary attention masks such that a sequence only attends to itself
        during prefilling and generation.

        This class creates a fixed size cache of size max_tokens or sum of prompt sizes
        + the max number of generated tokens per sequence.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.top_k = cfg.top_k

        self.max_gen_len = cfg.max_gen_len
        self.max_tokens = cfg.max_tokens
        self.max_prompt_len = cfg.max_prompt_len
        self.until = cfg.until
        self.max_until_size = max([len(e) for e in self.until]) if self.until else 1
        self.device = cfg.device

        # Compile if necessary
        self.prefill = torch.compile(self.prefill, disable=not cfg.compile_prefilling)
        self.generate_next_token = torch.compile(
            self.generate_next_token,
            mode="reduce-overhead",
            disable=not cfg.reduce_generation_overhead,
        )

        self.show_progress = cfg.show_progress
        self.dtype = dict(fp32=torch.float32, bf16=torch.bfloat16)[cfg.dtype]

        self.prefill_doc_id, self.prefill_tok_id = None, None
        self.padded_doc_id, self.padded_tok_id = None, None
        self.current_doc_id, self.current_tok_id = None, None
        self.padded_doc_start = None
        self.prefill_mask = None
        self.dump_dir = cfg.dump_dir

    def clear_cache(self, offset):
        for module in self.model.modules():
            if isinstance(module, Attention):
                if not hasattr(module, "kv_cache"):
                    module.kv_cache = KVCache(
                        1,
                        self.max_tokens,
                        module.n_kv_heads,
                        module.head_dim,
                        self.dtype,
                        self.device,
                    )
                module.kv_cache.offset = offset

    @torch.compiler.disable
    def setup_prefilling(self, lengths: torch.Tensor):
        # The KV cache is a fixed size tensor of size max_tokens that we need
        # to update in order to do correct autoregressive generation.

        # Here we will generate token by token but on multiple sequences
        # at once. To do so, we need to have an attention mask that makes
        # each sequence independent.

        # Each sequence will write to its allocated space in the KV Cache.
        # We allocate len(seq) + max_gen_len to each sequence in the cache.

        # We will generate max_gen_len for each document
        padded_lengths = lengths + self.max_gen_len
        max_tokens = self.max_tokens or padded_lengths.sum().item()
        # The last document might have more padding to fill up to max_tokens
        padded_lengths[-1] += max_tokens - padded_lengths.sum()

        # This is the start index in the cache for each document
        self.padded_doc_start = lengths_to_start_ids(padded_lengths)
        # For example with ab--123--cdef--
        # this would be 0, 4, 9 if max_gen_len is 2

        # We repeat interleave to align with tokens for prefilling
        # Ex: ab--123--cdef--
        #     000044444999999
        prefill_offset = torch.repeat_interleave(self.padded_doc_start, lengths)
        # This offset will make sure the tokens are written to the
        # correct positions in the cache during prefilling

        # We either init the cache or clear it by resetting the offset to prefill_offset
        self.clear_cache(prefill_offset)

        # The prefilling mask looks like the following for
        # the two packed sequences ab and 123 : ab123
        # Where spaces are empty cache positions
        #                 keys
        #                ab---123---
        #   queries    a 10000000000
        #              b 11000000000
        #              1 00000100000
        #              2 00000110000
        #              3 00000111000
        # We make sure to skip the empty cache positions
        # and only attend to positions within the same sequence
        doc_mask_mod = generate_doc_mask_mod(
            causal_mask, padded_lengths, padded_lengths
        )
        self.prefill_mask = create_mask(
            doc_mask_mod, B=1, H=1, Q_LEN=padded_lengths.sum(), KV_LEN=max_tokens
        )

        # This creates the prefilling token ids which look like
        # the following for the packed sequence abcdefg1234
        # abcdefg1234
        # 01234560123
        # The token id gives us the position within each sequence
        # This is used to compute ROPE and to update the cache
        # At each forward pass the current tokens are written to
        # offset + tok_id
        self.prefill_doc_id, self.prefill_tok_id = lengths_to_local_ids(lengths)

        # This creates the padded token and document ids
        # which look like the following for the packed sequence ab123
        #               ab---123---               ab---123---
        # padded_doc_id 00000111111 padded_tok_id 01234012345
        # This will later be useful for the attention mask at generation
        self.padded_doc_id, self.padded_tok_id = lengths_to_local_ids(padded_lengths)

    @torch.compiler.disable
    def setup_generation(self, lengths):
        # KV Cache offset is set to the start of the padded documents
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.kv_cache.offset = self.padded_doc_start
        # The token ids during generations correspond to the lengths of each doc
        # current_tok_id will be incremented during generation
        self.current_tok_id = lengths.clone()
        # Since we're generating one token per document
        # the document id is just an arange
        self.current_doc_id = torch.arange(lengths.size(0), device=lengths.device)

    # From here on some methods for generation
    def prefill(self, tokens: torch.Tensor, lengths: torch.Tensor):
        # Prefilling is done by taking multiple packed sequences and
        # doing block diagonal attention on them so they remain independent
        self.setup_prefilling(lengths=lengths)
        prefill_out = self.model.forward(
            tokens,
            tok_idx=self.prefill_tok_id,
            mask=self.prefill_mask,
            attn_impl="sdpa",  # "flex_attention",
        )
        self.setup_generation(lengths=lengths)
        return prefill_out

    def generate_next_token(self, current_token):
        # Since we're doing generation with multiple sequences at once
        # we need to ignore tokens and cache entries from other sequences
        # or in the future.
        # Example mask :
        #                  keys
        #                abc--1234--
        #   queries    c 11100000000
        #              4 00000111100

        # mask shape : (n_seqs, cache_size)
        # doc_mask = self.current_doc_id.unsqueeze(1) == self.padded_doc_id.unsqueeze(0)
        # caus_mask = self.current_tok_id.unsqueeze(1) >= self.padded_tok_id.unsqueeze(0)
        # mask = doc_mask & caus_mask
        # self.prefill_mask[:, :, self.current_tok_id] = mask
        out = self.model.forward(
            current_token,
            tok_idx=self.current_tok_id,  # n_seqs
            mask=self.prefill_mask,
            attn_impl="sdpa",
        )
        self.current_tok_id += 1
        return out

    @torch.inference_mode()
    def generate(
        self,
        prompts,
        tgt_ppl_only: bool = False,
        src_tgt_sep: str = "<SRC_TGT_SEP>",
        save_maps: bool = False,
    ):
        # Tokenize
        if src_tgt_sep in prompts[0]:
            tok_prompts = []
            tgt = []
            for p in prompts:
                src, text_tgt = p.split(src_tgt_sep)
                tok_prompts.append(
                    self.tokenizer.encode(src + text_tgt, add_bos=True, add_eos=False)
                )
                tgt.append(
                    len(self.tokenizer.encode(text_tgt, add_bos=False, add_eos=False))
                )
            prompts = tok_prompts
        else:
            prompts = [
                self.tokenizer.encode(p, add_bos=True, add_eos=False) for p in prompts
            ]
        # Truncate
        max_seqlen = (
            self.max_tokens
            if not hasattr(self.model, "max_seqlen")
            else self.model.max_seqlen
        )
        max_prompt_len = self.max_prompt_len or min(
            max_seqlen - self.max_gen_len, self.max_tokens - self.max_gen_len
        )
        prompts = [p[-max_prompt_len:] for p in prompts]
        assert self.max_tokens >= max_prompt_len, max_prompt_len
        # Account for the generation in lengths
        padded_lengths = [len(p) + self.max_gen_len for p in prompts]

        generation = []
        loglikelihood = []
        greedy = []

        it = batch_prompts(prompts, self.max_tokens, lengths=padded_lengths)
        if self.show_progress:
            it = tqdm(it)
        global_seq_id = 0
        for batch_id, batch in enumerate(it):
            n_seqs = len(batch)
            generated_tokens = [[] for _ in range(n_seqs)]
            is_done = [False for _ in range(n_seqs)]
            packed_batch, lengths = pack_prompts(batch)
            packed_batch, lengths = packed_batch.cuda(), lengths.cuda()
            n_seqs = lengths.size(0)

            # Prefilling cache
            prompt_logits = self.prefill(packed_batch.unsqueeze(0), lengths)
            # Selecting last token in each prompt
            all_tokens = sample_tokens(
                prompt_logits, self.temperature, self.top_p, self.top_k
            )
            start_token = all_tokens[:, lengths.cumsum(0) - 1]

            for seq_id, tok in enumerate(start_token.squeeze(0).tolist()):
                generated_tokens[seq_id].append(tok)

            current_token = start_token
            for i in range(1, self.max_gen_len):

                next_logits = self.generate_next_token(current_token)
                next_token = sample_tokens(
                    next_logits.clone(), self.temperature, self.top_p, self.top_k
                )

                for seq_id, tok in enumerate(next_token.squeeze(0).tolist()):
                    if not is_done[seq_id]:
                        generated_tokens[seq_id].append(tok)
                        current_end_str = self.tokenizer.decode(
                            generated_tokens[seq_id][-self.max_until_size :]
                        )
                        contains_end_string = any(
                            [e in current_end_str for e in self.until]
                        )
                        is_done[seq_id] = (
                            contains_end_string or tok == self.tokenizer.eos_id
                        )
                if all(is_done):
                    break

                current_token = next_token

            generation.extend([self.tokenizer.decode(g) for g in generated_tokens])

            for p, logit in zip(
                batch, prompt_logits.squeeze(0).split(lengths.tolist())
            ):
                if tgt_ppl_only:
                    x = logit[tgt[global_seq_id] : -1]
                    y = torch.tensor(p[tgt[global_seq_id] + 1 :], device=x.device)
                    global_seq_id += 1
                else:
                    x = logit[:-1]
                    y = torch.tensor(p[1:], device=x.device)
                loglikelihood.append(-F.cross_entropy(x, y, reduction="none").cpu())
                greedy.append((x.argmax(dim=-1) == y).cpu())

            # save maps
            if save_maps:
                local_rank = get_local_rank()
                is_rank_0 = local_rank == 0
                if self.dump_dir:
                    out_tokens = []
                    sample_lengths = torch.zeros(n_seqs, dtype=torch.int)
                    for i, gen_toks in enumerate(generated_tokens):
                        # cut to max gen len
                        toks = prompts[i] + gen_toks
                        sample_lengths[i] = len(toks)  # probably shape
                        # cut to eos tok if any
                        if self.tokenizer.eos_id and self.tokenizer.eos_id in toks:
                            eos_idx = toks.index(self.tokenizer.eos_id)
                            # toks = toks[:eos_idx]
                            sample_lengths[i] = eos_idx
                            out_tokens.append(toks[:eos_idx])
                        else:
                            out_tokens.append(toks)
                        # out_logprobs.append(probs)

                    max_sample_length = torch.max(sample_lengths)

                    if is_rank_0:
                        if not os.path.isdir(f"{self.dump_dir}/{batch_id}"):
                            os.makedirs(f"{self.dump_dir}/{batch_id}")
                        torch.save(
                            sample_lengths,
                            f"{self.dump_dir}/{batch_id}/last_tok_indices",
                        )
                        # if self.model.params.is_dump_rs:
                        #     rs0_cache = self.model.layers[0].cache_rs0[:bsz,:max_sample_length - 1, :]
                        #     torch.save(rs0_cache,f"{dump_dir}/{batch_id}/rs0_dump")
                        torch.save(
                            out_tokens,
                            f"{self.dump_dir}/{batch_id}/output_token_indices",
                        )
                        actual_tokens = []
                        for s in out_tokens:
                            actual_tokens.append(
                                [self.tokenizer.decode([t]) for t in s]
                            )
                        torch.save(
                            actual_tokens, f"{self.dump_dir}/{batch_id}/output_tokens"
                        )
                    else:
                        time.sleep(0.1)
                    for l in range(0, self.model.n_layers):
                        a_cache_attn = self.model.layers[l].attention.cache_attn[
                            :, :, : max_sample_length - 1, : max_sample_length - 1
                        ]
                        torch.save(
                            a_cache_attn,
                            f"{self.dump_dir}/{batch_id}/attn_dump.{l:02d}.{local_rank}",
                        )

        return generation, loglikelihood, greedy


def load_consolidated_model_and_tokenizer(
    consolidated_path,
    model_cls=LMTransformer,
    model_args_cls=LMTransformerArgs,
    is_dump_attn: Optional[bool] = False,
):
    ckpt_path = Path(consolidated_path)
    config = ckpt_path / "params.json"
    config = OmegaConf.load(config)

    config.model.cache_attention_maps = is_dump_attn

    param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
        config.distributed.model_dtype
    ]
    model_args = dataclass_from_dict(model_args_cls, config.model, strict=False)
    tokenizer = build_tokenizer(config.data.tokenizer.name, config.data.tokenizer.path)
    model = model_cls(model_args)
    st_dict = torch.load(ckpt_path / CONSOLIDATE_NAME, weights_only=True)
    model.load_state_dict(st_dict["model"])
    model = model.cuda().eval()
    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)
    return model, tokenizer, config


def main():
    # Load CLI arguments (overrides) and combine with a YAML config
    cfg = OmegaConf.from_cli()
    gen_cfg = dataclass_from_dict(
        PackedCausalTransformerGeneratorArgs, cfg, strict=False
    )

    model, tokenizer, _ = load_consolidated_model_and_tokenizer(
        cfg.ckpt, is_dump_attn=True
    )
    model.eval()

    generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)

    prompts = []
    while True:
        prompt = input("Enter a prompt (or press enter to finish): ")
        if not prompt:
            break
        prompts.append(prompt)

    # Start generation
    start_time = time.time()
    generation, loglikelihood, greedy = generator.generate(prompts, save_maps=True)
    end_time = time.time()

    # Calculate tokens per second
    total_tokens = sum(len(tokenizer.encode(gen, False, False)) for gen in generation)
    tokens_per_second = total_tokens / (end_time - start_time)

    # Display the results
    for i, gen in enumerate(generation):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Generated Text: {gen}")

    print(f"\nTokens per second: {tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
