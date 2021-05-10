# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
import torch

from argparse import Namespace
from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Any
from omegaconf import MISSING, II

from fairseq.data import (
    AddTargetDataset,
    BinarizedAudioDataset,
    FileAudioDatasetShelve,
    Dictionary,
    FileAudioDataset,
    encoders,
)
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig

from . import FairseqTask, register_task
from .. import utils
from ..logging import metrics

from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer


logger = logging.getLogger(__name__)


class HF2FairSeqDictWrapper(Dictionary):
    def __init__(self, tokenizer_string: str, hf_tokenizer: PreTrainedTokenizer = None):
        if hf_tokenizer is None:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_string)
        else:
            self.hf_tokenizer = hf_tokenizer
        
        self.indices = self.hf_tokenizer.get_vocab()
        self.symbols = list(self.hf_tokenizer.get_vocab().keys())
        self.count = []
        self.nspecial = 4
        
        self.bos_index = self.hf_tokenizer.bos_token_id
        self.pad_index = self.hf_tokenizer.pad_token_id
        self.eos_index = self.hf_tokenizer.eos_token_id
        self.unk_index = self.hf_tokenizer.unk_token_id
        
        self.bos_word = self.hf_tokenizer.bos_token
        self.pad_word = self.hf_tokenizer.pad_token
        self.eos_word = self.hf_tokenizer.eos_token
        self.unk_word = self.hf_tokenizer.unk_token
    
    '''
    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.hf_tokenizer.bos_token_id

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.hf_tokenizer.pad_token_id

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.hf_tokenizer.eos_token_id

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.hf_tokenizer.unk_token_id
    
    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.hf_tokenizer)
    
    def __contains__(self, sym):
        return sym in self.hf_tokenizer.get_vocab()
    
    def __getitem__(self, idx):
        if idx < len(self.hf_tokenizer):
            return self.hf_tokenizer.convert_ids_to_tokens(idx)
        return self.hf_tokenizer.unk_token
    
    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.hf_tokenizer.get_vocab():
            return self.hf_tokenizer.convert_tokens_to_ids(sym)
        return self.hf_tokenizer.unk_token_id
    '''
    
    def string(
        self,
        tensor,
        bpe_symbol=None,
        escape_unk=False,
        extra_symbols_to_ignore=None,
        unk_string=None,
        include_eos=False,
    ):
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(
                self.string(t, bpe_symbol, escape_unk, extra_symbols_to_ignore, include_eos=include_eos)
                for t in tensor
            )
        
        return self.hf_tokenizer.decode(tensor, skip_special_tokens=not include_eos)
    
    def encode_line(
        self,
        line,
        #line_tokenizer=tokenize_line,
        add_if_not_exist=True,
        consumer=None,
        append_eos=True,
        reverse_order=False,
    ):
        tokens = self.hf_tokenizer.encode(line, add_special_tokens=False)
        
        if reverse_order:
            tokens = list(reversed(tokens))
        #nwords = len(tokens)
        
        if append_eos:
            tokens.append(self.hf_tokenizer.eos_token_id)
        
        return torch.IntTensor(tokens)


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


@dataclass
class AudioPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "extension of the label file to load, used for fine-tuning"},
    )
    binarized_dataset: bool = field(
        default=False,
        metadata={
            "help": "if true, loads binarized dataset (useful for very large datasets). "
            "See examples/wav2vec/scripts/binarize_manifest.sh"
        },
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False, metadata={"help": "pad shorter samples instead of cropping"}
    )
    max_sample_size: Optional[int] = field(
        default=None, metadata={"help": "max sample size to crop to for batching"}
    )
    min_sample_size: Optional[int] = field(
        default=None, metadata={"help": "min sample size to skip small examples"}
    )
    max_sample_length: Optional[int] = field(
        default=None, metadata={"help": "max sample size to skip large examples"}
    )

    # Options for reporting WER metrics during validation. Only applicable to
    # Seq2Seq models during fine-tuning
    eval_wer: bool = field(
        default=False, metadata={"help": "compute WER for Seq2Seq models"}
    )
    eval_wer_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(),
        metadata={"help": "beam search config for evaluating wer during training"},
    )
    eval_wer_tokenizer: Any = field(
        default=None,
        metadata={"help": "tokenizer config for evaluating wer during training"},
    )
    eval_wer_post_process: str = field(
        default="letter",
        metadata={
            "help": "remove BPE tokens before scoring (can be sentencepiece, letter, and more)"
        },
    )
    autoregressive: bool = field(
        default=False,
        metadata={
            "help": "required for autoregressive decoders (like seq2seq models); "
            "adds 'prev_output_tokens' to input and appends eos to target"
        },
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={"help": "number of buckets"},
    )
    precompute_mask_indices: bool = field(
        default=False,
        metadata={
            "help": "flag to compute mask indices in data preparation.",
        },
    )
    hf_tokenizer: Optional[str] = field(
        default=None, metadata={"help": "optional HuggingFace tokenizer to use"}
    )
    # The following are needed to precompute mask and mask channel indices
    #   before model's forward.
    mask_length: Optional[int] = II("model.mask_length")
    mask_prob: Optional[float] = II("model.mask_prob")
    mask_selection: Optional[str] = II("model.mask_selection")
    mask_other: Optional[float] = II("model.mask_other")
    no_mask_overlap: Optional[bool] = II("model.no_mask_overlap")
    mask_min_space: Optional[int] = II("model.mask_min_space")
    mask_channel_length: Optional[int] = II("model.mask_channel_length")
    mask_channel_prob: Optional[float] = II("model.mask_channel_prob")
    mask_channel_selection: Optional[str] = II("model.mask_channel_selection")
    mask_channel_other: Optional[float] = II("model.mask_channel_other")
    no_mask_channel_overlap: Optional[bool] = II("model.no_mask_channel_overlap")
    mask_channel_min_space: Optional[int] = II("model.mask_channel_min_space")

    conv_feature_layers: Optional[str] = II("model.conv_feature_layers")
    encoder_embed_dim: Optional[int] = II("model.encoder_embed_dim")

    tpu: bool = II("common.tpu")


@register_task("audio_pretraining", dataclass=AudioPretrainingConfig)
class AudioPretrainingTask(FairseqTask):
    """"""

    cfg: AudioPretrainingConfig

    def __init__(
        self,
        cfg: AudioPretrainingConfig,
    ):
        super().__init__(cfg)
        if cfg.eval_wer:
            assert cfg.labels is not None, "eval_wer can only be set during fine-tuning"
        self.blank_symbol = "<s>"

        self.state.add_factory("target_dictionary", self.load_target_dictionary)

    @classmethod
    def setup_task(cls, cfg: AudioPretrainingConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        return cls(cfg)

    def load_target_dictionary(self):
        if self.cfg.labels:
            if self.cfg.hf_tokenizer is None:
                dict_path = os.path.join(self.cfg.data, f"dict.{self.cfg.labels}.txt")
                target_dictionary = Dictionary.load(dict_path)
            else:
                target_dictionary = HF2FairSeqDictWrapper(self.cfg.hf_tokenizer)
        else:
            target_dictionary = None
        
        return target_dictionary

    def _get_mask_precompute_kwargs(self, cfg):
        if self.cfg.precompute_mask_indices or self.cfg.tpu:
            args = [
                "mask_length",
                "mask_prob",
                "mask_selection",
                "mask_other",
                "no_mask_overlap",
                "mask_min_space",
                "mask_channel_length",
                "mask_channel_prob",
                "mask_channel_selection",
                "mask_channel_other",
                "no_mask_channel_overlap",
                "mask_channel_min_space",
                "encoder_embed_dim",
                "conv_feature_layers",
            ]
            return {arg: cfg[arg] for arg in args}
        else:
            return {}

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        # upgrade old task
        if isinstance(task_cfg, Namespace):
            if not hasattr(task_cfg, "autoregressive"):
                task_cfg.autoregressive = not task_cfg.criterion == "ctc"

        if getattr(task_cfg, 'binarized_dataset', False):
            self.datasets[split] = BinarizedAudioDataset(
                data_path,
                split=split,
                sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                pad=task_cfg.labels is not None or task_cfg.enable_padding,
                normalize=task_cfg.normalize,
                num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                compute_mask_indices=(self.cfg.precompute_mask_indices or self.cfg.tpu),
                **self._get_mask_precompute_kwargs(task_cfg),
            )
        else:
            old_fmt = '.tsv' in [os.path.splitext(os.path.basename(x))[-1] for x in os.listdir(data_path)]
            
            if old_fmt:
                manifest_path = os.path.join(data_path, "{}.tsv".format(split))
                self.datasets[split] = FileAudioDataset(
                    manifest_path=manifest_path,
                    sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                    max_sample_size=self.cfg.max_sample_size,
                    min_sample_size=self.cfg.min_sample_size,
                    pad=task_cfg.labels is not None or task_cfg.enable_padding,
                    normalize=task_cfg.normalize,
                    num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                    compute_mask_indices=(self.cfg.precompute_mask_indices or self.cfg.tpu),
                    **self._get_mask_precompute_kwargs(task_cfg),
                )
            else:
                manifest_db = os.path.join(data_path, f"db_shelve_{split}")
                self.datasets[split] = FileAudioDatasetShelve(
                    db_name=manifest_db,
                    sample_rate=task_cfg.get('sample_rate', self.cfg.sample_rate),
                    max_sample_size=self.cfg.max_sample_size,
                    min_sample_size=self.cfg.min_sample_size,
                    max_sample_length=self.cfg.max_sample_length,
                    pad=task_cfg.labels is not None or task_cfg.enable_padding,
                    normalize=task_cfg.normalize,
                    num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                    compute_mask_indices=(self.cfg.precompute_mask_indices or self.cfg.tpu),
                    **self._get_mask_precompute_kwargs(task_cfg),
                )

        if self.cfg.tpu and task_cfg["mask_channel_prob"] == 0.0:
            logger.info(
                "Pretraining on TPUs may suffer convergence "
                "issues when training with `mask_channel_prob` value of "
                "0. You may want to set this to a low value close to 0."
            )

        if task_cfg.labels:
            if old_fmt:
                label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
                skipped_indices = getattr(self.datasets[split], 'skipped_indices', set())
                with open(label_path, "r") as f:
                    labels = [
                        line
                        for i, line in enumerate(f)
                        if i not in skipped_indices
                    ]
            else:
                label_path = f"labels_{task_cfg.labels}"
                labels = getattr(self.datasets[split], label_path)

            assert len(labels) == len(self.datasets[split]), (
                f"labels length ({len(labels)}) and dataset length "
                f"({len(self.datasets[split])}) do not match"
            )

            process_label = LabelEncoder(self.target_dictionary)

            self.datasets[split] = AddTargetDataset(
                self.datasets[split],
                labels,
                pad=self.target_dictionary.pad(),
                eos=self.target_dictionary.eos(),
                batch_targets=True,
                process_label=process_label,
                add_to_input=task_cfg.get("autoregressive", False),
            )

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.state.target_dictionary

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self,
        indices,
        dataset,
        max_positions=None,
        ignore_invalid_inputs=False,
    ):
        # we do not need to filter by size in this task as dataloaders take care of this
        return indices

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_wer and self.cfg.autoregressive:
            metrics = self._inference_with_wer(self.sequence_generator, sample, model)
            logging_output["_num_char_errors"] = metrics["num_char_errors"]
            logging_output["_num_chars"] = metrics["num_chars"]
            logging_output["_num_word_errors"] = metrics["num_word_errors"]
            logging_output["_num_words"] = metrics["num_words"]
        return loss, sample_size, logging_output

    def build_model(self, model_cfg: FairseqDataclass):
        model = super().build_model(model_cfg)

        if self.cfg.eval_wer and self.cfg.autoregressive:
            self.sequence_generator = self.build_generator(
                [model],
                self.cfg.eval_wer_config,
            )
            if self.cfg.eval_wer_tokenizer:
                self.tokenizer = encoders.build_tokenizer(self.cfg.eval_wer_tokenizer)
            else:
                self.tokenizer = None
        return model

    def _inference_with_wer(self, generator, sample, model):
        import editdistance

        def decode(toks):
            s = self.target_dictionary.string(
                toks.int().cpu(),
                self.cfg.eval_wer_post_process,
                escape_unk=True,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        num_word_errors, num_char_errors = 0, 0
        num_chars, num_words = 0, 0
        gen_out = self.inference_step(generator, [model], sample, None)
        for i in range(len(gen_out)):
            hyp = decode(gen_out[i][0]["tokens"])
            ref = decode(
                utils.strip_pad(sample["target"][i], self.target_dictionary.pad()),
            )
            num_char_errors += editdistance.eval(hyp, ref)
            num_chars += len(ref)
            hyp_words = hyp.split()
            ref_words = ref.split()
            num_word_errors += editdistance.eval(hyp_words, ref_words)
            num_words += len(ref_words)

        return {
            "num_char_errors": num_char_errors,
            "num_chars": num_chars,
            "num_word_errors": num_word_errors,
            "num_words": num_words,
        }

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        zero = torch.scalar_tensor(0.0)
        num_char_errors = sum(
            log.get("_num_char_errors", zero) for log in logging_outputs
        )
        num_chars = sum(log.get("_num_chars", zero) for log in logging_outputs)
        num_word_errors = sum(
            log.get("_num_word_errors", zero) for log in logging_outputs
        )
        num_words = sum(log.get("_num_words", zero) for log in logging_outputs)
        metrics.log_scalar("_num_char_errors", num_char_errors)
        metrics.log_scalar("_num_chars", num_chars)
        metrics.log_scalar("_num_word_errors", num_word_errors)
        metrics.log_scalar("_num_words", num_words)
        if num_words > 0:
            metrics.log_derived(
                "uer",
                lambda meters: meters["_num_char_errors"].sum
                * 100.0
                / meters["_num_chars"].sum
                if meters["_num_chars"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "wer",
                lambda meters: meters["_num_word_errors"].sum
                * 100.0
                / meters["_num_words"].sum
                if meters["_num_words"].sum > 0
                else float("nan"),
            )
