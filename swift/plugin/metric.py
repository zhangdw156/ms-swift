# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Literal

import numpy as np
import torch
import torch.distributed as dist
from transformers.trainer_utils import EvalPrediction

from swift.utils import Serializer, get_current_device, get_logger

logger = get_logger()


class Metric(ABC):

    def __init__(self):
        self._default = {}
        self._default_factory = {}

    def add_state(self, name: str, default=None, default_factory=None) -> None:
        if not hasattr(self, '_default'):
            raise AttributeError('Please call super().__init__() first.')
        if default is None:
            self._default_factory[name] = default_factory
            assert name not in self._default, f'self._default: {self._default}'
            default = default_factory()
        else:
            self._default[name] = default
            assert name not in self._default_factory, f'self._default_factory: {self._default_factory}'
        setattr(self, name, default)

    def reset(self):
        for k, v in self._default.items():
            setattr(self, k, v)
        for k, v in self._default_factory.items():
            setattr(self, k, v())

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute(self):
        pass


class InferStats(Metric):

    def __init__(self):
        super().__init__()
        self.add_state('start_runtime', default_factory=lambda: time.perf_counter())
        self.add_state('num_prompt_tokens', default_factory=dict)
        self.add_state('num_generated_tokens', default_factory=dict)

    def update(self, output):
        id_ = output.id
        self.num_prompt_tokens[id_] = output.usage.prompt_tokens
        self.num_generated_tokens[id_] = output.usage.completion_tokens

    def compute(self):
        runtime = time.perf_counter() - self.start_runtime
        num_samples = len(self.num_generated_tokens)
        num_generated_tokens = sum(self.num_generated_tokens.values())
        return {
            'num_prompt_tokens': sum(self.num_prompt_tokens.values()),
            'num_generated_tokens': num_generated_tokens,
            'num_samples': num_samples,
            'runtime': runtime,
            'samples/s': num_samples / runtime,
            'tokens/s': num_generated_tokens / runtime,
        }


class MeanMetric(Metric):

    def __init__(self, nan_value=0, device=None, group=None):
        super().__init__()
        self.nan_value = nan_value
        self.add_state('state', default=0.)
        self.add_state('count', default=0)
        if device is None:
            device = get_current_device()
        self.device = device
        self.group = group

    def update(self, state: torch.Tensor):
        if isinstance(state, (torch.Tensor, np.ndarray)):
            if state.ndim == 0:
                count = 1
                state = state.item()
            else:
                count = state.shape[0]
                state = state.sum().item()
        elif isinstance(state, (list, tuple)):
            count = len(state)
            state = sum(state)
        else:
            count = 1

        self.state += state
        self.count += count

    def compute(self):
        if dist.is_initialized():
            tensor = torch.tensor([self.state, self.count], device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
            self.state, self.count = tensor[0].item(), int(tensor[1].item())
        if self.count == 0:
            value = self.nan_value
        else:
            value = self.state / self.count
        return {
            'value': value,
        }


def compute_rouge_bleu(preds: List[str], labels: List[str]):
    import jieba
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge.rouge import Rouge
    score_dict = {key: MeanMetric() for key in ['rouge-1', 'rouge-2', 'rouge-l', 'bleu-4']}

    for pred, label in zip(preds, labels):
        hypothesis = [w.strip(' ') for w in jieba.cut(pred) if w.strip(' ')]
        reference = [w.strip(' ') for w in jieba.cut(label) if w.strip(' ')]
        if not hypothesis or not reference:
            continue
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))[0]
        for k, v in scores.items():
            score_dict[k].update(v['f'])
        bleu_score = sentence_bleu([reference], hypothesis, smoothing_function=SmoothingFunction().method3)
        score_dict['bleu-4'].update(bleu_score)

    return {k: round(v.compute()['value'] * 100, 6) for k, v in score_dict.items()}


def compute_nlg_metrics(prediction) -> Dict[str, float]:
    preds, labels = prediction[0], prediction[1]
    new_preds, new_labels = [], []
    for i in range(preds.shape[0]):
        new_preds.append(Serializer.from_tensor(preds[i]))
        new_labels.append(Serializer.from_tensor(labels[i]))
    return compute_rouge_bleu(new_preds, new_labels)


def compute_acc(preds,
                labels,
                *,
                acc_strategy: Literal['token', 'seq'] = 'token',
                is_encoder_decoder: bool = False,
                cu_seqlens=None) -> Dict[str, List[float]]:

    if isinstance(preds, torch.Tensor):
        if torch.is_floating_point(labels):
            return {}
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
    if preds.ndim >= 2 and not is_encoder_decoder:
        labels = labels[..., 1:]
        preds = preds[..., :-1]
    if np.issubdtype(labels.dtype, np.floating) or preds.shape != labels.shape:
        return {}

    masks = labels != -100
    if acc_strategy == 'token' or preds.ndim == 1:  # 'single_label_classification'
        acc_list = (preds[masks] == labels[masks]).tolist()
    else:
        acc_list = []
        if cu_seqlens is not None and masks.shape[0] == 1:
            # padding_free
            for i in range(cu_seqlens.shape[0] - 1):
                start, end = cu_seqlens[i], cu_seqlens[i + 1]
                acc_list.append(np.all(preds[0, start:end] == labels[0, start:end]))
        else:
            for i, m in enumerate(masks):
                acc_list.append(np.all(preds[i, m] == labels[i, m]))
    return {f'{acc_strategy}_acc' if preds.ndim >= 2 else 'acc': acc_list}


def compute_acc_metrics(eval_prediction: EvalPrediction,
                        *,
                        acc_strategy: Literal['token', 'seq'] = 'token',
                        is_encoder_decoder: bool = False) -> Dict[str, float]:

    metric = compute_acc(
        eval_prediction.predictions,
        eval_prediction.label_ids,
        acc_strategy=acc_strategy,
        is_encoder_decoder=is_encoder_decoder)
    if len(metric) == 0:
        return {}
    return {k: sum(v) / len(v) for k, v in metric.items()}


def preprocess_logits_for_acc(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    preds = logits.argmax(dim=-1)
    return preds


# Tokenizer for hardtry_tool_call metric (set by mixin when metric is used)
_hardtry_tokenizer = None


def set_hardtry_tokenizer(tokenizer):
    """Set tokenizer for hardtry_tool_call metric. Called by SwiftMixin when metric='hardtry_tool_call'."""
    global _hardtry_tokenizer
    _hardtry_tokenizer = tokenizer


def compute_hardtry_tool_call_metrics(eval_prediction: EvalPrediction) -> Dict[str, float]:
    """
    Compute tool_call score using hardtry RL reward (same as VeRL eval).
    Requires: hardtry on PYTHONPATH and metric tokenizer set by trainer.
    Returns metrics that will appear in eval logs and SwanLab (e.g. eval_tool_call_score).

    Reward module can be selected via env: HARDTRY_REWARD_MODULE=reward_fn_grpo (default),
    reward_fn_egpo (for <think>...</think> + tool_call), or reward_fn (format+correctness split).
    """
    global _hardtry_tokenizer
    module_name = os.environ.get('HARDTRY_REWARD_MODULE', 'reward_fn_grpo')
    try:
        from importlib import import_module
        mod = import_module(f'hardtry.rl.{module_name}')
        compute_score = mod.compute_score
    except ImportError:
        logger.warning(
            f'hardtry_tool_call metric: cannot import hardtry.rl.{module_name}. '
            'Add hardtry to PYTHONPATH or install it. Skipping metric.'
        )
        return {}

    tokenizer = _hardtry_tokenizer
    if tokenizer is None:
        logger.warning(
            'hardtry_tool_call metric: tokenizer not set. '
            'Ensure SwiftMixin sets it when metric=hardtry_tool_call. Skipping metric.'
        )
        return {}

    preds = eval_prediction.predictions
    labels = eval_prediction.label_ids
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    if preds.ndim < 2 or labels.ndim < 2 or preds.shape != labels.shape:
        return {}

    scores = []
    for i in range(preds.shape[0]):
        label_row = labels[i]
        pred_row = preds[i]
        # Only decode response part (where label != -100)
        mask = label_row != -100
        if not np.any(mask):
            continue
        pred_ids = pred_row[mask]
        label_ids = label_row[mask]
        try:
            solution_str = tokenizer.decode(pred_ids, skip_special_tokens=True)
            ground_truth = tokenizer.decode(label_ids, skip_special_tokens=True)
        except Exception:
            continue
        score = compute_score(None, solution_str, ground_truth, None)
        scores.append(score)

    if not scores:
        return {}
    return {'tool_call_score': float(np.mean(scores))}


# Add your own metric calculation method here, use --metric xxx to train
metric_mapping = {
    'acc': (compute_acc_metrics, preprocess_logits_for_acc),
    'nlg': (compute_nlg_metrics, None),
    'hardtry_tool_call': (compute_hardtry_tool_call_metrics, preprocess_logits_for_acc),
}


def get_metric(metric: str):
    if metric not in metric_mapping:
        raise KeyError(
            f'Unknown metric: {metric!r}. Available: {list(metric_mapping.keys())}. '
            'Register custom metrics in metric_mapping or via external_plugins.'
        )
    return metric_mapping[metric]
