import logging
from typing import Any, Dict, List
import numpy as np
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.functional import nll_loss
import os
import random
import traceback
import json

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, util
from allennlp.tools import squad_eval
from allennlp.training.metrics import Average, BooleanAccuracy, CategoricalAccuracy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

@Model.register("BERT_QA")
class BERT_QA(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 initializer: InitializerApplicator,
                 dropout: float = 0.2,
                 max_span_length: int = 30,
                 predictions_file = None,
                 use_multi_label_loss: bool = False,
                 stats_report_freq:float = None,
                 debug_experiment_name:str = None) -> None:
        super().__init__(vocab)
        self._max_span_length = max_span_length
        self._text_field_embedder = text_field_embedder
        self._stats_report_freq = stats_report_freq
        self._debug_experiment_name = debug_experiment_name
        self._use_multi_label_loss = use_multi_label_loss
        self._predictions_file = predictions_file

        # TODO move to predict
        if predictions_file is not None and os.path.isfile(predictions_file):
            os.remove(predictions_file)

        # see usage below for explanation
        self._all_qa_count = 0
        self._qas_used_fraction = 1.0
        self.qa_outputs = torch.nn.Linear(self._text_field_embedder.get_output_dim(), 2)

        initializer(self)

        self._official_f1 = Average()
        self._official_EM = Average()

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        batch_size, num_of_passage_tokens = passage['bert'].size()

        # BERT for QA is a fully connected linear layer on top of BERT producing 2 vectors of
        # start and end spans.
        embedded_passage = self._text_field_embedder(passage)
        passage_length = embedded_passage.size(1)
        logits = self.qa_outputs(embedded_passage)
        start_logits, end_logits = logits.split(1, dim=-1)
        span_start_logits = start_logits.squeeze(-1)
        span_end_logits = end_logits.squeeze(-1)

        # Adding some masks with numerically stable values
        passage_mask = util.get_text_field_mask(passage).float()
        repeated_passage_mask = passage_mask.unsqueeze(1).repeat(1, 1, 1)
        repeated_passage_mask = repeated_passage_mask.view(batch_size, passage_length)
        span_start_logits = util.replace_masked_values(span_start_logits, repeated_passage_mask, -1e7)
        span_end_logits = util.replace_masked_values(span_end_logits, repeated_passage_mask, -1e7)

        output_dict: Dict[str, Any] = {}

        # We may have multiple instances per questions, moving to per-question
        intances_question_id = [insta_meta['question_id'] for insta_meta in metadata]
        question_instances_split_inds = np.cumsum(np.unique(intances_question_id, return_counts=True)[1])[:-1]
        per_question_inds = np.split(range(batch_size), question_instances_split_inds)
        metadata = np.split(metadata, question_instances_split_inds)

        # Compute F1 and preparing the output dictionary.
        output_dict['answers'] = [[] for i in range(batch_size)]
        output_dict['qid'] = [[] for i in range(batch_size)]
        output_dict['scores'] = [[] for i in range(batch_size)]

        # getting best span prediction for
        span_start_logits_numpy = span_start_logits.data.cpu().numpy()
        span_end_logits_numpy = span_end_logits.data.cpu().numpy()

        for i in range(batch_size):
            # Normalize logits and spans to retrieve the answer
            start_ = span_start_logits_numpy[i]
            start_ = np.exp(start_ - start_.max(axis=-1, keepdims=True))
            start_ = start_ / start_.sum()

            end_ = span_end_logits_numpy[i]
            end_ = np.exp(end_ - end_.max(axis=-1, keepdims=True))
            end_ = end_ / end_.sum()

            # Mask CLS
            start_[0] = end_[0] = 0.0

            starts, ends, scores = self._decode(start_, end_, 10, 30)
            # Iterating over every qu.shapeestion (which may contain multiple instances, one per chunk)
            for start, end, score in zip(starts, ends, scores):

                passage_str = metadata[0][i]['original_passage']
                offsets = metadata[0][i]['token_offsets']

                start_offset = offsets[start][0]
                end_offset = offsets[end][1]
                best_span_string = passage_str[start_offset:end_offset]


                output_dict['answers'][0].append(best_span_string)
                output_dict['qid'][i] = [metadata[0][i]['question_id']]
                output_dict['scores'][0].append(score)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'EM': self._official_EM.get_metric(reset),
                'f1': self._official_f1.get_metric(reset),
                'qas_used_fraction': 1.0}


    @staticmethod
    def _get_example_predications(span_start_logits: torch.Tensor,
                                      span_end_logits: torch.Tensor,
                                      max_span_length: int) -> torch.Tensor:
        # Returns the index of highest-scoring span that is not longer than 30 tokens, as well as
        # yesno prediction bit and followup prediction bit from the predicted span end token.
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        max_span_log_prob = [-1e20] * batch_size
        span_start_argmax = [0] * batch_size

        best_word_span = span_start_logits.new_zeros((batch_size, 4), dtype=torch.long)

        span_start_logits = span_start_logits.data.cpu().numpy()
        span_end_logits = span_end_logits.data.cpu().numpy()
        for b_i in range(batch_size):  # pylint: disable=invalid-name
            for j in range(passage_length):
                val1 = span_start_logits[b_i, span_start_argmax[b_i]]
                if val1 < span_start_logits[b_i, j]:
                    span_start_argmax[b_i] = j
                    val1 = span_start_logits[b_i, j]
                val2 = span_end_logits[b_i, j]
                if val1 + val2 > max_span_log_prob[b_i]:
                    if j - span_start_argmax[b_i] > max_span_length:
                        continue
                    best_word_span[b_i, 0] = span_start_argmax[b_i]
                    best_word_span[b_i, 1] = j
                    max_span_log_prob[b_i] = val1 + val2
        for b_i in range(batch_size):
            j = best_word_span[b_i, 1]

        return best_word_span

    def _decode(
            self, start: np.ndarray, end: np.ndarray, topk: int, max_answer_len: int
        ):
            """
            Take the output of any `ModelForQuestionAnswering` and will generate probabilities for each span to be the
            actual answer.
            In addition, it filters out some unwanted/impossible cases like answer len being greater than max_answer_len or
            answer end position being before the starting position. The method supports output the k-best answer through
            the topk argument.
            Args:
                start (`np.ndarray`): Individual start probabilities for each token.
                end (`np.ndarray`): Individual end probabilities for each token.
                topk (`int`): Indicates how many possible answer span(s) to extract from the model output.
                max_answer_len (`int`): Maximum size of the answer to extract from the model's output.
                undesired_tokens (`np.ndarray`): Mask determining tokens that can be part of the answer
            """
            # Ensure we have batch axis
            if start.ndim == 1:
                start = start[None]

            if end.ndim == 1:
                end = end[None]

            # Compute the score of each tuple(start, end) to be the real answer
            outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

            # Remove candidate with end < start and end - start > max_answer_len
            candidates = np.tril(np.triu(outer), max_answer_len - 1)

            #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
            scores_flat = candidates.flatten()
            if topk == 1:
                idx_sort = [np.argmax(scores_flat)]
            elif len(scores_flat) < topk:
                idx_sort = np.argsort(-scores_flat)
            else:
                idx = np.argpartition(-scores_flat, topk)[0:topk]
                idx_sort = idx[np.argsort(-scores_flat[idx])]

            starts, ends = np.unravel_index(idx_sort, candidates.shape)[1:]
            scores = candidates[0, starts, ends]
            return starts, ends, scores
