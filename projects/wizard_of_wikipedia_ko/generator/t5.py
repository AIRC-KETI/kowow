#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

See <https://arxiv.org/abs/1910.10683>

The T5 agent can be instantiated as simply `-m t5`
"""
import torch
from typing import Optional, Dict, Any, Tuple, Union
from transformers import T5ForConditionalGeneration, T5EncoderModel

try:
    from transformers.models.t5.modeling_t5 import T5Stack
except ModuleNotFoundError:
    # Prior versions of transformers package do not have T5Stack
    T5Stack = object

from parlai.agents.hugging_face.hugging_face import HF_VERSION
from parlai.agents.hugging_face.dict import T5DictionaryAgent

from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import Batch, TorchAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel


def build_t5(opt: Opt) -> T5ForConditionalGeneration:
    if not HF_VERSION >= 4.3:
        raise RuntimeError('Must use transformers package >= 4.3 to use t5')
    return T5ForConditionalGeneration.from_pretrained(
        opt['t5_model_arch'], dropout_rate=opt['t5_dropout']
    )

def build_t5_encoder(opt: Opt) -> T5EncoderModel:
    if not HF_VERSION >= 4.3:
        raise RuntimeError('Must use transformers package >= 4.3 to use t5')
    return T5EncoderModel.from_pretrained(
        opt['t5_encoder_model_arch'], dropout_rate=opt['t5_dropout']
    )


def set_device(func):
    """
    Decorator for setting device.

    HF's model parallel uses `torch.cuda.set_device`, which does not vibe well with
    ParlAI.
    """

    def wrap(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.set_device('cuda:0')
        ret = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.set_device('cuda:0')
        return ret

    return wrap


class T5Agent(TorchGeneratorAgent):
    """
    T5 Agent.

    Relies on the T5 model implemented in huggingface
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        group = parser.add_argument_group('T5 Args')
        group.add_argument(
            '--t5-model-arch',
            type=str,
            default='t5-base',
        )
        group.add_argument(
            '--t5-model-parallel',
            type='bool',
            default=False,
            help='use HF model parallel',
        )
        group.add_argument(
            '--t5-dropout', type=float, default=0.0, help='Dropout for T5'
        )
        group.add_argument(
            '--t5-generation-config',
            type=str,
            default=None,
            choices=[
                'summarization',
                'translation_en_to_de',
                'translation_en_to_fr',
                'translation_en_to_ro',
            ],
            help='Task specific generation config for T5',
        )
        return parser

    def build_model(self) -> 'ParlaiT5Model':
        """
        Build and return model.
        """
        model = ParlaiT5Model(self.opt, self.dict)
        if self.opt['t5_model_parallel']:
            model.t5.parallelize()
        return model

    def build_dictionary(self):
        """
        Overrides TorchAgent.build_dictionary to use t5 dict.
        """
        return T5DictionaryAgent(self.opt)

    def vectorize(self, *args, **kwargs):
        """
        Override vectorize for T5.

        T5 dict already adds the end token.
        """
        kwargs['add_start'] = False  # model does this in module code
        kwargs['add_end'] = False  # T5 tokenizer takes care of this
        return TorchAgent.vectorize(self, *args, **kwargs)

    def observe(self, observation):
        """
        Override to include prefix, if necessary.
        """
        if self.opt['t5_generation_config'] is not None and 'text' in observation:
            config = TASK_CONFIGS[self.opt['t5_generation_config']]
            try:
                observation.force_set('text', config['prefix'] + observation['text'])
            except AttributeError:
                observation['text'] = config['prefix'] + observation['text']

        return super().observe(observation)

    def _generate(
        self,
        batch: Batch,
        beam_size: int,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        Generate an output with beam search.

        Use HF's built-in generation to perform beam search.
        """
        bad_words_ids = None
        if self.beam_block_list is not None:
            bad_words_ids = [
                gram for _, ngram in self.beam_block_list.items() for gram in ngram
            ]

        method = self.opt.get('inference', 'greedy')

        generation_params = {
            'input_ids': batch.text_vec,
            'max_length': max_ts,
            'min_length': self.beam_min_length,
            'do_sample': self.opt['inference'] in ['topk', 'topp'],
            'early_stopping': None,
            'num_beams': beam_size,
            'temperature': self.temperature,
            'top_k': self.opt['topk'] if method in ['topk', 'delayedbeam'] else None,
            'top_p': self.opt['topp'] if method == 'nucleus' else None,
            'repetition_penalty': None,
            'bad_words_ids': bad_words_ids if bad_words_ids else None,
            'bos_token_id': self.START_IDX,
            'pad_token_id': self.NULL_IDX,
            'eos_token_id': self.END_IDX,
            'length_penalty': self.opt['beam_length_penalty'],
            'no_repeat_ngram_size': self.beam_block_ngram,
            'num_return_sequences': None,
            'attention_mask': batch.text_vec != self.NULL_IDX,
            'decoder_start_token_id': self.NULL_IDX,
        }

        if self.opt['t5_generation_config']:
            config = TASK_CONFIGS[self.opt['t5_generation_config']]
            config.pop('prefix', None)
            generation_params.update(config)
        if overrides:
            generation_params.update(overrides)

        outputs = self.model.t5.generate(**generation_params)
        outputs = [(outputs[i], 0) for i in range(outputs.size(0))]
        return outputs, []


##############
# T5 Modules #
##############


class ParlaiT5Encoder(torch.nn.Module):
    def __init__(self, opt: Opt, encoder: T5Stack, dictionary: T5DictionaryAgent):
        super().__init__()
        self.stack = encoder
        self.padding_idx = dictionary[dictionary.null_token]
        self.paralleled = not opt[
            't5_model_parallel'
        ]  # need to parallel in forward; bug in HF

    @set_device
    def forward(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The input IDs
        :param LongTensor[batch,seqlen] positions:
            Positions for input IDs
        :param LongTensor[batch,seqlen] segments:
            If provided, additionally adds ``segments`` as extra embedding features.
        """
        if not self.paralleled:
            self.stack.parallelize()
        mask = input != self.padding_idx
        outputs = self.stack(input, attention_mask=mask, output_hidden_states=False)
        for k in outputs:
            if torch.is_tensor(outputs[k]):
                outputs[k] = outputs[k].to(input.device)
        return outputs[0], mask


class ParlaiT5Decoder(torch.nn.Module):
    def __init__(self, opt: Opt, decoder: T5Stack, dictionary: T5DictionaryAgent):
        super().__init__()
        self.stack = decoder
        self.padding_idx = dictionary[dictionary.null_token]
        self.paralleled = not opt[
            't5_model_parallel'
        ]  # need to parallel in forward; bug in HF

    @set_device
    def forward(
        self, input: torch.LongTensor, encoder_state: Tuple[Any], incr_state=None
    ):
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The decoder inputs (partial or full decoded token IDs).
        :param encoder_state:
            Output from the encoder module forward pass.
        :param incr_state:
            The incremental state: a dictionary whose keys index the layers and whose
            values contain the incremental state for each layer.
        """
        if not self.paralleled:
            self.stack.parallelize()
        encoder_output, encoder_mask = encoder_state[0], encoder_state[1]

        mask = input != self.padding_idx
        mask[:, 0] = True  # first token is pad

        outputs = self.stack(
            input_ids=input,
            attention_mask=mask,
            encoder_hidden_states=encoder_output.to(input.device),
            encoder_attention_mask=encoder_mask.to(input.device),
        )
        return outputs[0].to(input.device), incr_state


class ParlaiT5Model(TorchGeneratorModel):
    """
    Wrap T5 in ParlAI.
    """

    def __init__(self, opt, dictionary):
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = self.pad_idx
        self.end_idx = dictionary[dictionary.end_token]
        super().__init__(self.pad_idx, self.start_idx, self.end_idx)
        self.t5 = build_t5(opt)
        self.encoder = ParlaiT5Encoder(opt, self.t5.get_encoder(), dictionary)
        self.decoder = ParlaiT5Decoder(opt, self.t5.get_decoder(), dictionary)

    @set_device
    def _get_initial_forced_decoder_input(self, bsz: int, inputs: torch.LongTensor):
        """
        Return initial input to the decoder.

        :param bsz:
            batchsize
        :param inputs:
            inputs to decode

        :return initial_input:
            initial input for the decoder.
        """
        inputs = torch.cat([self.START.detach().expand(bsz, 1), inputs], 1)
        return inputs

    @set_device
    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states.

        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(
        self, incremental_state: Dict[int, dict], inds: torch.Tensor
    ) -> Dict[int, dict]:
        """
        Not *quite* sure how to reconcile this with HF.
        """
        return {}

    @set_device
    def output(self, tensor):
        """
        Compute output logits.
        """
        # Taken directly from HuggingFace
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        tensor = tensor * (self.t5.model_dim ** -0.5)
        lm_logits = self.t5.lm_head(tensor)
        return lm_logits


###########################################################################
# Task-specific generation configs for T5.                                #
# Taken from HF: https://huggingface.co/t5-small/resolve/main/config.json #
###########################################################################

TASK_CONFIGS = {
    "summarization": {
        "early_stopping": True,
        "length_penalty": 2.0,
        "max_length": 200,
        "min_length": 30,
        "no_repeat_ngram_size": 3,
        "num_beams": 4,
        "prefix": "summarize: ",
    },
    "translation_en_to_de": {
        "early_stopping": True,
        "max_length": 300,
        "num_beams": 4,
        "prefix": "translate English to German: ",
    },
    "translation_en_to_fr": {
        "early_stopping": True,
        "max_length": 300,
        "num_beams": 4,
        "prefix": "translate English to French: ",
    },
    "translation_en_to_ro": {
        "early_stopping": True,
        "max_length": 300,
        "num_beams": 4,
        "prefix": "translate English to Romanian: ",
    },
}










## TorchGeneratorModel
# _get_initial_forced_decoder_input(self, bsz: int, inputs: torch.LongTensor)
# decode_forced(self, encoder_states, ys)
# reorder_encoder_states(self, encoder_states, indices) - abstractmethod
# reorder_decoder_incremental_state(self, incremental_state, inds) - abstractmethod
# forward(self, *xs, ys=None, prev_enc=None, maxlen=None, bsz=None)
## forward: encoder_states = prev_enc if prev_enc is not None else self.encoder(*xs) -> scores, preds = self.decode_forced(encoder_states, ys) -> return scores, preds, encoder_states

## TorchGeneratorAgent
# def upgrade_opt(cls, opt_from_disk: Opt) - classmethod
# def add_cmdline_args - classmethod
# def build_criterion(self)
# def _v2t(self, vec)
# def set_interactive_mode(self, mode, shared=False)
# def _dummy_batch(self, batchsize, maxlen)
# def _init_cuda_buffer(self, batchsize, maxlen, force=False)
# def reset_metrics(self)
# def share(self)
# def vectorize(self, *args, **kwargs)
# def batchify(self, obs_batch, sort=True)
# def _model_input(self, batch)
# def _encoder_input(self, batch)
# def compute_loss(self, batch, return_output=False)
# def train_step(self, batch)
# def _construct_token_losses(self, labels, model_output)
# def _compute_fairseq_bleu(self, batch: Batch, preds)
# def _add_generation_metrics(self, batch, preds)
# def rank_eval_label_candidates(self, batch, batchsize)
# def eval_step(self, batch)
# def _treesearch_factory(self, device)
# def _get_context(self, batch, batch_idx)
# def _get_initial_decoder_input
# def _get_next_decoder_input
# def _generate
# def _load_beam_block_list(self) -> SearchBlocklist
## train_step: loss=compute_loss(batch) -> back(loss) -> update_params()
## eval_step: beam_preds_scores, beams = self._generate(batch, self.beam_size, maxlen) -> preds, scores = zip(*beam_preds_scores) -> ... -> retval
## compute_loss: model_output = self.model(*self._model_input(batch), ys=batch.label_vec) -> scores, preds, *_ = model_output -> return (loss, model_output) if return_output==True else return loss

## TorchAgent
# def optim_opts(cls) - classmethod
# def dictionary_class() - classmethod
# def history_class(cls) - classmethod
# def add_cmdline_args - classmethod
# def build_history(self)
# def build_dictionary(self)
# def _resize_token_embeddings(self, state_dict, msg=None)
# def _get_init_model(self, opt: Opt, shared)
# def _get_special_tokens(self) -> List[str]
# def build_model(self) - abstractmethod
# def _should_initialize_optimizer(self) -> bool
# def init_optim(self, params, optim_states=None, saved_optim_type=None) -> bool
# def build_lr_scheduler(self, states=None, hard_reset=False)
# def _control_local_metrics(self, enabled: bool = False, disabled: bool = False)
# def record_local_metric(self, keyname: str, values: List[Metric])
# def report(self)
# def _gpu_usage(self)
# def receive_metrics(self, metrics_dict)
# def _get_embtype(self, emb_type)
# def _project_vec(self, vec, target_dim, method='random')
# def _copy_embeddings(self, weight, emb_type, log=True)
# def share(self)
# def _add_start_end_tokens(self, vec, add_start=False, add_end=False)
# def _v2t(self, vec)
# def _vectorize_text_with_truncate_stats
# def _vectorize_text
# def _check_truncate(self, vec, truncate, truncate_left=False)
# def _set_text_vec(self, obs, history, truncate)
# def _set_label_vec(self, obs, add_start, add_end, truncate)
# def _set_label_cands_vec(self, obs, add_start, add_end, truncate)
# def vectorize( self, obs, history, add_start=True, add_end=True, text_truncate=None, label_truncate=None,)
# def _pad_tensor
# def is_valid(self, obs)
# def batchify(self, obs_batch, sort=False)
# def match_batch(self, batch_reply, valid_inds, output=None)
# def get_temp_history(self, observation) -> Optional[str]
# def observe(self, observation)
# def self_observe(self, self_message: Message) -> None
# def _validate_observe_invariants(self)
# def _validate_self_observe_invariants(self)
# def state_dict(self)
# def save(self, path=None)
# def load_state_dict(self, state_dict)
# def load(self, path: str) -> Dict[str, Any]
# def upgrade_opt(cls, opt_from_disk: Opt) - classmethod
# def reset(self)
# def reset_metrics(self)
# def act(self)
# def batch_act(self, observations)
# def train_step(self, batch) - abstractmethod
# def eval_step(self, batch) - abstractmethod
# def set_interactive_mode(self, mode, shared)
# def backward(self, loss)
# def update_params(self)
# def zero_grad(self)
## act: batch_act -> self_observe -> return response from batch_act
## batch_act: batchfy(observation) -> train_step(batch) or eval_step(batch) -> return batch_reply 

## HF t5 stack
# output: (hidden_states, present_key_value_states, all_hidden_states, all_attentions, all_cross_attentions) if it exists...


from parlai.utils.torch import neginf
from parlai.utils.misc import round_sigfigs

def universal_sentence_embedding(sentences, mask, sqrt=True):
    """
    Perform Universal Sentence Encoder averaging (https://arxiv.org/abs/1803.11175).

    This is really just sum / sqrt(len).

    :param Tensor sentences: an N x T x D of Transformer outputs. Note this is
        the exact output of TransformerEncoder, but has the time axis first
    :param ByteTensor: an N x T binary matrix of paddings

    :return: an N x D matrix of sentence embeddings
    :rtype Tensor:
    """
    # need to mask out the padded chars
    sentence_sums = torch.bmm(
        sentences.permute(0, 2, 1), mask.float().unsqueeze(-1)
    ).squeeze(-1)
    divisor = mask.sum(dim=1).view(-1, 1).float()
    if sqrt:
        divisor = divisor.sqrt()
    sentence_sums /= divisor
    return sentence_sums


class T5ContextKnowledgeEncoder(torch.nn.Module):
    def __init__(self, opt: Opt, encoder: T5Stack, dictionary: T5DictionaryAgent):
        super().__init__()
        self.stack = encoder
        self.embed_dim = encoder.config.d_model
        self.truncate = opt.get('truncate', 512)
        self.padding_idx = dictionary[dictionary.null_token]
        self.paralleled = not opt[
            't5_model_parallel'
        ]  # need to parallel in forward; bug in HF

    @set_device
    def forward(
        self, 
        src_tokens, 
        know_tokens: Optional[torch.LongTensor] = None, 
        ck_mask: Optional[torch.LongTensor] = None,
        cs_ids: Optional[torch.LongTensor] = None,
        use_cs_ids: Optional[bool] = True,
        mode: Optional[str] = 'base',
    ) -> Union[Tuple[torch.Tensor, torch.BoolTensor], Tuple[torch.Tensor, torch.BoolTensor, torch.Tensor]]:
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The input IDs
        :param LongTensor[batch,seqlen] positions:
            Positions for input IDs
        :param LongTensor[batch,seqlen] segments:
            If provided, additionally adds ``segments`` as extra embedding features.
        """
        if not self.paralleled:
            self.stack.parallelize()
        
        if mode == 'base':
            return self.base_forward(src_tokens)
        else:
            return self.context_knowledge_mips(src_tokens, know_tokens, ck_mask, cs_ids, use_cs_ids)
    
    def base_forward(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        mask = input != self.padding_idx
        outputs = self.stack(input, attention_mask=mask, output_hidden_states=False)
        for k in outputs:
            if torch.is_tensor(outputs[k]):
                outputs[k] = outputs[k].to(input.device)
        return outputs[0], mask
    
    def context_knowledge_mips(self, src_tokens, know_tokens, ck_mask, cs_ids, use_cs_ids):
        # encode the context, pretty basic
        context_encoded, context_mask = self.base_forward(src_tokens)

        # make all the knowledge into a 2D matrix to encode
        N, K, Tk = know_tokens.size()
        know_flat = know_tokens.reshape(-1, Tk)
        know_encoded, know_mask = self.base_forward(know_flat)

        # compute our sentence embeddings for context and knowledge
        context_use = universal_sentence_embedding(context_encoded, context_mask)
        know_use = universal_sentence_embedding(know_encoded, know_mask)

        # remash it back into the shape we need
        know_use = know_use.reshape(N, know_tokens.size(1), self.embed_dim)
        context_use /= np.sqrt(self.embed_dim)
        know_use /= np.sqrt(self.embed_dim)

        ck_attn = torch.bmm(know_use, context_use.unsqueeze(-1)).squeeze(-1)
        # fill with near -inf
        ck_attn.masked_fill_(~ck_mask, neginf(context_encoded.dtype))

        if not use_cs_ids:
            # if we're not given the true chosen_sentence (test time), pick our
            # best guess
            _, cs_ids = ck_attn.max(1)

        # pick the true chosen sentence. remember that TransformerEncoder outputs
        #   (batch, time, embed)
        # but because know_encoded is a flattened, it's really
        #   (N * K, T, D)
        # We need to compute the offsets of the chosen_sentences
        cs_offsets = torch.arange(N, device=cs_ids.device) * K + cs_ids

        know_tokens_selected = know_flat[cs_offsets]

        # remove eos tokens from knowledge tokens
        # know_tokens_selected = know_tokens_selected[:,:-1]

        # but padding is (N * K, T)
        cs_mask = know_mask[cs_offsets]

        # finally, concatenate it all
        full_tk = torch.cat([know_tokens_selected, src_tokens], dim=1)
        full_mask = torch.cat([cs_mask, context_mask], dim=1)

        # truncate
        full_tk = full_tk[:,:self.truncate]
        full_mask = full_mask[:,:self.truncate]

        # also return the knowledge selection mask for the loss
        return full_tk, full_mask, ck_attn


class T5ContextKnowledgeDecoder(torch.nn.Module):
    def __init__(self, opt: Opt, decoder: T5Stack, dictionary: T5DictionaryAgent):
        super().__init__()
        self.stack = decoder
        self.padding_idx = dictionary[dictionary.null_token]
        self.paralleled = not opt[
            't5_model_parallel'
        ]  # need to parallel in forward; bug in HF

    @set_device
    def forward(
        self, input: torch.LongTensor, encoder_state: Tuple[Any], incr_state=None
    ):
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The decoder inputs (partial or full decoded token IDs).
        :param encoder_state:
            Output from the encoder module forward pass.
        :param incr_state:
            The incremental state: a dictionary whose keys index the layers and whose
            values contain the incremental state for each layer.
        """
        if not self.paralleled:
            self.stack.parallelize()
        encoder_output, encoder_mask, _ = encoder_state

        mask = input != self.padding_idx
        mask[:, 0] = True  # first token is pad

        outputs = self.stack(
            input_ids=input,
            attention_mask=mask,
            encoder_hidden_states=encoder_output.to(input.device),
            encoder_attention_mask=encoder_mask.to(input.device),
        )
        return outputs[0].to(input.device), incr_state


class T5EndtoEndModel(TorchGeneratorModel):
    """
    Wrap T5 in ParlAI.
    """

    def __init__(self, opt, dictionary):
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = self.pad_idx
        self.end_idx = dictionary[dictionary.end_token]
        super().__init__(self.pad_idx, self.start_idx, self.end_idx)
        self.t5 = build_t5(opt)
        self.encoder = T5ContextKnowledgeEncoder(opt, self.t5.get_encoder(), dictionary)
        self.decoder = T5ContextKnowledgeDecoder(opt, self.t5.get_decoder(), dictionary)

    @set_device
    def _get_initial_forced_decoder_input(self, bsz: int, inputs: torch.LongTensor):
        """
        Return initial input to the decoder.

        :param bsz:
            batchsize
        :param inputs:
            inputs to decode

        :return initial_input:
            initial input for the decoder.
        """
        inputs = torch.cat([self.START.detach().expand(bsz, 1), inputs], 1)
        return inputs

    @set_device
    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states.

        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """
        enc, mask, ckattn = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        ckattn = torch.index_select(ckattn, 0, indices)
        return enc, mask, ckattn

    def reorder_decoder_incremental_state(
        self, incremental_state: Dict[int, dict], inds: torch.Tensor
    ) -> Dict[int, dict]:
        """
        Not *quite* sure how to reconcile this with HF.
        """
        return {}

    @set_device
    def output(self, tensor):
        """
        Compute output logits.
        """
        # Taken directly from HuggingFace
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        tensor = tensor * (self.t5.model_dim ** -0.5)
        lm_logits = self.t5.lm_head(tensor)
        return lm_logits
    
    def forward(self, *xs, ys=None, prev_enc=None, maxlen=None, bsz=None):
        """
        Get output predictions from the model.

        :param xs:
            input to the encoder
        :type xs:
            LongTensor[bsz, seqlen]
        :param ys:
            Expected output from the decoder. Used
            for teacher forcing to calculate loss.
        :type ys:
            LongTensor[bsz, outlen]
        :param prev_enc:
            if you know you'll pass in the same xs multiple times, you can pass
            in the encoder output from the last forward pass to skip
            recalcuating the same encoder output.
        :param maxlen:
            max number of tokens to decode. if not set, will use the length of
            the longest label this model has seen. ignored when ys is not None.
        :param bsz:
            if ys is not provided, then you must specify the bsz for greedy
            decoding.

        :return:
            (scores, candidate_scores, encoder_states) tuple

            - scores contains the model's predicted token scores.
              (FloatTensor[bsz, seqlen, num_features])
            - candidate_scores are the score the model assigned to each candidate.
              (FloatTensor[bsz, num_cands])
            - encoder_states are the output of model.encoder. Model specific types.
              Feed this back in to skip encoding on the next call.
        """
        assert ys is not None, "Greedy decoding in TGModel.forward no longer supported."
        # TODO: get rid of longest_label
        # keep track of longest label we've ever seen
        # we'll never produce longer ones than that during prediction
        self.longest_label = max(self.longest_label, ys.size(1))

        # use cached encoding if available
        if prev_enc is not None:
            encoder_states = prev_enc
        else:
            full_tk, full_mask, ck_attn = self.encoder(*xs, mode='mips')
            enc, mask = self.encoder(full_tk, mode='base')
            encoder_states = (enc, mask, ck_attn)

        # use teacher forcing
        scores, preds = self.decode_forced(encoder_states, ys)
        return scores, preds, encoder_states


TOKEN_DIALOG = '__dialog__'


DEFAULT_OPTS = {
    "learningrate": 5e-4,
    "optimizer": "adam",
    "lr_scheduler": "invsqrt",
    "warmup_updates": 5000,
    "clip_norm": 0.1,
    "betas": "0.9,0.98",
    "truncate": 256,
    "knowledge_truncate": 64,
    "text_truncate": 256,
    "include_knowledge_separator": True,
    "dict_textfields": "text,labels,chosen_topic,checked_sentence,knowledge,title",
}

from functools import lru_cache
from itertools import chain

import numpy as np
from parlai.utils.torch import padded_tensor
from parlai.tasks.wizard_of_wikipedia_ko.agents import TOKEN_KNOWLEDGE, TOKEN_END_KNOWLEDGE

# TOKEN_NOCHOSEN = 'no_passages_used'
# TOKEN_KNOWLEDGE = '__knowledge__'
# TOKEN_END_KNOWLEDGE = '__endknowledge__'

class _T5GenericWizardAgent(TorchGeneratorAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser.set_defaults(**DEFAULT_OPTS)
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        group = parser.add_argument_group('T5E2E Args')
        group.add_argument(
            '--t5-model-arch',
            type=str,
            default='t5-base',
        )
        group.add_argument(
            '--t5-model-parallel',
            type='bool',
            default=False,
            help='use HF model parallel',
        )
        group.add_argument(
            '--t5-dropout', type=float, default=0.0, help='Dropout for T5'
        )
        return parser

    def batchify(self, obs_batch):
        batch = super().batchify(obs_batch)
        reordered_observations = [obs_batch[i] for i in batch.valid_indices]

        checked_sentences = []
        for obs in reordered_observations:
            checked_sentence = '{} {} {} {} '.format(
                obs.get('title', ''), TOKEN_KNOWLEDGE, obs.get('checked_sentence', ''), TOKEN_END_KNOWLEDGE
            )
            checked_sentences.append(checked_sentence)

        batch['checked_sentence'] = checked_sentences

        return batch


class T5EndToEndAgent(_T5GenericWizardAgent):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self._vectorize_text = lru_cache(int(2 ** 20))(self._vectorize_text)

        # knowledge truncate defaults to the same as --truncate
        self.knowledge_truncate = opt.get('knowledge_truncate')
        if not self.knowledge_truncate:
            self.knowledge_truncate = opt['truncate']
        self.max_knowledge = opt.get('max_knowledge')
        self.knowledge_alpha = opt['knowledge_alpha']
        self.use_gold_knowledge = opt.get('use_gold_knowledge', False)

    def _dummy_batch(self, bsz, maxlen):
        batch = super()._dummy_batch(bsz, maxlen)
        batch['know_vec'] = torch.ones(bsz, 4, 8).long().cuda()
        # bool/uint8 backwards for pytorch 1.0/1.2 compatibility
        ck_mask = (torch.ones(bsz, 4, dtype=torch.uint8) != 0).cuda()
        batch['ck_mask'] = ck_mask
        batch['cs_ids'] = torch.zeros(bsz).long().cuda()
        batch['use_cs_ids'] = True
        return batch

    def compute_loss(self, batch, return_output=False):
        # first compute our regular forced decoding loss
        token_loss, model_output = super().compute_loss(batch, return_output=True)
        notnull = batch.label_vec.ne(self.NULL_IDX)
        num_tokens = notnull.long().sum().item()

        encoder_states = model_output[2]
        ctx_know_attn = encoder_states[2]

        if self.knowledge_alpha == 0.0:
            loss = token_loss
        else:
            _, know_pred = ctx_know_attn.max(1)
            know_acc = (know_pred == batch.cs_ids).float().sum().item()
            know_chance = batch.ck_mask.sum(1).float().reciprocal().sum().item()
            self.metrics['know_chance'] += know_chance
            self.metrics['bsz'] += batch.text_vec.size(0)
            self.metrics['know_acc'] += know_acc

            know_loss = torch.nn.functional.cross_entropy(
                ctx_know_attn, batch.cs_ids, reduction='mean'
            )

            self.metrics['know_loss'] += know_loss.item() * batch.text_vec.size(0)
            # in the original paper the loss was scaled by num_tokens for both
            # know_loss and token_loss
            know_loss /= num_tokens

            loss = (
                1 - self.knowledge_alpha
            ) * token_loss + self.knowledge_alpha * know_loss
        
        if return_output:
            return (loss, model_output)
        else:
            return loss

    def reset_metrics(self):
        super().reset_metrics()
        self.metrics['bsz'] = 0.0
        self.metrics['know_acc'] = 0.0
        self.metrics['know_loss'] = 0.0
        self.metrics['know_chance'] = 0.0

    def report(self):
        r = super().report()
        bsz = max(self.metrics['bsz'], 1)
        for k in ['know_loss', 'know_acc', 'know_chance']:
            # round and average across all items since last report
            r[k] = round_sigfigs(self.metrics[k] / bsz, 4)
        return r

    def _parse_knowledge(self, obs):
        if 'knowledge_parsed' in obs:
            # make a copy of the list to prevent the future padding step from
            # being destructive
            return list(obs['knowledge_parsed'])

        if 'checked_sentence' not in obs:
            # interactive time. we're totally on our own
            obs_know = [
                k.strip() + ' {} '.format(TOKEN_END_KNOWLEDGE) for k in obs.get('knowledge', 'no_passages_used').split('\n')
            ]
            obs_know = [k for k in obs_know if k]
            obs['knowledge_parsed'] = obs_know
            return obs['knowledge_parsed']

        checked_sentence = '{} {} {} {} '.format(
            obs['title'], TOKEN_KNOWLEDGE, obs['checked_sentence'], TOKEN_END_KNOWLEDGE
        )
        # grab all the nonempty knowledge
        obs_know = [
            k.strip() + ' {} '.format(TOKEN_END_KNOWLEDGE) for k in obs.get('knowledge', 'no_passages_used').split('\n')
        ]
        obs_know = [k for k in obs_know if k]

        # we want the correct knowledge to always be in index 0
        try:
            i = obs_know.index(checked_sentence)
        except ValueError:
            # uh oh, couldn't find the sentence in the knowledge. This happens for
            # one or two examples in the training set. We can just artificially
            # put it back in
            i = 0
            obs_know[0] = checked_sentence
        obs_know[0], obs_know[i] = obs_know[i], obs_know[0]

        obs['knowledge_parsed'] = obs_know
        obs['checked_sentence_parsed'] = checked_sentence
        return obs['knowledge_parsed']

    def batchify(self, obs_batch):
        """
        Wizard custom batchify, which passes along the knowledge/title.

        Following the docstring of TorchAgent.batchify, it calls super, then
        uses an extended version of the torch_agent.Batch namedtuple.

        The purpose of extending the info is to keep track of some custom
        metrics.
        """
        batch = super().batchify(obs_batch)
        reordered_observations = [obs_batch[i] for i in batch.valid_indices]
        is_training = 'labels' in reordered_observations[0]

        # first parse and compile all the knowledge together
        all_knowledges = []  # list-of-lists knowledge items for each observation
        knowledge_counts = []  # how much knowledge each observation gets
        for obs in reordered_observations:
            obs_know = self._parse_knowledge(obs)
            # downsample if desired
            if (
                is_training
                and self.max_knowledge
                and len(obs_know) > self.max_knowledge
            ):
                # offset by one so that we don't choose 0
                keepers = 1 + np.random.choice(
                    len(obs_know) - 1, self.max_knowledge, False
                )
                # correct answer is always the first one
                keepers[0] = 0
                obs_know = [obs_know[i] for i in keepers]
            all_knowledges.append(obs_know)
            knowledge_counts.append(len(obs_know))

        # now we want to actually pack this into a tensor, along with the mask
        N = len(reordered_observations)
        K = max(knowledge_counts)
        # round out the array so everything is equally sized
        for i in range(N):
            all_knowledges[i] += [''] * (K - knowledge_counts[i])
        flattened_knowledge = list(chain(*all_knowledges))

        knowledge_vec = [
            self._vectorize_text(
                # the beginning of the sentence is more useful
                k,
                truncate=self.knowledge_truncate,
                add_end=False,
                truncate_left=False,
            )
            for k in flattened_knowledge
        ]

        knowledge_vec, _ = padded_tensor(
            knowledge_vec, pad_idx=self.NULL_IDX, left_padded=True
        )

        knowledge_vec[:, -1] = self.END_IDX
        T = knowledge_vec.size(-1)
        knowledge_vec = knowledge_vec.view(N, K, T)


        # knowledge mask is a N x K tensor saying which items we're allowed to
        # attend over
        bsz = len(reordered_observations)
        ck_mask = torch.zeros(bsz, K, dtype=torch.uint8)
        for i, klen in enumerate(knowledge_counts):
            ck_mask[i, :klen] = 1
        ck_mask = ck_mask != 0  # for pytorch 1.0/1.2 uint8/bool compatibility
        # and the correct labels
        cs_ids = torch.LongTensor(bsz).zero_()

        if self.use_cuda:
            knowledge_vec = knowledge_vec.cuda()
            ck_mask = ck_mask.cuda()
            cs_ids = cs_ids.cuda()
        

        batch['know_vec'] = knowledge_vec
        batch['ck_mask'] = ck_mask
        batch['cs_ids'] = cs_ids
        batch['use_cs_ids'] = is_training or self.use_gold_knowledge
        batch['knowledge'] = np.array(flattened_knowledge).reshape(N, K)
        return batch

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        group = parser.add_argument_group("EndToEnd Agent")
        group.add_argument(
            '--knowledge-alpha',
            type=float,
            default=0.95,
            help='Weight on the knowledge-attn loss',
        )
        group.add_argument(
            '--knowledge-truncate',
            type=int,
            default=32,
            help='Knowledge truncation field. Defaults to same as --truncate.',
        )
        group.add_argument(
            '--max-knowledge',
            type=int,
            help='Reduce the amount of negative knowledge at train time.',
        )
        group.add_argument(
            '--use-gold-knowledge',
            type='bool',
            default=False,
            help='use gold knowledge on evaluation',
        )
        return parser

    def _model_input(self, batch):
        return (
            batch.text_vec,
            batch.know_vec,
            batch.ck_mask,
            batch.cs_ids,
            batch.use_cs_ids,
        )

    def build_model(self):
        model = T5EndtoEndModel(self.opt, self.dict)
        if self.opt['t5_model_parallel']:
            model.t5.parallelize()
        return model

        if self.use_cuda:
            self.model = self.model.cuda()
        return self.model
    
    def build_dictionary(self):
        """
        Overrides TorchAgent.build_dictionary to use t5 dict.
        """
        return T5DictionaryAgent(self.opt)

    def vectorize(self, *args, **kwargs):
        """
        Override vectorize for T5.

        T5 dict already adds the end token.
        """
        kwargs['add_start'] = False  # model does this in module code
        kwargs['add_end'] = False  # T5 tokenizer takes care of this
        return TorchAgent.vectorize(self, *args, **kwargs)

    # def observe(self, observation):
    #     """
    #     Override to include prefix, if necessary.
    #     """
    #     if self.opt['t5_generation_config'] is not None and 'text' in observation:
    #         config = TASK_CONFIGS[self.opt['t5_generation_config']]
    #         try:
    #             observation.force_set('text', config['prefix'] + observation['text'])
    #         except AttributeError:
    #             observation['text'] = config['prefix'] + observation['text']

    #     return super().observe(observation)
    
    def eval_step(self, batch):
        ret = super().eval_step(batch)

        full_tk, _, _ = self.model.encoder( batch.text_vec,
                                            batch.know_vec,
                                            batch.ck_mask,
                                            batch.cs_ids,
                                            batch.use_cs_ids,
                                            mode='mips')

        enc_output = [self._v2t(p) for p in full_tk] if full_tk is not None else None
        ret.enc_output = enc_output

        return ret
    
    def _v2t_ext(self, vec):
        """
        Convert token indices to string of tokens.
        """
        new_vec = []
        if hasattr(vec, 'cpu'):
            vec = vec.cpu()
        for i in vec:
            new_vec.append(i)
        return self.dict.vec2txt(new_vec)

    def _generate(
        self,
        batch: Batch,
        beam_size: int,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        Generate an output with beam search.

        Use HF's built-in generation to perform beam search.
        """
        bad_words_ids = None
        if self.beam_block_list is not None:
            bad_words_ids = [
                gram for _, ngram in self.beam_block_list.items() for gram in ngram
            ]

        method = self.opt.get('inference', 'greedy')

        # select knowledge (full_tk, full_mask, ck_attn)
        full_tk, _, _ = self.model.encoder( batch.text_vec,
                                            batch.know_vec,
                                            batch.ck_mask,
                                            batch.cs_ids,
                                            batch.use_cs_ids,
                                            mode='mips')

        generation_params = {
            'input_ids': full_tk,
            'max_length': max_ts,
            'min_length': self.beam_min_length,
            'do_sample': self.opt['inference'] in ['topk', 'topp'],
            'early_stopping': None,
            'num_beams': beam_size,
            'temperature': self.temperature,
            'top_k': self.opt['topk'] if method in ['topk', 'delayedbeam'] else None,
            'top_p': self.opt['topp'] if method == 'nucleus' else None,
            'repetition_penalty': None,
            'bad_words_ids': bad_words_ids if bad_words_ids else None,
            'bos_token_id': self.START_IDX,
            'pad_token_id': self.NULL_IDX,
            'eos_token_id': self.END_IDX,
            'length_penalty': self.opt['beam_length_penalty'],
            'no_repeat_ngram_size': self.beam_block_ngram,
            'num_return_sequences': None,
            'attention_mask': full_tk != self.NULL_IDX,
            'decoder_start_token_id': self.NULL_IDX,
        }


        if overrides:
            generation_params.update(overrides)

        outputs = self.model.t5.generate(**generation_params)
        outputs = [(outputs[i], 0) for i in range(outputs.size(0))]
        return outputs, []



class T5EncoderModelContextKnowledgeEncoder(torch.nn.Module):
    def __init__(self, opt: Opt, encoder: T5Stack, dictionary: T5DictionaryAgent):
        super().__init__()
        self.stack = encoder
        self.embed_dim = encoder.config.d_model
        self.truncate = opt.get('truncate', 512)
        self.padding_idx = dictionary[dictionary.null_token]
        self.paralleled = not opt[
            't5_model_parallel'
        ]  # need to parallel in forward; bug in HF
    
    def base_forward(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        mask = input != self.padding_idx
        outputs = self.stack(input, attention_mask=mask, output_hidden_states=False)
        for k in outputs:
            if torch.is_tensor(outputs[k]):
                outputs[k] = outputs[k].to(input.device)
        return outputs[0], mask

    @set_device
    def forward(
        self, 
        src_tokens, 
        know_tokens: Optional[torch.LongTensor] = None, 
        ck_mask: Optional[torch.LongTensor] = None,
        cs_ids: Optional[torch.LongTensor] = None,
        use_cs_ids: Optional[bool] = True
    ):
        # encode the context, pretty basic
        context_encoded, context_mask = self.base_forward(src_tokens)

        # make all the knowledge into a 2D matrix to encode
        N, K, Tk = know_tokens.size()
        know_flat = know_tokens.reshape(-1, Tk)
        know_encoded, know_mask = self.base_forward(know_flat)

        # compute our sentence embeddings for context and knowledge
        context_use = universal_sentence_embedding(context_encoded, context_mask)
        know_use = universal_sentence_embedding(know_encoded, know_mask)

        # remash it back into the shape we need
        know_use = know_use.reshape(N, know_tokens.size(1), self.embed_dim)
        context_use /= np.sqrt(self.embed_dim)
        know_use /= np.sqrt(self.embed_dim)

        ck_attn = torch.bmm(know_use, context_use.unsqueeze(-1)).squeeze(-1)
        # fill with near -inf
        ck_attn.masked_fill_(~ck_mask, neginf(context_encoded.dtype))

        if not use_cs_ids:
            # if we're not given the true chosen_sentence (test time), pick our
            # best guess
            _, cs_ids = ck_attn.max(1)

        # pick the true chosen sentence. remember that TransformerEncoder outputs
        #   (batch, time, embed)
        # but because know_encoded is a flattened, it's really
        #   (N * K, T, D)
        # We need to compute the offsets of the chosen_sentences
        cs_offsets = torch.arange(N, device=cs_ids.device) * K + cs_ids

        know_tokens_selected = know_flat[cs_offsets]

        # remove eos tokens from knowledge tokens
        #know_tokens_selected = know_tokens_selected[:,:-1]

        # but padding is (N * K, T)
        cs_mask = know_mask[cs_offsets]

        # finally, concatenate it all
        full_tk = torch.cat([know_tokens_selected, src_tokens], dim=1)
        full_mask = torch.cat([cs_mask, context_mask], dim=1)

        # truncate
        full_tk = full_tk[:,:self.truncate]
        full_mask = full_mask[:,:self.truncate]

        # also return the knowledge selection mask for the loss
        return full_tk, full_mask, ck_attn


class T5EndtoEndTwoModel(TorchGeneratorModel):
    """
    Wrap T5 in ParlAI.
    """

    def __init__(self, opt, dictionary):
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = self.pad_idx
        self.end_idx = dictionary[dictionary.end_token]
        super().__init__(self.pad_idx, self.start_idx, self.end_idx)
        self.t5_encoder = build_t5_encoder(opt)
        self.t5 = build_t5(opt)

        self.knowledge_encoder = T5EncoderModelContextKnowledgeEncoder(opt, self.t5_encoder.get_encoder(), dictionary)
        self.encoder = ParlaiT5Encoder(opt, self.t5.get_encoder(), dictionary)
        self.decoder = ParlaiT5Decoder(opt, self.t5.get_decoder(), dictionary)

    @set_device
    def _get_initial_forced_decoder_input(self, bsz: int, inputs: torch.LongTensor):
        """
        Return initial input to the decoder.

        :param bsz:
            batchsize
        :param inputs:
            inputs to decode

        :return initial_input:
            initial input for the decoder.
        """
        inputs = torch.cat([self.START.detach().expand(bsz, 1), inputs], 1)
        return inputs

    @set_device
    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states.

        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """
        enc, mask, ckattn = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        ckattn = torch.index_select(ckattn, 0, indices)
        return enc, mask, ckattn

    def reorder_decoder_incremental_state(
        self, incremental_state: Dict[int, dict], inds: torch.Tensor
    ) -> Dict[int, dict]:
        """
        Not *quite* sure how to reconcile this with HF.
        """
        return {}

    @set_device
    def output(self, tensor):
        """
        Compute output logits.
        """
        # Taken directly from HuggingFace
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        tensor = tensor * (self.t5.model_dim ** -0.5)
        lm_logits = self.t5.lm_head(tensor)
        return lm_logits
    
    def forward(self, *xs, ys=None, prev_enc=None, maxlen=None, bsz=None):
        """
        Get output predictions from the model.

        :param xs:
            input to the encoder
        :type xs:
            LongTensor[bsz, seqlen]
        :param ys:
            Expected output from the decoder. Used
            for teacher forcing to calculate loss.
        :type ys:
            LongTensor[bsz, outlen]
        :param prev_enc:
            if you know you'll pass in the same xs multiple times, you can pass
            in the encoder output from the last forward pass to skip
            recalcuating the same encoder output.
        :param maxlen:
            max number of tokens to decode. if not set, will use the length of
            the longest label this model has seen. ignored when ys is not None.
        :param bsz:
            if ys is not provided, then you must specify the bsz for greedy
            decoding.

        :return:
            (scores, candidate_scores, encoder_states) tuple

            - scores contains the model's predicted token scores.
              (FloatTensor[bsz, seqlen, num_features])
            - candidate_scores are the score the model assigned to each candidate.
              (FloatTensor[bsz, num_cands])
            - encoder_states are the output of model.encoder. Model specific types.
              Feed this back in to skip encoding on the next call.
        """
        assert ys is not None, "Greedy decoding in TGModel.forward no longer supported."
        # TODO: get rid of longest_label
        # keep track of longest label we've ever seen
        # we'll never produce longer ones than that during prediction
        self.longest_label = max(self.longest_label, ys.size(1))

        # use cached encoding if available
        if prev_enc is not None:
            encoder_states = prev_enc
        else:
            full_tk, full_mask, ck_attn = self.knowledge_encoder(*xs)
            enc, mask = self.encoder(full_tk)
            encoder_states = (enc, mask, ck_attn)

        # use teacher forcing
        scores, preds = self.decode_forced(encoder_states, ys)
        return scores, preds, encoder_states


class T5EndToEndTwoAgent(_T5GenericWizardAgent):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self._vectorize_text = lru_cache(int(2 ** 20))(self._vectorize_text)

        # knowledge truncate defaults to the same as --truncate
        self.knowledge_truncate = opt.get('knowledge_truncate')
        if not self.knowledge_truncate:
            self.knowledge_truncate = opt['truncate']
        self.max_knowledge = opt.get('max_knowledge')
        self.knowledge_alpha = opt['knowledge_alpha']
        self.use_gold_knowledge = opt.get('use_gold_knowledge', False)

    def _dummy_batch(self, bsz, maxlen):
        batch = super()._dummy_batch(bsz, maxlen)
        batch['know_vec'] = torch.ones(bsz, 4, 8).long().cuda()
        # bool/uint8 backwards for pytorch 1.0/1.2 compatibility
        ck_mask = (torch.ones(bsz, 4, dtype=torch.uint8) != 0).cuda()
        batch['ck_mask'] = ck_mask
        batch['cs_ids'] = torch.zeros(bsz).long().cuda()
        batch['use_cs_ids'] = True
        return batch

    def compute_loss(self, batch, return_output=False):
        # first compute our regular forced decoding loss
        token_loss, model_output = super().compute_loss(batch, return_output=True)
        notnull = batch.label_vec.ne(self.NULL_IDX)
        num_tokens = notnull.long().sum().item()

        encoder_states = model_output[2]
        ctx_know_attn = encoder_states[2]

        if self.knowledge_alpha == 0.0:
            loss = token_loss
        else:
            _, know_pred = ctx_know_attn.max(1)
            know_acc = (know_pred == batch.cs_ids).float().sum().item()
            know_chance = batch.ck_mask.sum(1).float().reciprocal().sum().item()
            self.metrics['know_chance'] += know_chance
            self.metrics['bsz'] += batch.text_vec.size(0)
            self.metrics['know_acc'] += know_acc

            know_loss = torch.nn.functional.cross_entropy(
                ctx_know_attn, batch.cs_ids, reduction='mean'
            )

            self.metrics['know_loss'] += know_loss.item() * batch.text_vec.size(0)
            # in the original paper the loss was scaled by num_tokens for both
            # know_loss and token_loss
            know_loss /= num_tokens

            loss = (
                1 - self.knowledge_alpha
            ) * token_loss + self.knowledge_alpha * know_loss
        
        if return_output:
            return (loss, model_output)
        else:
            return loss

    def reset_metrics(self):
        super().reset_metrics()
        self.metrics['bsz'] = 0.0
        self.metrics['know_acc'] = 0.0
        self.metrics['know_loss'] = 0.0
        self.metrics['know_chance'] = 0.0

    def report(self):
        r = super().report()
        bsz = max(self.metrics['bsz'], 1)
        for k in ['know_loss', 'know_acc', 'know_chance']:
            # round and average across all items since last report
            r[k] = round_sigfigs(self.metrics[k] / bsz, 4)
        return r

    def _parse_knowledge(self, obs):
        if 'knowledge_parsed' in obs:
            # make a copy of the list to prevent the future padding step from
            # being destructive
            return list(obs['knowledge_parsed'])

        if 'checked_sentence' not in obs:
            # interactive time. we're totally on our own
            obs_know = [
                k.strip() + ' {} '.format(TOKEN_END_KNOWLEDGE) for k in obs.get('knowledge', 'no_passages_used').split('\n')
            ]
            obs_know = [k for k in obs_know if k]
            obs['knowledge_parsed'] = obs_know
            return obs['knowledge_parsed']

        checked_sentence = '{} {} {} {} '.format(
            obs['title'], TOKEN_KNOWLEDGE, obs['checked_sentence'], TOKEN_END_KNOWLEDGE
        )
        # grab all the nonempty knowledge
        obs_know = [
            k.strip() + ' {} '.format(TOKEN_END_KNOWLEDGE) for k in obs.get('knowledge', 'no_passages_used').split('\n')
        ]
        obs_know = [k for k in obs_know if k]

        # we want the correct knowledge to always be in index 0
        try:
            i = obs_know.index(checked_sentence)
        except ValueError:
            # uh oh, couldn't find the sentence in the knowledge. This happens for
            # one or two examples in the training set. We can just artificially
            # put it back in
            i = 0
            obs_know[0] = checked_sentence
        obs_know[0], obs_know[i] = obs_know[i], obs_know[0]

        obs['knowledge_parsed'] = obs_know
        obs['checked_sentence_parsed'] = checked_sentence
        return obs['knowledge_parsed']

    def batchify(self, obs_batch):
        """
        Wizard custom batchify, which passes along the knowledge/title.

        Following the docstring of TorchAgent.batchify, it calls super, then
        uses an extended version of the torch_agent.Batch namedtuple.

        The purpose of extending the info is to keep track of some custom
        metrics.
        """
        batch = super().batchify(obs_batch)
        reordered_observations = [obs_batch[i] for i in batch.valid_indices]
        is_training = 'labels' in reordered_observations[0]

        # first parse and compile all the knowledge together
        all_knowledges = []  # list-of-lists knowledge items for each observation
        knowledge_counts = []  # how much knowledge each observation gets
        for obs in reordered_observations:
            obs_know = self._parse_knowledge(obs)
            # downsample if desired
            if (
                is_training
                and self.max_knowledge
                and len(obs_know) > self.max_knowledge
            ):
                # offset by one so that we don't choose 0
                keepers = 1 + np.random.choice(
                    len(obs_know) - 1, self.max_knowledge, False
                )
                # correct answer is always the first one
                keepers[0] = 0
                obs_know = [obs_know[i] for i in keepers]
            all_knowledges.append(obs_know)
            knowledge_counts.append(len(obs_know))

        # now we want to actually pack this into a tensor, along with the mask
        N = len(reordered_observations)
        K = max(knowledge_counts)
        # round out the array so everything is equally sized
        for i in range(N):
            all_knowledges[i] += [''] * (K - knowledge_counts[i])
        flattened_knowledge = list(chain(*all_knowledges))

        knowledge_vec = [
            self._vectorize_text(
                # the beginning of the sentence is more useful
                k,
                truncate=self.knowledge_truncate,
                add_end=False,
                truncate_left=False,
            )
            for k in flattened_knowledge
        ]

        knowledge_vec, _ = padded_tensor(
            knowledge_vec, pad_idx=self.NULL_IDX, left_padded=True
        )

        knowledge_vec[:, -1] = self.END_IDX
        T = knowledge_vec.size(-1)
        knowledge_vec = knowledge_vec.view(N, K, T)


        # knowledge mask is a N x K tensor saying which items we're allowed to
        # attend over
        bsz = len(reordered_observations)
        ck_mask = torch.zeros(bsz, K, dtype=torch.uint8)
        for i, klen in enumerate(knowledge_counts):
            ck_mask[i, :klen] = 1
        ck_mask = ck_mask != 0  # for pytorch 1.0/1.2 uint8/bool compatibility
        # and the correct labels
        cs_ids = torch.LongTensor(bsz).zero_()

        if self.use_cuda:
            knowledge_vec = knowledge_vec.cuda()
            ck_mask = ck_mask.cuda()
            cs_ids = cs_ids.cuda()
        

        batch['know_vec'] = knowledge_vec
        batch['ck_mask'] = ck_mask
        batch['cs_ids'] = cs_ids
        batch['use_cs_ids'] = is_training or self.use_gold_knowledge
        batch['knowledge'] = np.array(flattened_knowledge).reshape(N, K)
        return batch

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        group = parser.add_argument_group("EndToEnd Agent")
        group.add_argument(
            '--knowledge-alpha',
            type=float,
            default=0.95,
            help='Weight on the knowledge-attn loss',
        )
        group.add_argument(
            '--knowledge-truncate',
            type=int,
            default=32,
            help='Knowledge truncation field. Defaults to same as --truncate.',
        )
        group.add_argument(
            '--max-knowledge',
            type=int,
            help='Reduce the amount of negative knowledge at train time.',
        )
        group.add_argument(
            '--use-gold-knowledge',
            type='bool',
            default=False,
            help='use gold knowledge on evaluation',
        )
        group.add_argument(
            '--t5-encoder-model-arch',
            type=str,
            default='t5-small',
        )
        return parser

    def _model_input(self, batch):
        return (
            batch.text_vec,
            batch.know_vec,
            batch.ck_mask,
            batch.cs_ids,
            batch.use_cs_ids,
        )

    def build_model(self):
        model = T5EndtoEndTwoModel(self.opt, self.dict)
        if self.opt['t5_model_parallel']:
            model.t5.parallelize()
            model.t5_encoder.parallelize()
        return model
    
    def build_dictionary(self):
        """
        Overrides TorchAgent.build_dictionary to use t5 dict.
        """
        return T5DictionaryAgent(self.opt)

    def vectorize(self, *args, **kwargs):
        """
        Override vectorize for T5.

        T5 dict already adds the end token.
        """
        kwargs['add_start'] = False  # model does this in module code
        kwargs['add_end'] = False  # T5 tokenizer takes care of this
        return TorchAgent.vectorize(self, *args, **kwargs)

    # def observe(self, observation):
    #     """
    #     Override to include prefix, if necessary.
    #     """
    #     if self.opt['t5_generation_config'] is not None and 'text' in observation:
    #         config = TASK_CONFIGS[self.opt['t5_generation_config']]
    #         try:
    #             observation.force_set('text', config['prefix'] + observation['text'])
    #         except AttributeError:
    #             observation['text'] = config['prefix'] + observation['text']

    #     return super().observe(observation)

    def eval_step(self, batch):
        ret = super().eval_step(batch)

        full_tk, _, _ = self.model.knowledge_encoder( batch.text_vec,
                                            batch.know_vec,
                                            batch.ck_mask,
                                            batch.cs_ids,
                                            batch.use_cs_ids)

        enc_output = [self._v2t(p) for p in full_tk] if full_tk is not None else None
        ret.enc_output = enc_output

        return ret
    
    def _v2t_ext(self, vec):
        """
        Convert token indices to string of tokens.
        """
        new_vec = []
        if hasattr(vec, 'cpu'):
            vec = vec.cpu()
        for i in vec:
            new_vec.append(i)
        return self.dict.vec2txt(new_vec)
    
    def _generate(
        self,
        batch: Batch,
        beam_size: int,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        Generate an output with beam search.

        Use HF's built-in generation to perform beam search.
        """
        bad_words_ids = None
        if self.beam_block_list is not None:
            bad_words_ids = [
                gram for _, ngram in self.beam_block_list.items() for gram in ngram
            ]

        method = self.opt.get('inference', 'greedy')

        # select knowledge (full_tk, full_mask, ck_attn)
        full_tk, _, _ = self.model.knowledge_encoder( batch.text_vec,
                                            batch.know_vec,
                                            batch.ck_mask,
                                            batch.cs_ids,
                                            batch.use_cs_ids)

        generation_params = {
            'input_ids': full_tk,
            'max_length': max_ts,
            'min_length': self.beam_min_length,
            'do_sample': self.opt['inference'] in ['topk', 'topp'],
            'early_stopping': None,
            'num_beams': beam_size,
            'temperature': self.temperature,
            'top_k': self.opt['topk'] if method in ['topk', 'delayedbeam'] else None,
            'top_p': self.opt['topp'] if method == 'nucleus' else None,
            'repetition_penalty': None,
            'bad_words_ids': bad_words_ids if bad_words_ids else None,
            'bos_token_id': self.START_IDX,
            'pad_token_id': self.NULL_IDX,
            'eos_token_id': self.END_IDX,
            'length_penalty': self.opt['beam_length_penalty'],
            'no_repeat_ngram_size': self.beam_block_ngram,
            'num_return_sequences': None,
            'attention_mask': full_tk != self.NULL_IDX,
            'decoder_start_token_id': self.NULL_IDX,
        }


        if overrides:
            generation_params.update(overrides)

        outputs = self.model.t5.generate(**generation_params)
        outputs = [(outputs[i], 0) for i in range(outputs.size(0))]
        return outputs, []



# t5_encoder_model_arch T5EndToEndTwoAgent




