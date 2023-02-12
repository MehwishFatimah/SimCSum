import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForSeq2SeqLM
from transformers.modeling_outputs import Seq2SeqLMOutput

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, model_1, model_2, lambda_):
        """Inheriting from PreTrainedModel because then we can benefit from the 
        Huggingface Trainer implementation

        Args:
            encoder (Encoder): the shared encoder for our model
            model_1 (Model): The decoder + LM head for the summarization part of our model
            model_2 (Model): The decoder + LM head for the simplification part of our model
            lambda_ (int): value to weigh the main task loss
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.sum_model = model_1
        self.sim_model = model_2
        self.model_list = [self.sum_model, self.sim_model]
        self.lambda_ = lambda_
        self.config = self.sum_model.config
        self.init_shared_cross_attention()

    @classmethod
    def create(cls, model_name, main_target_lang_id, aux_target_lang_id, max_length, lambda_):
        """        This creates a MultitaskModel using the model class and model name
        from single-task models.

        We do this by creating each single-task model, and having them share
        the same encoder.

        Args:
            model_name (str): The pretrained model name or checkpoint you want train further on
            target_lang_id (int): The language id for the summarization decoder
            source_lang_id (int): The language id for the simplification decoer.
            lambda_ (float, optional): The value with which we weigh how much the main task loss influences the total loss.

        Returns:
            MultitaskModel: Multitask model with two submodels who share an encoder.
        """
        logging.info(
            f"Loading {model_name} for summarization from pretrained...")
        model_1 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        logging.info(
            f"Loading {model_name} for simplification from pretrained...")
        model_2 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model_1.config.decoder_start_token_id = main_target_lang_id
        model_2.config.decoder_start_token_id = aux_target_lang_id
        model_1.config.forced_bos_token_id = main_target_lang_id
        model_2.config.forced_bos_token_id = aux_target_lang_id
        model_1.config.max_length = max_length
        model_2.config.max_length = max_length
        shared_encoder = model_1.get_encoder() # you could change this line to shared_encoder_decoder = model_1.model
        logging.info(f"Setting shared encoder...{shared_encoder}")
        model_2.model.encoder = shared_encoder # and this to model_2.model = shared_encoder_decoder
        # This way not only the encoder would be shared, but the encoder an the decoder.
        # You would have to think about the way to input both the summarization labels and the simplification labels into the decoder. 
        # One possibility: By alternating randomly between summarization and simplification labels
        logging.info("Finished initizializing multitask model.")
        return cls(shared_encoder, model_1, model_2, lambda_)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        sim_labels=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        """Forward function of our multitask model. Arguments taken from MBartForConditionalGeneration code.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`.): 
                Indices of input sequence tokens in the vocab
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, optional): 
                Mask to avoid performing attention on padding token indices. Defaults to None.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, optional): 
                Summarization labels. The labels are used for computing the masked language modeling loss. Defaults to None.
            sim_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, optional): 
                Simplification labels. The labels are used for computing the masked language modeling loss. Defaults to None.
            decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, optional):  
                Indices of decoder input sequence tokens in the vocabulary.. Defaults to None.

        Returns:
            Seq2SeqModelOutput: The output of our forward function, more information here:
            https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput
        """
        if (sim_labels is not None):
            if encoder_outputs is None:
                encoder_outputs = self.encoder(input_ids, attention_mask)
            sum_outputs = self.sum_model(
                input_ids,
                attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
                head_mask,
                decoder_head_mask,
                cross_attn_head_mask,
                encoder_outputs,
                past_key_values,
                inputs_embeds,
                decoder_inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict
            )
            sim_outputs = self.sim_model(
                input_ids,
                attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
                head_mask,
                decoder_head_mask,
                cross_attn_head_mask,
                encoder_outputs,
                past_key_values,
                inputs_embeds,
                decoder_inputs_embeds,
                sim_labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict
            )
            sum_outputs.loss = self.lambda_ * sum_outputs.loss + (1 - self.lambda_) * sim_outputs.loss
            return sum_outputs
        else: # when no simplification labels are given, we only use the summarization part of the model.
            return self.sum_model(
                input_ids,
                attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
                head_mask,
                decoder_head_mask,
                cross_attn_head_mask,
                encoder_outputs,
                past_key_values,
                inputs_embeds,
                decoder_inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict
            )

    def resize_token_embeddings(self, new_num_tokens):
        for model in self.model_list:
            embeddings = model.resize_token_embeddings(new_num_tokens)
        return embeddings

    def _apply(self, fn):
        """
        Not sure if this function is needed, but included it to make sure that multitask_model.to(device) is applied to
        the summarization and simplification models as well.
        """
        super()._apply(fn)
        for model in self.model_list:
            model._apply(fn)
        return self

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        """
        Prepares decoder input ids for generation. Used for generation by Seq2SeqTrainer if predict_with_generate is True. We just return 
        the MBartForConditionalGeneration method for this.
        """

        return self.sum_model.prepare_inputs_for_generation(
            decoder_input_ids,
            past,
            attention_mask,
            head_mask,
            decoder_head_mask,
            cross_attn_head_mask,
            use_cache,
            encoder_outputs,
            **kwargs
        )

    def _reorder_cache(self, past, beam_idx):
        """
        Also used during evaluation. Return the the MBartForConditionalGeneration method for this. 
        """
        return self.sum_model._reorder_cache(past, beam_idx)

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        """
        Also used during evaluation. Return the the MBartForConditionalGeneration method for this. 
        """
        return self.sum_model.prepare_decoder_input_ids_from_labels(labels)

    def get_encoder(self):
        """
        Returns the shared encoder. Used for evaluation by Seq2SeqTrainer.
        """
        return self.encoder

    def get_decoder(self):
        """
        Returns the decoder of our summarization model. Used for evaluation by Seq2SeqTrainer.
        """
        return self.sum_model.get_decoder()
    
    def init_shared_crossattention(self):
        """
        Initialize a shared crossattention between the summarization decoder and the simplification decoder.
        """
        sum_decoder = self.sum_model.get_decoder()
        sim_decoder = self.sim_model.get_decoder()
        logging.info("Setting shared cross attention layer for every decoder layer...")
        for sum_layer, sim_layer in zip(sum_decoder.layers, sim_decoder.layers):
            shared_cross_attention_layer = sum_layer.encoder_attn
            sim_layer.encoder_attn = shared_cross_attention_layer
        logging.info(f"Done! Last shared cross attention layer: {shared_cross_attention_layer}")