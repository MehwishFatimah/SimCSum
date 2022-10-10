import torch
import torch.nn as nn
import transformers


class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, model_1, model_2, alpha, beta):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.shared_encoder = encoder
        self.sum_model = model_1
        self.sim_model = model_2
        self.model_list = [self.sum_model, self.sim_model]
        self.alpha = alpha
        self.beta = beta
        
        self.config.max_length = 200
        self.sum_model.config.max_length = 200
        self.sum_model.config.max_length = 512
        # self.config.bad_words_ids = self.sum_model.config.bad_words_ids
        # self.config.bos_token_id = self.sum_model.config.bos_token_id
        # self.config.pad_token_id = self.sum_model.config.pad_token_id
        # self.config.eos_token_id = self.sum_model.config.eos_token_id
        # self.config.sep_token_id = self.sum_model.config.sep_token_id
        # self.config.vocab_size = self.sum_model.config.vocab_size
        self.config.hidden_size = self.sum_model.config.hidden_size
        # self.config.num_attention_heads = self.sum_model.config.num_attention_heads
        # self.config.num_hidden_layers = self.sum_model.config.num_hidden_layers
        # self.config.length_penalty = self.sum_model.config.length_penalty
        # self.config.no_repeat_ngram_size = self.sum_model.config.no_repeat_ngram_size
        # self.config.repetition_penalty = self.sum_model.config.repetition_penalty
        # self.config.num_return_sequences = self.sum_model.config.num_return_sequences
        # self.config.num_beams = self.sum_model.config.num_beams
    @classmethod
    def create(cls, model_name, model_type, alpha=0.7, beta=0.3):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models.

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        model_1 = model_type.from_pretrained(model_name)
        model_2 = model_type.from_pretrained(model_name)
        shared_encoder = model_1.model.encoder
        model_2.model.encoder = shared_encoder
        return cls(shared_encoder, model_1, model_2, alpha, beta)


    def forward(self, input_ids, attention_mask=None, labels=None, sim_labels=None, **kwargs):
        """Generate the output for both summarization and simplification tasks.

        Args:
            sum_batch (dict): encoded batch from summarization task
            sim_batch (dict): encoded batch from simplification task
            task_name (Str):  if you want to do a forward pass on a specific task, provide a task name

        Returns:
            outputs (Seq2SeqLMOutput): outputs for summarization task or simplification task, or both
            
        TODO: parallelize: 1 gpu per model?
        """
        if (labels is not None):
            encoder_outputs = self.shared_encoder(input_ids, attention_mask)
            sum_outputs = self.sum_model(input_ids=input_ids, 
                                         attention_mask=attention_mask, 
                                         labels=labels, 
                                         encoder_outputs=encoder_outputs,
                                         **kwargs)
            sim_outputs = self.sim_model(input_ids=input_ids, 
                                         attention_mask=attention_mask, 
                                         labels=sim_labels, 
                                         encoder_outputs=encoder_outputs,
                                         **kwargs)
            sum_outputs.loss = self.alpha * sum_outputs.loss + self.beta * sim_outputs.loss
            return sum_outputs
        else:
            return self.sum_model(input_ids=input_ids, **kwargs)

    
    def resize_token_embeddings(self, new_num_tokens):
        for model in self.model_list:
            embeddings = model.resize_token_embeddings(new_num_tokens)
        return embeddings
    
    def _apply(self, fn):
        super()._apply(fn)
        for model in self.model_list:
            model._apply(fn)
        return self
    

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }
    
    def _reorder_cache(self, past, beam_idx):
        return self.sum_model._reorder_cache(past, beam_idx)