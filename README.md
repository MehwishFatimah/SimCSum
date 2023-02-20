
# SIMCSUM

This is the code for the paper [Link]




## Requirements

Tested with Python [version number].
## Setting up

git checkout https://github.com/timkolber/mtl_sum.git \
cd mtl_sum \
python run_summarization.py [--[ModelArguments](#ModelArguments)] [--[DataTrainingArguments](#DataTrainingArguments)]

## Arguments

- Seq2SeqTrainingArguments
- ModelArguments
- DataTrainingArguments

ModelArguments and DataTrainingArguments are defined in training_arguments.py

### ModelArguments
- model_name_or_path: Path to pretrained model or model identifier from huggingface.co/models
- config_name: Pretrained config name or path if not the same as model_name
- tokenizer_name: Pretrained tokenizer name or path if not the same as model_name
- cache_dir: Where to store the pretrained models downloaded from huggingface.co
- use_fast_tokenizer: Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.
- model_revision: The specific model version to use (can be a branch name, tag name or commit id).
- resize_position_embeddings: Whether to automatically resize the position embeddings if `max_source_length` exceeds the model's position embeddings.

### DataTrainingArguments
- src_lang: Source Language id.

- main_tgt_lang: Target Language id for the main task.
    
- aux_tgt_lang: Target Language id for the auxiliairy task.

- dataset_name: The name of the dataset to use (via the datasets library).

- dataset_config_name: The configuration name of the dataset to use (via the datasets library).

- text_column: The name of the column in the datasets containing the full texts (for summarization).

- summary_column: The name of the column in the datasets containing the summaries (for summarization).

- simple_column: The name of the column in the datasets containing the summaries (for summarization).

- train_file: The input training data file (a jsonlines or csv file)."}

- validation_file: An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file).

- test_file: An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file).

- test_output_path: An optional path and file name for the predictions/references file used for evaluation.

- overwrite_cache: Overwrite the cached training and evaluation sets"

- preprocessing_num_workers: The number of processes to use for the preprocessing."

- max_source_length: The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.

- max_target_length: The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.

- val_max_target_length: The maximum total sequence length for validation target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`. This argument is also used to override the `max_length` param of `model.generate`, which is used 
                "during `evaluate` and `predict`.

- pad_to_max_length: Whether to pad all samples to model maximum sentence length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More efficient on GPU but very bad for TPU.

- max_train_samples: For debugging purposes or quicker training, truncate the number of training examples to this value if set.

- max_eval_samples: For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.

- max_predict_samples: For debugging purposes or quicker training, truncate the number of prediction examples to this value if set.

- num_beams: Number of beams to use for evaluation. This argument will be passed to `model.generate`, which is used during `evaluate` and `predict`.

- lambda_: Value of the lambda weight, used for the weighted sum of the summarization loss and simplification loss.

- ignore_pad_token_for_loss: Whether to ignore the tokens corresponding to padded labels in the loss computation or not.

- wandb_enabled: Whether to enable wandb.

## Examples

