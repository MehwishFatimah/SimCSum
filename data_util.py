import os
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
import logging
import nltk

logger = logging.getLogger(__name__)


def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        
def detect_last_checkpoint(training_args):
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            

class DataPreprocessing():
    def __init__(self, sum_tokenizer, sim_tokenizer, model_args, data_args, training_args):
        self.sum_tokenizer = sum_tokenizer
        self.sim_tokenizer = sim_tokenizer
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.raw_datasets = self.load_datasets()
        self.max_target_length = data_args.max_target_length
        self.padding = "max_length" if data_args.pad_to_max_length else False
        self.column_names = self.find_column_names()
        self.init_data_columns()
        
    def load_datasets(self):
        # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
        # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
        # (the dataset will be downloaded automatically from the datasets Hub).
        #
        # For CSV/JSON files this script will use the first column for the full texts and the second column for the
        # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
        #
        # In distributed training, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.
        if self.data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config_name,
                cache_dir=self.model_args.cache_dir,
            )
        else:
            data_files = {}
            if self.data_args.train_file is not None:
                data_files["train"] = self.data_args.train_file
                extension = self.data_args.train_file.split(".")[-1]
            if self.data_args.validation_file is not None:
                data_files["validation"] = self.data_args.validation_file
                extension = self.data_args.validation_file.split(".")[-1]
            if self.data_args.test_file is not None:
                data_files["test"] = self.data_args.test_file
                extension = self.data_args.test_file.split(".")[-1]
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=self.model_args.cache_dir,
            )
        return raw_datasets
    
    def get_column_names(self):
        # Get the column names for input/target.
        if self.data_args.text_column is None:
            text_column = "text"
        else:
            text_column = self.data_args.text_column
            if text_column not in self.find_column_names():
                raise ValueError(
                    f"--text_column' value '{self.data_args.text_column}' needs to be one of: {', '.join(self.column_names)}"
                )
        if self.data_args.summary_column is None:
            summary_column = "summary"
        else:
            summary_column = self.data_args.summary_column
            if summary_column not in self.find_column_names():
                raise ValueError(
                    f"--summary_column' value '{self.data_args.summary_column}' needs to be one of: {', '.join(self.column_names)}"
                )
        if self.data_args.simple_column is None:
            simple_column = "simple"
        else:
            simple_column = self.data_args.simple_column
            if simple_column not in self.find_column_names():
                raise ValueError(
                    f"--simple_column' value '{self.data_args.simple_column}' needs to be one of: {', '.join(self.column_names)}"
                )
        return text_column, summary_column, simple_column
        
    def find_column_names(self):
        if self.training_args.do_train:
            column_names = self.raw_datasets["train"].column_names
        elif self.training_args.do_eval:
            column_names = self.raw_datasets["validation"].column_names
        elif self.training_args.do_predict:
            column_names = self.raw_datasets["test"].column_names
        else:
            logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return column_names
        
    def init_data_columns(self):
        self.text_column, self.summary_column, self.simple_column = self.get_column_names()
        
    def preprocess_function_train(self, examples):
        """The preprocess function for the train dataset, where simplification targets are included.    

        Args:
            examples (Dataset): a batch of examples containing the data

        Returns:
            model_inputs (Dict): a dict containing the input for the models forward function, containing:
            {
                "input_ids": ...,
                "attention_mask": ...,
                "labels": ...,
                "sim_labels: ...
                
            }
        """
        # remove pairs where at least one record is None
        inputs, sum_targets, sim_targets = [], [], []
        for i in range(len(examples[self.text_column])):
            if examples[self.text_column][i] and examples[self.summary_column][i] and examples[self.simple_column][i]:
                inputs.append(examples[self.text_column][i])
                sum_targets.append(examples[self.summary_column][i])
                sim_targets.append(examples[self.simple_column][i])

        model_inputs = self.sum_tokenizer(inputs, max_length=self.data_args.max_source_length, padding=self.padding, truncation=True)
        # Tokenize targets with the `text_target` keyword argument
        labels = self.sum_tokenizer(text_target=sum_targets, max_length=self.max_target_length, padding=self.padding, truncation=True)
        sim_labels = self.sim_tokenizer(text_target=sim_targets, max_length=self.data_args.max_source_length, padding=self.padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        for lab in [labels, sim_labels]:
            if self.padding == "max_length" and self.data_args.ignore_pad_token_for_loss:
                lab["input_ids"] = [
                    [(l if l != self.sum_tokenizer.pad_token_id else -100) for l in label] for label in lab["input_ids"]
                ]

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["sim_labels"] = sim_labels["input_ids"]
        return model_inputs
    
    def preprocess_function_eval(self, examples):
        """The preprocess function for the test dataset, where only summarization targets are included

        Args:
            examples (Dataset): a batch of examples containing the data

        Returns:
            model_inputs (Dict): a dict containing the input for the models forward function, containing:
            {
                "input_ids": ...,
                "attention_mask": ...,
                "labels": ...,
            }
        """
        # remove pairs where at least one record is None
        inputs, sum_targets = [], []
        for i in range(len(examples[self.text_column])):
            if examples[self.text_column][i] and examples[self.summary_column][i] and examples[self.simple_column][i]:
                inputs.append(examples[self.text_column][i])
                sum_targets.append(examples[self.summary_column][i])
        model_inputs = self.sum_tokenizer(inputs, max_length=self.data_args.max_source_length, padding=self.padding, truncation=True)
        labels = self.sum_tokenizer(text_target=sum_targets, max_length=self.max_target_length, padding=self.padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == "max_length" and self.data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.sum_tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def run_preprocessing(self):
        train_dataset, eval_dataset, predict_dataset = None, None, None
        if self.training_args.do_train:
            train_dataset = self.run_train_preprocessing()
        if self.training_args.do_eval:
            eval_dataset = self.run_eval_preprocessing()
        if self.training_args.do_predict:
            predict_dataset = self.run_predict_preprocessing()
        return train_dataset, eval_dataset, predict_dataset
    
    def run_train_preprocessing(self):
        if "train" not in self.raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = self.raw_datasets["train"]
        if self.data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), self.data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with self.training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                self.preprocess_function_train,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=self.get_column_names(),
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        return train_dataset
    
    def run_eval_preprocessing(self):
        self.set_max_target_length(self.data_args.val_max_target_length)
        if "validation" not in self.raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = self.raw_datasets["validation"]
        if self.data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), self.data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with self.training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                self.preprocess_function_train,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=self.get_column_names(),
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        return eval_dataset
    
    def run_predict_preprocessing(self):
        self.set_max_target_length(self.data_args.val_max_target_length)
        if "test" not in self.raw_datasets: 
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = self.raw_datasets["test"]
        if self.data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), self.data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with self.training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                self.preprocess_function_eval,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=self.get_column_names(),
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )    
        return predict_dataset
    
    def get_raw_datasets(self):
        return self.raw_datasets
    
    def get_column_names(self):
        return self.column_names
    
    def get_references(self, dataset):
        references = []
        for i in range(len(dataset[self.summary_column])):
            if dataset[self.summary_column][i]:
                references.append(dataset[self.summary_column][i])
        return references
    
    def set_max_target_length(self, max_target_length):
        self.max_target_length = max_target_length
