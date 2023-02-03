def preprocess_function_train(examples, tokenizer, text_column, summary_column, simple_column):
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
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i] and examples[simple_column][i]:
                inputs.append(examples[text_column][i])
                sum_targets.append(examples[summary_column][i])
                sim_targets.append(examples[simple_column][i])

        model_inputs = sum_tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        # Tokenize targets with the `text_target` keyword argument
        labels = sum_tokenizer(text_target=sum_targets, max_length=max_target_length, padding=padding, truncation=True)
        sim_labels = sim_tokenizer(text_target=sim_targets, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        for lab in [labels, sim_labels]:
            if padding == "max_length" and data_args.ignore_pad_token_for_loss:
                lab["input_ids"] = [
                    [(l if l != sum_tokenizer.pad_token_id else -100) for l in label] for label in lab["input_ids"]
                ]

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["sim_labels"] = sim_labels["input_ids"]
        return model_inputs
    
def preprocess_function_eval(examples):
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
    for i in range(len(examples[text_column])):
        if examples[text_column][i] and examples[summary_column][i] and examples[simple_column][i]:
            inputs.append(examples[text_column][i])
            sum_targets.append(examples[summary_column][i])
    model_inputs = sum_tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
    labels = sum_tokenizer(text_target=sum_targets, max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != sum_tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs