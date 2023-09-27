import torchtext.transforms


def prepare_words(tokens, tokenizer, max_input_length, init_token, sep_token):
    """Preprocesses words such that they may be passed into BERT.

    Parameters
    ---
    tokens : List
        List of strings, each of which corresponds to one token in a sequence.
    tokenizer : transformers.models.bert.tokenization_bert.BertTokenizer
        Tokenizer to be used for transforming word strings into word indices
        to be used with BERT.
    max_input_length : int
        Maximum input length of each sequence as expected by our version of BERT.
    init_token : str
        String representation of the beginning of sentence marker for our tokenizer.
    sep_token : str
        String representation of the end of sentence marker for our tokenizer.

    Returns
    ---
    tokens : List
        List of preprocessed tokens.
    """
    # Append beginning of sentence and end of sentence markers
    # lowercase each token and cut them to the maximum length
    # (minus two to account for beginning and end of sentence).
    tokens = (
        [init_token]
        + [i.lower() for i in tokens[: max_input_length - 2]] 
        + [sep_token]
    )
    # Convert word strings to indices.
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    return tokens


def prepare_tags(tokens, max_input_length, init_token, sep_token, pos_vocab):
    """Convert tag strings into indices for use with torch. For symmetry, we perform
        identical preprocessing as on our words, even though we do not need beginning
        of sentence and end of sentence markers for our tags.

    Parameters
    ---
    tokens : List
        List of strings, each of which corresponds to one token in a sequence.
    max_input_length : int
        Maximum input length of each sequence as expected by our version of BERT.
    init_token : str
        String representation of the beginning of sentence marker for our tokenizer.
    sep_token : str
        String representation of the end of sentence marker for our tokenizer.

    Returns
    ---
    tokens : List
        List of preprocessed tags.
    """
    # Append beginning of sentence and end of sentence markers
    # cut the tagging sequence to the maximum length (minus two to account for beginning and end of sentence).
    tokens = [init_token] + tokens[: max_input_length - 2] + [sep_token]
    # Convert tag strings to indices.
    tokens = torchtext.transforms.VocabTransform(pos_vocab)(tokens)
    return tokens


