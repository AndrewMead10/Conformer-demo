def pad_sequence(sequences, batch_first=False, padding_value=0):

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(1) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_size[0], max_len)
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)

    for i, tensor in enumerate(sequences):
        width = tensor.size(1)
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, :width] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor