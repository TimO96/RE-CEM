def h5_to_state_dict(h5_file, mapping):
    """Create Pytorch state_dict from h5 weight with mapping."""
    state_dict = {}
    with h5py.File(h5_file, 'r') as f:
        for h5, state in mapping.items():
            # Weights in tensorflow are transposed with respect to Pytorch.
            state_dict[state] = from_numpy(f[h5][:].T)

    return state_dict

def load_AE(codec_prefix, print_summary=False, dir="models/"):
    """Load autoencoder from json. Optionally print summary."""
    # Weights file.
    weight_file = dir + codec_prefix  + ".h5"
    if not os.path.isfile(weight_file):
        raise Exception(f"Decoder weight file {weight_file} not found.")

    # AE mapping from keras format to Pytorch.
    AE_map = {
        'conv2d_4/conv2d_4/bias:0'       : 'decoder.0.bias',
        'conv2d_4/conv2d_4/kernel:0'     : 'decoder.0.weight',
        'conv2d_5/conv2d_5/bias:0'       : 'decoder.3.bias',
        'conv2d_5/conv2d_5/kernel:0'     : 'decoder.3.weight',
        'conv2d_6/conv2d_6/bias:0'       : 'decoder.5.bias',
        'conv2d_6/conv2d_6/kernel:0'     : 'decoder.5.weight',
        'sequential_1/conv2d_1/bias:0'   : 'encoder.0.bias',
        'sequential_1/conv2d_1/kernel:0' : 'encoder.0.weight',
        'sequential_1/conv2d_2/bias:0'   : 'encoder.2.bias',
        'sequential_1/conv2d_2/kernel:0' : 'encoder.2.weight',
        'sequential_1/conv2d_3/bias:0'   : 'encoder.5.bias',
        'sequential_1/conv2d_3/kernel:0' : 'encoder.5.weight',
    }

    # Create autoencoder instance.
    ae = AE(h5_to_state_dict(weight_file, AE_map))

    # Print autoencoder structure.
    if print_summary:
        summary(ae, (ae.image_size, ae.image_size, ae.num_channels))

    return ae
