

import torch



def copy_weights_ignore_name(src_model, dst_model):
    src_params = list(src_model.parameters())
    dst_params = list(dst_model.parameters())

    assert len(src_params) == len(dst_params), \
        f"Parameter count mismatch: {len(src_params)} vs {len(dst_params)}"

    for src, dst in zip(src_params, dst_params):
        assert src.shape == dst.shape, \
            f"Shape mismatch: {src.shape} vs {dst.shape}"
        dst.data.copy_(src.data)
        
