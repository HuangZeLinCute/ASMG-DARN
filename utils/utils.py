import os
import random
from collections import OrderedDict

import numpy as np
import torch


def seed_everything(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, epoch, model_name, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    checkpoint_file = os.path.join(outdir, model_name + '_' + 'epoch_' + str(epoch) + '.pth')
    torch.save(state, checkpoint_file)


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location=lambda storage, loc: storage.cuda(0))
    
    # Handle different weight file formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # If these keys don't exist, assume the entire checkpoint is state_dict
        state_dict = checkpoint
    
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module'):
            name = key[7:]
        else:
            name = key
        new_state_dict[name] = value
    
    # Compatibility loading: if boundary attention modules don't exist in weights but exist in model, skip these layers
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(new_state_dict.keys())
    
    # Find keys that exist in model but not in weights (newly added boundary attention modules)
    missing_keys = model_keys - checkpoint_keys
    # Find keys that exist in weights but not in model
    unexpected_keys = checkpoint_keys - model_keys
    
    if missing_keys:
        print(f"Warning: The following keys are missing in the weights file (possibly newly added boundary attention modules):")
        for key in sorted(missing_keys):
            if 'doc_boundary' in key:
                print(f"  - {key}")
        print("These layers will use randomly initialized weights.")
    
    if unexpected_keys:
        print(f"Warning: The weights file contains the following unexpected keys:")
        for key in sorted(unexpected_keys):
            print(f"  - {key}")
    
    # Load only matching weights
    filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in model_keys}
    
    # Use strict=False to allow partial loading
    model.load_state_dict(filtered_state_dict, strict=False)
    
    print(f"Successfully loaded {len(filtered_state_dict)}/{len(model_keys)} weight parameters")

