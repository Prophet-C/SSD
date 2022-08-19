import torch 


ckpt_dict = torch.load('./checpoints/last.pth', map_location="cpu")

# import pdb
# pdb.set_trace()
# ckpt_dict['model'].keys()

checkpoint = ckpt_dict['model']

checkpoint_new = {}

import pdb
pdb.set_trace()

for key in checkpoint.keys():
    if key.startswith('encoder.module.'):
        key_new = key.replace('encoder.module.', 'backbone.')
        checkpoint_new[key_new] = checkpoint[key]
        

    elif key.startswith('head.'):
        key_new = key.replace('head.', 'simclr_head.')
        checkpoint_new[key_new] = checkpoint[key]

torch.save(checkpoint_new, './checpoints/last_new.pth')
print()