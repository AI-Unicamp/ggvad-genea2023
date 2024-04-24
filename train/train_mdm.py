# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
import numpy as np

def main():
    args = train_args()
    fixseed(args.seed)

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    if args.wandb:
        projectname = os.path.basename(os.path.normpath(args.save_dir))
        import wandb
        wandb.login(anonymous="allow")
        wandb.init(project='ggvad-genea2023', config=vars(args))
        args.wandb = wandb

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data = get_dataset_loader(name=args.dataset, 
                              data_dir=args.data_dir, 
                              batch_size=args.batch_size, 
                              num_frames=args.num_frames, 
                              step=args.step, 
                              use_wavlm=args.use_wavlm, 
                              use_vad=args.use_vad, 
                              vadfromtext=args.vadfromtext)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, model, diffusion, data).run_loop()

if __name__ == "__main__":
    main()
