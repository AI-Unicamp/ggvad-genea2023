import os

import numpy as np
import torch

from embedding_space_evaluator import EmbeddingSpaceEvaluator
from train_AE import make_tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_fgd(fgd_evaluator, gt_data, test_data):
    fgd_evaluator.reset()

    print('Pushing real samples...')
    fgd_evaluator.push_real_samples(gt_data)
    print('Pushing generated samples...')
    fgd_evaluator.push_generated_samples(test_data)
    print('Calculating FGD with feat space...')
    fgd_on_feat = fgd_evaluator.get_fgd(use_feat_space=True)
    #print('Calculating FGD without feat space...')
    #fdg_on_raw = fgd_evaluator.get_fgd(use_feat_space=False)
    return fgd_on_feat#, fdg_on_raw


def add_noise(x):
    noise_level = 1
    x_noise = x + (noise_level ** 0.5) * torch.randn(x.size()).to(device)
    return x_noise


def exp_base(chunk_len):
    # AE model
    ae_path = f'./evaluation_metric/output/model_checkpoint_{chunk_len}.bin'
    fgd_evaluator = EmbeddingSpaceEvaluator(ae_path, chunk_len, device)

    # load human data
    human_data = make_tensor(f'./dataset/Genea2023/trn/main-agent/motion_npy_rotpos', chunk_len).to(device)
    print(human_data.size())

    # simulate generated motion by adding noise to human motion
    # load the generated motion when you actually use this code for model evaluation
    #test_data = add_noise(human_data)
    test_data = make_tensor(f'./dataset/Genea2023/val/main-agent/motion_npy_rotpos', chunk_len).to(device)
    print(human_data.size())

    print(f'----- Experiment (motion chunk length: {chunk_len}) -----')
    print('FGDs on feature space and raw data space')
    #fgd_on_feat, fgd_on_raw = run_fgd(fgd_evaluator, human_data, test_data)
    fgd_on_feat = run_fgd(fgd_evaluator, human_data, test_data)
    #print(f'{fgd_on_feat:8.3f}, {fgd_on_raw:8.3f}')
    print(f'{fgd_on_feat:8.3f}')
    print()


if __name__ == '__main__':
    exp_base(120)
