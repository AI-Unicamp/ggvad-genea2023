from data_loaders.gesture.scripts import motion_process as mp
from data_loaders.get_data import get_dataset_loader
import numpy as np
from tqdm import tqdm
from utils import dist_util
import torch
import bvhsdk
from evaluation_metric.embedding_space_evaluator import EmbeddingSpaceEvaluator
from evaluation_metric.train_AE import make_tensor
import matplotlib.pyplot as plt

# Imports for calling from command line
from utils.parser_util import generate_args
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_model_wo_clip


class GeneaEvaluator:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fgd_evaluator = EmbeddingSpaceEvaluator('./evaluation_metric/output/model_checkpoint_120.bin', 120, self.device)
        
    def eval(self, samples=None, chunks=None):
        print('Starting evaluation...')

        # Compute FGD
        fgd_on_feat = self.fgd(40, None)
    
    def fgd(self, n_samples=100, n_chunks=1):
        # "Direct" ground truth positions
        real_val = make_tensor(f'./dataset/Genea2023/trn/main-agent/motion_npy_rotpos', 120, max_files=n_samples, n_chunks=n_chunks, stride=40).to(self.device)
        test_data = make_tensor(f'./dataset/Genea2023/val/main-agent/motion_npy_rotpos', 120, max_files=n_samples, n_chunks=n_chunks, stride=40).to(self.device)

        fgd_on_feat = self.run_fgd(real_val, test_data)
        print(f'Validation to train: {fgd_on_feat:8.3f}')
        return fgd_on_feat

    def fgd_prep(self, data, n_frames=120, stride=None):
        # Prepare samples for FGD evaluation
        samples = []
        stride = n_frames // 2 if stride is None else stride
        for take in data:
            for i in range(0, len(take) - n_frames, stride):
                sample = take[i:i+n_frames]
                sample = (sample - self.data.mean[self.idx_positions]) / self.data.std[self.idx_positions]
                samples.append(sample)
        return torch.Tensor(samples)

    def run_fgd(self, gt_data, test_data):
        # Run FGD evaluation on the given data
        self.fgd_evaluator.reset()
        self.fgd_evaluator.push_real_samples(gt_data)
        self.fgd_evaluator.push_generated_samples(test_data)
        fgd_on_feat = self.fgd_evaluator.get_fgd(use_feat_space=True)
        return fgd_on_feat
    
def main():
    GeneaEvaluator().eval()


if __name__ == '__main__':
    main()