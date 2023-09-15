from argparse import ArgumentParser
import os
import numpy as np
import librosa
import torch
from wavlm.WavLM import WavLM, WavLMConfig
import torch.nn.functional as F
from data_loaders.gesture.scripts.motion_process import bvh2representations2
import bvhsdk
from tqdm import tqdm

def main(args):
    #paths_check(args.data_dir)
    assert args.split in ['all', 'trn', 'tst', 'val'], f"Split {args.split} not recognized. Options: \'all\', \'trn\', \'tst\', \'val\'" # Check if user is trying to process a split that does not exist
    splits = [args.split] if args.split != 'all' else ['trn', 'tst', 'val']
    assert args.step in ['all', 'bvh', 'wav', 'wavlm'], f"Step {args.step} not recognized. Options: \'all\', \'bvh\', \'wav\', \'wavlm\'" # Check if user is trying to process a step that does not exist
    steps = [args.step] if args.step != 'all' else ['bvh', 'wav', 'wavlm']
    print('WARNING: Running all steps and all splits will take a long time.')
    print('Processing splits: ', splits)
    print('Processing steps: ', steps)
    for split in splits:
        print(f'Processing {split} split')
        if 'bvh' in steps and split != 'tst':
            print(f'Processing bvh for {split} split')
            r6p, rp = process_bvh(args.data_dir, split)
            statspath = os.path.join(args.data_dir, split, 'main-agent')
            print(f'Computing mean and std for {split} split')
            compute_meanstd(r6p, os.path.join(statspath, 'rot6dpos'), npstep=5)
            compute_meanstd(rp, os.path.join(statspath, 'rotpos'), npstep=5)
            compute_meanstd(rp, os.path.join(statspath, 'velrotpos'), npstep=5, vel=True)
        if 'wav' in steps:
            print(f'Processing wav for {split} split')
            process_wav(args.data_dir, split)
        if 'wavlm' in steps:
            print(f'Processing wavlm for {split} split')
            process_wavlm(args.data_dir, split)

def process_bvh(path, split):
    sourcepath = os.path.join(path, split, 'main-agent', 'bvh')
    savepathrot6d = os.path.join(path, split, 'main-agent', 'motion_npy_rot6dpos')
    savepathrot = os.path.join(path, split, 'main-agent', 'motion_npy_rotpos')
    assert not os.path.exists(savepathrot6d), f"motion_npy_rot6dpos already exists in {savepathrot6d}. Delete it to process again."
    assert not os.path.exists(savepathrot), f"motion_npy_rotpos already exists in {savepathrot}. Delete it to process again."
    if not os.path.exists(savepathrot6d):
        os.mkdir(savepathrot6d)
    if not os.path.exists(savepathrot):
        os.mkdir(savepathrot)
    for file in tqdm(os.listdir(sourcepath)):
        #if not os.path.exists(os.path.join(savepathrot6d, file[:-4] + '.npy')) or not os.path.exists(os.path.join(savepathrot, file[:-4] + '.npy')):
        anim = bvhsdk.ReadFile(os.path.join(sourcepath, file))
        rot6dpos, rotpos = bvh2representations2(anim)
        np.save(os.path.join(savepathrot6d, file[:-4]), rot6dpos)
        np.save(os.path.join(savepathrot, file[:-4]), rotpos)
    return savepathrot6d, savepathrot

def compute_meanstd(path, savepath, npstep=1, vel=False):
    all_data = []
    for f in os.listdir(path)[::npstep]:
        data = np.load(os.path.join(path, f))
        if vel:
            data = data[1:,:] - data[:-1,:]
            data[0,:] = np.zeros(data.shape[1])
        all_data.append(data)
    all_data = np.vstack(all_data)
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
    np.save(savepath + '_Mean.npy', mean)
    np.save(savepath + '_Std.npy', std)


def process_wav(path, split, sr=16000):
    sourcepath = os.path.join(path, split, 'main-agent', 'wav')
    savepath = os.path.join(path, split, 'main-agent', 'audio16k_npy')
    assert not os.path.exists(savepath), f"audio_16k_npy already exists in {savepath}. Delete it to process again."
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    for file in tqdm(os.listdir(sourcepath)):
        #if not os.path.exists(os.path.join(savepath, file[:-4] + '.npy')):
        signal, _sr = librosa.load(os.path.join(sourcepath, file), mono=True, sr=sr)
        assert _sr == sr
        np.save(os.path.join(savepath, file[:-4]+'.npy'), signal)
    return savepath

def process_wavlm(path, split):
    wavlm_layer = 11 
    fps=30
    sr=16000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sourcepath = os.path.join(path, split, 'main-agent', 'audio16k_npy')
    savepath = os.path.join(path, split, 'main-agent', 'wavlm_representations')
    #assert os.path.exists(sourcepath), f"audio16k_npy not found in {sourcepath}. Required to process wavlm representations, make sure wav files were processed first."
    #assert os.path.exists(savepath), f"wavlm model directory not found in current directory."
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    checkpoint = torch.load('./wavlm/WavLM-Base+.pt')
    wavlm_cfg = WavLMConfig(checkpoint['cfg'])
    wavlm = WavLM(wavlm_cfg)
    #wavlm.to(device)
    wavlm.load_state_dict(checkpoint['model'])
    wavlm.eval()
    for file in tqdm(os.listdir(sourcepath)):
        if not os.path.exists(os.path.join(savepath, file)):
            audio_path = os.path.join(sourcepath, file)
            # Load with Numpy
            signal = np.load(audio_path)
            # Set to model innput format
            signal = torch.tensor(signal).unsqueeze(0)#.to(device)
            # Normalize
            if wavlm_cfg.normalize:
                signal_norm = torch.nn.functional.layer_norm(signal , signal.shape)
            else:
                signal_norm = signal
            # Run Model (rep=Desired Layer, layer_results=all layers)
            rep, layer_results = wavlm.extract_features(signal_norm, output_layer=wavlm_layer, ret_layer_results=True)[0]
            layer_reps = [x.transpose(0, 1) for x, _ in layer_results] # fix shape
            # Get Number of Seconds of Audio File
            n_secs = signal.shape[1] / sr
            # Get Number of poses equivalent to audio file duration, given fps (alignment len)
            n_pose = n_secs * fps
            # Interpolate number of representations to match number of poses corresponding to audio file
            interp_reps = F.interpolate(rep.transpose(1, 2), size=int(n_pose), align_corners=True, mode='linear')
            # Prepare to save
            interp_reps = interp_reps.squeeze(0).transpose(0,1).cpu().detach().data.cpu().numpy()
            # Double check dimension
            assert (interp_reps.shape[0] == int(np.ceil(n_pose)) or interp_reps.shape[0] == int(np.floor(n_pose)))
            np.save(os.path.join(savepath, file), interp_reps)


def paths_check(data_dir):
    # First check if everything is in place
    for split in ['trn', 'tst', 'val']:
        split_dir = os.path.join(data_dir, split)
        assert os.path.exists(split_dir), f"Split {split} not found in {data_dir}"
        main_agent_dir = os.path.join(split_dir, 'main-agent')
        assert os.path.exists(main_agent_dir), f"main_agent not found in {split_dir}"
        tsv_dir = os.path.join(main_agent_dir, 'tsv')
        wav_dir = os.path.join(main_agent_dir, 'wav')
        assert os.path.exists(tsv_dir), f"tsv not found in {main_agent_dir}"
        assert os.path.exists(wav_dir), f"wav not found in {main_agent_dir}"
        assert len(os.listdir(tsv_dir)) == len(os.listdir(wav_dir)), f"tsv and wav have different number of files in {main_agent_dir}"
        if split != 'tst':
            bvh_dir = os.path.join(main_agent_dir, 'bvh')
            assert os.path.exists(bvh_dir), f"bvhs not found in {main_agent_dir}"
            assert len(os.listdir(tsv_dir)) == len(os.listdir(bvh_dir)), f"tsv and bvh have different number of files in {main_agent_dir}"
    print('Data paths and files seem correct')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset/Genea2023', help='path to the dataset directory')
    parser.add_argument('--split', type=str, default='all', help='Which split to process. Use \'all\' to process all splits')
    parser.add_argument('--step', type=str, default='all', help='Which step to process. Use \'all\' to process all steps. Options: \'bvh\', \'wav\', \'wavlm\'')
    args = parser.parse_args()
    main(args)