from argparse import ArgumentParser
import os
import numpy as np
import torch
from speechbrain.pretrained import VAD
import torchaudio
from scipy.signal import resample
from tqdm import tqdm



def main(args):
    #paths_check(args.data_dir)
    assert args.split in ['all', 'trn', 'tst', 'val'], f"Split {args.split} not recognized. Options: \'all\', \'trn\', \'tst\', \'val\'" # Check if user is trying to process a split that does not exist
    splits = [args.split] if args.split != 'all' else ['trn', 'tst', 'val']
   
    print('Processing VAD.')
    print('Processing splits: ', splits)
    for split in splits:
        print(f'Processing vad for {split} split')
        process_vad(args.data_dir, split)
    
    

def process_vad(path, split):
    sr=16000
    fps=30
    sourcepath = os.path.join(path, split, 'main-agent', 'wav')
    savepathrot = os.path.join(path, split, 'main-agent', 'vad')
    _VAD = VAD.from_hparams(source= "speechbrain/vad-crdnn-libriparty", savedir= os.path.join(path, '..','..','speechbrain', 'pretrained_models', 'vad-crdnn-libriparty'))
    #assert not os.path.exists(savepathrot), f"vad already exists in {savepathrot}. Delete it to process again."
    if not os.path.exists(savepathrot):
        os.mkdir(savepathrot)
    # VAD requires a torch tensor with sample rate = 16k. This process saves a temporary wav file with 16k sr. It can be deleted after processing.
    for file in tqdm(os.listdir(sourcepath)):
        audio, old_sr = torchaudio.load(os.path.join(sourcepath,file))
        audio = torchaudio.functional.resample(audio, orig_freq=44100, new_freq=sr)
        tmpfile = "tmp.wav"
        torchaudio.save(
        tmpfile , audio, sr
        )
        boundaries = _VAD.get_speech_prob_file(audio_file=tmpfile, large_chunk_size=4, small_chunk_size=0.2)
        boundaries = resample(boundaries[0,:,0], int(boundaries.shape[1]*fps/100))
        boundaries[boundaries>=0.5] = 1
        boundaries[boundaries<0.5] = 0
        np.save(os.path.join(savepathrot, file[:-4]+'.npy'), boundaries)

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
    args = parser.parse_args()
    main(args)