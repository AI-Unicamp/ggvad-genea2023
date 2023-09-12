from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import gg_collate

def get_dataset_class(name):
    if name in ["genea2023", "genea2023+"]:
        from data_loaders.gesture.data.dataset import Genea2023
        return Genea2023
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    if name in ["genea2023", "genea2023+"]:
        return gg_collate
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_dataset(name, num_frames, seed_poses, step, use_wavlm, use_vad, vadfromtext, split='trn', hml_mode='train', ):
    DATA = get_dataset_class(name)
    dataset = DATA(name=name, split=split, window=num_frames, n_seed_poses=seed_poses, step=step, use_wavlm=use_wavlm, use_vad=use_vad, vadfromtext=vadfromtext)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, step, use_wavlm, use_vad, vadfromtext, split='trn', hml_mode='train', seed_poses=10):
    dataset = get_dataset(name, num_frames, seed_poses, step, use_wavlm, use_vad, vadfromtext, split, hml_mode)
    collate = get_collate_fn(name, hml_mode)
    
    shuffled = True if split == 'trn' else False
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffled,
        num_workers=16, drop_last=True, collate_fn=collate
    )

    return loader