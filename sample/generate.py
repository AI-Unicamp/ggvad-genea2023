# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
import shutil
from data_loaders.tensors import gg_collate
import bvhsdk
import utils.rotation_conversions as geometry
from scipy.signal import savgol_filter

def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    if args.dataset in ['genea2023', 'genea2023+']:
        fps = 30
        n_joints = 83
        #TODO: change to receive args.bvh_reference_file
        bvhreference = bvhsdk.ReadFile('./dataset/Genea2023/trn/main-agent/bvh/trn_2023_v0_000_main-agent.bvh', skipmotion=True)
    else:
        raise NotImplementedError
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    # Hard-coded takes to be generated
    num_samples = 70
    takes_to_generate = np.arange(num_samples)
    

    args.batch_size = num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    #inputs_i = [155,271,320,400,500,600,700,800,1145,1185]

    print('Loading dataset...')
    data = load_dataset(args, num_samples)

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    model.to(dist_util.dev())
    model.eval()  # disable random masking

    all_motions = [] #np.zeros(shape=(num_samples, n_joints, 3, args.num_frames*chunks_per_take))
    all_motions_rot = []
    all_lengths = []
    all_text = []
    all_audios = []
    
    # dummy motion, batch_text, window, batch_audio, batch_audio_rep, dummy seed_poses, max_length
    dummy_motion, data_text, chunk_len, data_audio, data_audio_rep, dummy_seed, max_length, vad_vals, takenames  = data.dataset.gettestbatch(num_samples)

    chunks_per_take = int(max_length/chunk_len)
    for chunk in range(chunks_per_take): # Motion is generated in chunks, for each chunk we load the corresponding data
        empty = np.array([])
        inputs = []
        for take in range(num_samples): # For each take we will load the current chunk
            vad = vad_vals[take][..., chunk:chunk+chunk_len] if args.use_vad else empty
            inputs.append((empty, data_text[take][chunk], chunk_len, empty, data_audio_rep[take][..., chunk:chunk+chunk_len], dummy_seed, vad, takenames[take]))

        _, model_kwargs = gg_collate(inputs) # gt_motion: [num_samples(bs), njoints, 1, chunk_len]
        model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()} #seed: [num_samples(bs), njoints, 1, seed_len]

        if chunk == 0: 
            pass #send mean pose
        else:
            model_kwargs['y']['seed'] = sample_out[...,-args.seed_poses:]
            


        print('### Sampling chunk {} of {}'.format(chunk+1, int(max_length/chunk_len)))

        # add CFG scale to batch
        if args.guidance_param != 1: # default 2.5
            model_kwargs['y']['scale'] = torch.ones(num_samples, device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop

        sample_out = sample_fn(
            model,
            (num_samples, model.njoints, model.nfeats, args.num_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        ) # [num_samples(bs), njoints, 1, chunk_len]

        sample = data.dataset.inv_transform(sample_out.cpu().permute(0, 2, 3, 1)).float() # [num_samples(bs), 1, chunk_len, njoints]


        # Separating positions and rotations
        if args.dataset == 'genea2023':
            idx_positions = np.asarray([ [i*6+3, i*6+4, i*6+5] for i in range(n_joints) ]).flatten()
            idx_rotations = np.asarray([ [i*6, i*6+1, i*6+2] for i in range(n_joints) ]).flatten()
            sample, sample_rot = sample[..., idx_positions], sample[..., idx_rotations]

            #rotations
            sample_rot = sample_rot.view(sample_rot.shape[:-1] + (-1, 3))
            sample_rot = sample_rot.view(-1, *sample_rot.shape[2:]).permute(0, 2, 3, 1)


        elif args.dataset == 'genea2023+':
            idx_rotations = np.asarray([ [i*9, i*9+1, i*9+2, i*9+3, i*9+4, i*9+5] for i in range(n_joints) ]).flatten()
            idx_positions = np.asarray([ [i*9+6, i*9+7, i*9+8] for i in range(n_joints) ]).flatten()
            sample, sample_rot = sample[..., idx_positions], sample[..., idx_rotations] # sample_rot: [num_samples(bs), 1, chunk_len, n_joints*6]
            
            #rotations
            sample_rot = sample_rot.view(sample_rot.shape[:-1] + (-1, 6)) # [num_samples(bs), 1, chunk_len, n_joints, 6]
            sample_rot = geometry.rotation_6d_to_matrix(sample_rot) # [num_samples(bs), 1, chunk_len, n_joints, 3, 3]
            sample_rot = geometry.matrix_to_euler_angles(sample_rot, "ZXY")[..., [1, 2, 0] ]*180/np.pi # [num_samples(bs), 1, chunk_len, n_joints, 3]
            sample_rot = sample_rot.view(-1, *sample_rot.shape[2:]).permute(0, 2, 3, 1) # [num_samples(bs)*chunk_len, n_joints, 3]
            
        else:
            raise ValueError(f'Unknown dataset: {args.dataset}')

        #positions
        sample = sample.view(sample.shape[:-1] + (-1, 3))                           # [num_samples(bs), 1, chunk_len, n_joints/3, 3]
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)             # [num_samples(bs), n_joints/3, 3, chunk_len]

        text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
        all_text += model_kwargs['y'][text_key]
        
        all_audios.append(model_kwargs['y']['audio'].cpu().numpy())
        all_motions.append(sample.cpu().numpy())
        all_motions_rot.append(sample_rot.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())



    all_audios = data_audio
    all_motions = np.concatenate(all_motions, axis=3)
    all_motions_rot = np.concatenate(all_motions_rot, axis=3)
    all_lengths = np.concatenate(all_lengths, axis=0)

    # Smooth chunk transitions
    inter_range = 10 #interpolation range in frames
    for transition in np.arange(1, chunks_per_take-1)*args.num_frames:
        all_motions[..., transition:transition+2] = np.tile(np.expand_dims(all_motions[..., transition]/2 + all_motions[..., transition-1]/2,-1),2)
        all_motions_rot[..., transition:transition+2] = np.tile(np.expand_dims(all_motions_rot[..., transition]/2 + all_motions_rot[..., transition-1]/2,-1),2)
        for i, s in enumerate(np.linspace(0, 1, inter_range-1)):
            forward = transition-inter_range+i
            backward = transition+inter_range-i
            all_motions[..., forward] = all_motions[..., forward]*(1-s) + all_motions[:, :, :, transition-1]*s  
            all_motions[..., backward] = all_motions[..., backward]*(1-s) + all_motions[:, :, :, transition]*s
            all_motions_rot[..., forward] = all_motions_rot[..., forward]*(1-s) + all_motions_rot[:, :, :, transition-1]*s
            all_motions_rot[..., backward] = all_motions_rot[..., backward]*(1-s) + all_motions_rot[:, :, :, transition]*s
            
    all_motions = savgol_filter(all_motions, 9, 3, axis=-1)
    all_motions_rot = savgol_filter(all_motions_rot, 9, 3, axis=-1)    

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)
    print(f"saving results to [{out_path}]")

    npy_path = os.path.join(out_path, 'results.npy')
    
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': len(takes_to_generate), 'num_chunks': chunks_per_take})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))


    for i, take in enumerate(takes_to_generate):
        final_frame = data.dataset.frames[i]
        save_file = data.dataset.takes[take][0]
        print('Saving take {}: {}'.format(i, save_file))
        positions = all_motions[i]
        positions = positions[..., :final_frame]
        positions = positions.transpose(2, 0, 1)

        # Saving generated motion as bvh file
        rotations = all_motions_rot[i] # [njoints/3, 3, chunk_len*chunks]
        rotations = rotations[..., :final_frame]
        rotations = rotations.transpose(2, 0, 1) # [chunk_len*chunks, njoints/3, 3]
        bvhreference.frames = rotations.shape[0]
        for j, joint in enumerate(bvhreference.getlistofjoints()):
            joint.rotation = rotations[:, j, :]
            joint.translation = np.tile(joint.offset, (bvhreference.frames, 1))
        bvhreference.root.translation = positions[:, 0, :]
        #bvhreference.root.children[0].translation = positions[:, 1, :]
        print('Saving bvh file...')
        bvhsdk.WriteBVH(bvhreference, path=out_path, name=save_file, frametime=1/fps, refTPose=False)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


def load_dataset(args, batch_size):
    data = get_dataset_loader(name=args.dataset,
                              data_dir=args.data_dir,
                              batch_size=batch_size,
                              num_frames=args.num_frames,
                              split='tst',
                              hml_mode='text_only',
                              step = args.num_frames,
                              use_wavlm=args.use_wavlm,
                              use_vad = args.use_vad,
                              vadfromtext = args.vadfromtext,)
    #data.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
