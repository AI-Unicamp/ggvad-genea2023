from model.mdm import MDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps


def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])


def create_model_and_diffusion(args, data):
    model = MDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args, data):

    # default args
    clip_version = 'ViT-B/32'

    if args.dataset in ['genea2023']:
        data_rep = 'genea_vec'
        njoints = 498
        nfeats = 1
    elif args.dataset in ['genea2023+']:
        data_rep = 'genea_vec+'
        njoints = 1245
        nfeats = 1

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mask_prob': args.cond_mask_prob, 
            'clip_version': clip_version, 'dataset': args.dataset,
            'use_text': args.use_text, 'mfcc_input': args.mfcc_input, 'use_wavlm': args.use_wavlm, 'use_vad':args.use_vad,
            'use_wav_enc':args.use_wav_enc, 'seed_poses': args.seed_poses} 


def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 1000
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
    )