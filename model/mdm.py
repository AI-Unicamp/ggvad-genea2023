import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.local_attention_diffstylegest import SinusoidalEmbeddings, apply_rotary_pos_emb
from model.local_attention_diffstylegest import LocalAttention

class MDM(nn.Module):
    def __init__(self, njoints, nfeats, pose_rep, data_rep, latent_dim=256, text_dim=64, ff_size=1024,
                  num_layers=8, num_heads=4, dropout=0.1, activation="gelu",
                 dataset='amass', clip_dim=512, clip_version=None, **kargs):
        super().__init__()
        print('Using MDM V2 (w/ CrossAtt+RPM)')

        # General Configs        
        self.dataset = dataset
        self.pose_rep = pose_rep
        self.data_rep = data_rep
        self.njoints = njoints
        self.nfeats = nfeats
        self.input_feats = self.njoints * self.nfeats
        self.latent_dim = latent_dim
        self.dropout = dropout

        # Timestep Network
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Text Encoder 
        self.use_text = kargs.get('use_text', False)
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.text_dim = text_dim
        self.clip_dim = clip_dim
        if self.use_text:
            self.embed_text = nn.Linear(self.clip_dim, self.text_dim)
            print('Using Text')
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

        # VAD
        self.use_vad = kargs.get('use_vad', False)
        if self.use_vad:
            vad_lat_dim = int(self.latent_dim)
            self.vad_lookup = nn.Embedding(2, vad_lat_dim)
            print('Using VAD')

        # Seed Pose Encoder
        self.seed_poses = kargs.get('seed_poses', 0)
        print('Using {} Seed Poses.'.format(self.seed_poses))
        if self.seed_poses > 0:
            if self.use_text:
                self.seed_pose_encoder = SeedPoseEncoder(self.njoints, self.seed_poses, self.latent_dim - self.text_dim)
            else:
                self.seed_pose_encoder = SeedPoseEncoder(self.njoints, self.seed_poses, self.latent_dim)

        # Audio Encoder
        self.mfcc_input = kargs.get('mfcc_input', False)
        self.use_wav_enc = kargs.get('use_wav_enc', False)
        self.use_wavlm = kargs.get('use_wavlm', False)
        print('Using Audio Features:')
        if self.mfcc_input:
            self.mfcc_dim = 26
            self.audio_feat_dim = 64
            self.wavlm_encoder = nn.Linear(26, self.audio_feat_dim)
            print('Selected Features: MFCCs')
        if self.use_wav_enc:
            self.wav_enc_dim = 32 
            self.audio_feat_dim = self.wav_enc_dim
            print('Selected Features: WavEncoder Representations')
            self.wav_encoder = WavEncoder()
        if self.use_wavlm:
            self.wavlm_proj_dim = 64
            self.audio_feat_dim = self.wavlm_proj_dim
            self.wavlm_encoder = nn.Linear(768, self.audio_feat_dim)
            print('Selected Features: WavLM Representations')

        # Pose Encoder
        self.input_process = InputProcess(self.data_rep, self.input_feats, self.latent_dim)

        # Cross-Local Attention
        self.cl_head=8
        if self.use_vad:
            self.project_to_lat = nn.Linear(self.latent_dim * 3 + self.audio_feat_dim, self.latent_dim)
            #self.project_to_lat = nn.Linear(vad_lat_dim + self.audio_feat_dim + self.latent_dim*2, self.latent_dim)
        else:
            self.project_to_lat = nn.Linear(self.latent_dim * 2 + self.audio_feat_dim, self.latent_dim)
        self.cross_local_attention = LocalAttention(
        #    dim=32,  # dimension of each head (you need to pass this in for relative positional encoding)
            window_size=10, 
            causal=True,  
            look_backward=1,  
            look_forward=0,     
            dropout=0.1, 
            exact_windowsize=False
        )

        # Positional Encodings
        self.rel_pos = SinusoidalEmbeddings(self.latent_dim // self.cl_head)

        # Self-Attention
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.activation = activation
        self.num_layers = num_layers
        self.seqTransEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                                                        d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation),
                                                        num_layers=self.num_layers)     

        # Project Representation to Output Pose
        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        self.log_train = False
        self.batch_log = {'text': [], 
                          'vad': [],
                          'seed': [],
                          'timestep': [], 
                          'audio': [],
                          'poses': [], 
                          'fg_embs': [], 
                          'coa_embs': [], 
                          'embs': [],
                          'audiovad': []}
        #self.log_seed = []
        #self.log_text = []
        #self.log_timestep = []
        #self.log_audio = []
        #self.log_vad = []
        #self.log_poses = []
        #self.log_fg_embs = []
        #self.log_coa_embs = []
        #self.log_embs = []

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # Sizes
        bs, njoints, nfeats, nframes = x.shape  # [BS, POSE_DIM, 1, CHUNK_LEN]
        force_mask = y.get('uncond', False)     # TODO: UNDERSTAND MASK

        #############################
        #### FEATURE CALCULATION ####
        #############################
        
        # Text Embeddings
        if self.use_text:
            enc_text = self.encode_text(y['text'])
            emb_text = self.embed_text(self.mask_cond(enc_text, force_mask=force_mask)) # [1, BS, TEXT_DIM]
            emb_text = emb_text.squeeze(0)                                              # [BS, TEXT_DIM]

        # Seed Poses Embeddings
        flat_seed = y['seed'].squeeze(2).reshape(bs, -1)        # [BS, POSE_DIM, 1, SEED_POSES] -> [BS, POSE_DIM, SEED_POSES] -> [BS, POSE_DIM * SEED_POSES] 
        #emb_seed = self.seed_pose_encoder(flat_seed)
        emb_seed = self.seed_pose_encoder(self.mask_cond(flat_seed, force_mask=force_mask)) # [BS, LAT_DIM] or [BS, LAT_DIM - TEXT_DIM]

        # VAD Embeddings
        if self.use_vad:
            vad_vals = y['vad']                                     # [BS, CHUNK_LEN]
            emb_vad = self.vad_lookup(vad_vals)                     # [BS, CHUNK_LEN, LAT_DIM]
            emb_vad = emb_vad.permute(1, 0, 2)                      # [CHUNK_LEN, BS, LAT_DIM]

        # Timesteps Embeddings
        emb_t = self.embed_timestep(timesteps)                  # [1, BS, LAT_DIM]

        # Audio Embeddings
        if self.mfcc_input:                                     # TODO: is it actually the raw mfccs? 
            emb_audio = y['audio_rep']                          # [BS, MFCC_DIM, 1, CHUNK_LEN]
            interp_reps = emb_audio.permute(0, 3, 2, 1)     # [BS, CHUNK_LEN, 1, 768]
            emb_audio = self.wavlm_encoder(interp_reps)         # [BS, CHUNK_LEN, 1, WAVLM_PROJ_DIM]
            emb_audio = emb_audio.permute(0, 3, 2, 1)         # [BS, WAVLM_PROJ_DIM, 1, CHUNK_LEN] 
        elif self.use_wav_enc:
            emb_audio = self.wav_encoder(y['audio'])            # [BS, WAV_ENC_DIM, 1, CHUNK_LEN]
            raise NotImplementedError                           # TODO: Resolve CNNs
        elif self.use_wavlm:
            interp_reps = y['audio_rep']                        # [BS, 768, 1, CHUNK_LEN]
            interp_reps = interp_reps.permute(0, 3, 2, 1)     # [BS, CHUNK_LEN, 1, 768]
            emb_audio = self.wavlm_encoder(interp_reps)         # [BS, CHUNK_LEN, 1, WAVLM_PROJ_DIM]
            emb_audio = emb_audio.permute(0, 3, 2, 1)         # [BS, WAVLM_PROJ_DIM, 1, CHUNK_LEN]     
        else:
            raise NotImplementedError
        emb_audio = emb_audio.squeeze(2)                        # [BS, AUDIO_DIM, CHUNK_LEN], (AUDIO_DIM = MFCC_DIM or WAV_ENC_DIM or WAVLM_PROJ_DIM)
        emb_audio = emb_audio.permute((2, 0, 1))                # [CHUNK_LEN, BS, AUDIO_DIM]

        # Pose Embeddings
        emb_pose = self.input_process(x)                        # [CHUNK_LEN, BS, LAT_DIM]

        #############################
        #### FEATURE AGGREGATION ####
        #############################

        # Cat Pose w/ Audio (Fine-Grained) Embeddings
        if self.use_vad:
            fg_embs = torch.cat((emb_pose, emb_audio, emb_vad), axis=2)      # [CHUNK_LEN, BS, LAT_DIM + AUDIO_DIM + LAT_DIM]
        else:
            fg_embs = torch.cat((emb_pose, emb_audio), axis=2)      # [CHUNK_LEN, BS, LAT_DIM + AUDIO_DIM]

        # Cat Seed w/ Text Embeddings (if exist)
        if self.use_text:
            embs_stxt = torch.cat((emb_text,emb_seed),axis=1)   # [BS, LAT_DIM] 
        else:
            embs_stxt = emb_seed                                # [BS, LAT_DIM] 

        # Sum All Coarse-Grained Embeddings (t + Seed w/ Text)
        coa_embs = (embs_stxt + emb_t)                          # [1, BS, LAT_DIM]
        
        # Repeat Coarse-Grained Summation (to match chunk)
        coa_embs_rep = coa_embs.repeat(nframes, 1, 1)           # [CHUNK_LEN, BS, LAT_DIM]

        # Concatenate All to form feature inputs
        embs = torch.cat((fg_embs, coa_embs_rep), axis=2)       # [CHUNK_LEN, BS, LAT_DIM + AUDIO_DIM + LAT_DIM + LAT_DIM] of 2* LAT_DIM If no VAD

        # Project to Latent Dim
        xseq = self.project_to_lat(embs)                        # [CHUNK_LEN, BS, LAT_DIM]

        ######################
        #### DENOISE PASS ####
        ######################

        ## Data Reshaping (Insert multiple att heads)
        xseq = xseq.permute(1, 0, 2)                            # [BS, CHUNK_LEN, LAT_DIM]
        xseq = xseq.view(bs, nframes, self.cl_head, -1)         # [BS, CHUNK_LEN, CL_HEAD, LAT_DIM / CL_HEAD] 
        xseq = xseq.permute(0, 2, 1, 3)                         # [BS, CL_HEAD, CHUNK_LEN, LAT_DIM / CL_HEAD]
        xseq = xseq.reshape(bs*self.cl_head, nframes, -1)       # [BS * CL_HEAD, CHUNK_LEN, LAT_DIM / CL_HEAD]

        ## RPE Embeddings
        pos_emb = self.rel_pos(xseq)                            # [CHUNK_LEN, BS] O CORRETO É [CHUNK_LEN, LAT_DIM / CL_HEAD]
        xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)     # [LAT_DIM, CHUNK_LEN, BS] O CORRETO É [BS * CL_HEAD, CHUNK_LEN, LAT_DIM / CL_HEAD]

        ## Apply Cross Local Attention
        packed_shape = [torch.Size([bs, self.cl_head])]         # [1] = [torch.Size([BS, CL_HEAD])
        mask_local = torch.ones(bs, nframes).bool().to(device=xseq.device)             # [BS, CHUNK_LEN]
        xseq = self.cross_local_attention(xseq, xseq, xseq,     
            packed_shape=packed_shape, mask=mask_local)         # [BS, CL_HEAD, CHUNK_LEN, LAT_DIM / CL_HEAD]
        
        # Data Reshaping to cat Global Information
        xseq = xseq.permute(0, 2, 1, 3)                         # [BS, CHUNK_LEN, CL_HEAD, LAT_DIM / CL_HEAD]
        xseq = xseq.reshape(bs, nframes, -1)                    # [BS, CHUNK_LEN, LAT_DIM] 
        xseq = xseq.permute(1, 0, 2)                            # [CHUNK_LEN, BS, LAT_DIM] 

        # Concat Coarse Grained Embeddings
        xseq = torch.cat((coa_embs, xseq), axis=0)              # [CHUNK_LEN+1, BS, LAT_DIM]   

        # Data Reshaping (Insert multiple att heads)
        xseq = xseq.permute(1, 0, 2)                            # [BS, CHUNK_LEN+1, LAT_DIM]
        xseq = xseq.view(bs, nframes + 1, self.cl_head, -1)     # [BS, CHUNK_LEN+1, CL_HEAD, LAT_DIM / CL_HEAD]
        xseq = xseq.permute(0, 2, 1, 3)                         # [BS, CL_HEAD, CHUNK_LEN+1, LAT_DIM / CL_HEAD]
        xseq = xseq.reshape(bs*self.cl_head, nframes + 1, -1)   # [BS * CL_HEAD, CHUNK_LEN+1, LAT_DIM / CL_HEAD]

        # RPE Embeddings
        pos_emb = self.rel_pos(xseq)                            # [CHUNK_LEN+1, BS] O CORRETO É [CHUNK_LEN+1, LAT_DIM / CL_HEAD]
        xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)     # [LAT_DIM, CHUNK_LEN+1, BS]  O CORRETO É [BS * CL_HEAD, CHUNK_LEN+1, LAT_DIM / CL_HEAD]

        # Data Reshaping
        xseq_rpe = xseq.reshape(bs,self.cl_head,nframes+1,-1)   # [BS, CL_HEAD, CHUNK_LEN+1, LAT_DIM / CL_HEAD]
        xseq = xseq_rpe.permute(0, 2, 1, 3)                     # [BS, CHUNK_LEN+1, CL_HEAD, LAT_DIM / CL_HEAD]   
        xseq = xseq.view(bs, nframes + 1, -1)                   # [BS, CHUNK_LEN+1, LAT_DIM]
        xseq = xseq.permute(1, 0, 2)                            # [CHUNK_LEN+1, BS, LAT_DIM]

        # Self-Attention
        output = self.seqTransEncoder(xseq)                     # [CHUNK_LEN+1, BS, LAT_DIM] 

        # Ignore First Token
        output = output[1:]                                     # [CHUNK_LEN, BS, LAT_DIM]

        # Linear Output Feature Pass
        output = self.output_process(output)                    # [BS, POSE_DIM, 1, CHUNK_LEN]

        if self.log_train:
            
            if self.use_text:
                mean = torch.mean(emb_text, dim=1) #emb_text: [BS, TEXT_DIM]
                self.batch_log['text'] = mean.detach().cpu().numpy()

            if self.use_vad:
                mean = torch.mean(torch.mean(emb_vad, dim=0), dim=1)  #emb_vad: [CHUNK_LEN, BS, LAT_DIM]  
                self.batch_log['vad'] = mean.detach().cpu().numpy()

            mean = torch.mean(emb_seed, dim=1)     #emb_seed: [BS, LAT_DIM - TEXT_DIM]
            self.batch_log['seed'] = mean.detach().cpu().numpy()

            mean = torch.mean(emb_t, dim=2)        #emb_t: [1, BS, LAT_DIM]
            self.batch_log['timestep'] = mean.detach().cpu().numpy()

            mean = torch.mean(torch.mean(emb_audio, dim=0), dim=1)    #emb_audio: [CHUNK_LEN, BS, AUDIO_DIM]
            self.batch_log['audio'] = mean.detach().cpu().numpy()

            mean = torch.mean(torch.mean(emb_pose, dim=0), dim=1)     #emb_pose: [CHUNK_LEN, BS, LAT_DIM]
            self.batch_log['poses'] = mean.detach().cpu().numpy()

            #mean = torch.mean(torch.mean(audiovad, dim=0), dim=1)     # [CHUNK_LEN, BS, AUDIO_DIM]
            #self.batch_log['audiovad'] = mean.detach().cpu().numpy()

            #std, mean = torch.std_mean(fg_embs, dim=1)     # fg embeddings: [CHUNK_LEN, BS, LAT_DIM + AUDIO_DIM + LAT_DIM]
            #self.log_fg_embs = [ std.detach().cpu().numpy(), mean.detach().cpu().numpy() ]
            #self.batch_log['fg_embs'] = self.log_fg_embs
#
            #std, mean = torch.std_mean(coa_embs, dim=1)      # coa embeddings: [1, BS, LAT_DIM]
            #self.log_coa_embs = [ std.detach().cpu().numpy(), mean.detach().cpu().numpy() ]
            #self.batch_log['coa_embs'] = self.log_coa_embs

            std, mean = torch.std_mean(torch.mean(embs, dim=0), dim=0)     # embeddings: [CHUNK_LEN, BS, LAT_DIM + AUDIO_DIM + LAT_DIM + LAT_DIM] of 2* LAT_DIM If no VAD
            self.log_embs = [ std.detach().cpu().numpy(), mean.detach().cpu().numpy() ]
            self.batch_log['embs'] = self.log_embs

        return output

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()
    
    #def _apply(self, fn):
    #    super()._apply(fn)
    #    self.rot2xyz.smpl_model._apply(fn)
#
    #def train(self, *args, **kwargs):
    #    super().train(*args, **kwargs)
    #    self.rot2xyz.smpl_model.train(*args, **kwargs)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)

class WavEncoder(nn.Module):
    '''
    Taken from https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context/
    '''
    def __init__(self):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=5, padding=1600, dilation = 1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(16, 32, 15, stride=5, dilation = 4),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(32, 64, 15, stride=5, dilation = 7),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(64, 32, 15, stride=5, dilation = 13),
        )

    def forward(self, wav_data):            # [B, 147000]
        wav_data = wav_data.unsqueeze(1)    # [B, 1, 147000]
        out = self.feat_extractor(wav_data) # [B, 32, 200] 
        return out.unsqueeze(2)             # [B, 32, 1, 200]
    
    def layer_output_size(self,l_in, padding, kernel_size, dilation, stride):
        l_out = int(np.floor((l_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1))
        return l_out

class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['genea_vec', 'genea_vec+']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        else:
            raise NotImplementedError

class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['genea_vec', 'genea_vec+']:
            output = self.poseFinal(output) # [CHUNK_LEN, BS, POSE_DIM]
        else:
            raise NotImplementedError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats) # [CHUNK_LEN, BS, POSE_DIM, 1]
        output = output.permute(1, 2, 3, 0)  
        return output
    
class SeedPoseEncoder(nn.Module):
    def __init__(self, njoints, seed_poses, latent_dim):
        super().__init__()
        self.njoints = njoints
        self.seed_poses = seed_poses
        self.latent_dim = latent_dim
        self.seed_embed = nn.Linear(self.njoints * self.seed_poses, self.latent_dim)

    def forward(self, x):
        x = self.seed_embed(x)
        return x