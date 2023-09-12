import numpy as np
import utils.rotation_conversions as geometry
import bvhsdk
from scipy.signal import savgol_filter


def get_indexes(dataset):
    n_joints = 83
    if dataset == 'genea2023':
        idx_positions = np.asarray([ [i*6+3, i*6+4, i*6+5] for i in range(n_joints) ]).flatten()
        idx_rotations = np.asarray([ [i*6, i*6+1, i*6+2] for i in range(n_joints) ]).flatten()
    elif dataset == 'genea2023+':
        idx_positions = np.asarray([ [i*9+6, i*9+7, i*9+8] for i in range(n_joints) ]).flatten()
        idx_rotations = np.asarray([ [i*9, i*9+1, i*9+2, i*9+3, i*9+4, i*9+5] for i in range(n_joints) ]).flatten()  
    else:
        raise NotImplementedError("This dataset is not implemented.")
    return idx_positions, idx_rotations

def split_pos_rot(dataset, data):
    # Split the data into positions and rotations
    # Shape expected [num_samples(bs), 1, chunk_len, 1245 or 498]
    # Output shape [num_samples(bs), 1, chunk_len, 498 or 249]
    idx_positions, idx_rotations = get_indexes(dataset)
    return data[..., idx_positions], data[..., idx_rotations]

def rot6d_to_euler(data):
    # Convert numpy array to euler angles
    # Shape expected [num_samples(bs), 1, chunk_len, 498]
    # Output shape [num_samples(bs) * chunk_len, n_joints, 3]
    n_joints = 83
    assert data.shape[-1] == n_joints*6
    sample_rot = data.view(data.shape[:-1] + (-1, 6))                   # [num_samples(bs), 1, chunk_len, n_joints, 6]
    sample_rot = geometry.rotation_6d_to_matrix(sample_rot)                         # [num_samples(bs), 1, chunk_len, n_joints, 3, 3]
    sample_rot = geometry.matrix_to_euler_angles(sample_rot, "ZXY")[..., [1, 2, 0] ]*180/np.pi # [num_samples(bs), 1, chunk_len, n_joints, 3]
    sample_rot = sample_rot.view(-1, *sample_rot.shape[2:]).permute(0, 2, 3, 1).squeeze()    # [num_samples(bs), n_joints, 3, chunk_len]
    return sample_rot

def tobvh(bvhreference, rotation, position=None):
    # Converts to bvh format
    # Shape expected  [njoints, 3, frames]
    # returns a bvh object
    rotation = rotation.transpose(2, 0, 1) # [frames, njoints, 3]
    bvhreference.frames = rotation.shape[0]
    for j, joint in enumerate(bvhreference.getlistofjoints()):
        joint.rotation = rotation[:, j, :]
        joint.translation = np.tile(joint.offset, (bvhreference.frames, 1))
    if position.any():
        position = position.transpose(2, 0, 1) # [frames, njoints, 3]
        bvhreference.root.translation = position[:, 0, :]
    return bvhreference

def posfrombvh(bvh):
    # Extracts positions from bvh
    # returns a numpy array shaped [frames, njoints, 3]
    position = np.zeros((bvh.frames, len(bvh.getlistofjoints()) * 3))
    # This way takes advantage of the implementarion of getPosition (16.9 seconds ~4000 frames)
    for frame in range(bvh.frames):
        for i, joint in enumerate(bvh.getlistofjoints()):
            position[frame, i*3:i*3+3] = joint.getPosition(frame)
    return position


def filter_and_interp(rotation, position, num_frames=120, chunks=None):
    # Smooth chunk transitions
    # 
    n_chunks = chunks if chunks else int(rotation.shape[-1]/num_frames)
    inter_range = 10 #interpolation range in frames
    for transition in np.arange(1, n_chunks-1)*num_frames:
        position[..., transition:transition+2] = np.tile(np.expand_dims(position[..., transition]/2 + position[..., transition-1]/2,-1),2)
        rotation[..., transition:transition+2] = np.tile(np.expand_dims(rotation[..., transition]/2 + rotation[..., transition-1]/2,-1),2)
        for i, s in enumerate(np.linspace(0, 1, inter_range-1)):
            forward = transition-inter_range+i
            backward = transition+inter_range-i
            position[..., forward] = position[..., forward]*(1-s) + position[:, :, :, transition-1]*s  
            position[..., backward] = position[..., backward]*(1-s) + position[:, :, :, transition]*s
            rotation[..., forward] = rotation[..., forward]*(1-s) + rotation[:, :, :, transition-1]*s
            rotation[..., backward] = rotation[..., backward]*(1-s) + rotation[:, :, :, transition]*s
            
    position = savgol_filter(position, 9, 3, axis=-1)
    rotation = savgol_filter(rotation, 9, 3, axis=-1)

    return position, rotation

def np_matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """
    Same as utils.rotation_conversions.matrix_to_rotation_6d but for numpy arrays.
    """
    return matrix[..., :2, :].copy().reshape(6)

def bvh2representations2(anim: bvhsdk.Animation):
    # Converts bvh to two representations: 6d rotations and 3d positions 
    # And 3d rotations (euler angles) and 3d positions
    # The 3d positions of both representations are the same (duplicated data)
    # This representation is used in the genea challenge
    njoints = len(anim.getlistofjoints())
    npyrot6dpos = np.empty(shape=(anim.frames, 9*njoints))
    npyrotpos = np.empty(shape=(anim.frames, 6*njoints))
    for i, joint in enumerate(anim.getlistofjoints()):
        npyrot6dpos[:,i*9:i*9+6] = [ np_matrix_to_rotation_6d(joint.getLocalTransform(frame)[:-1,:-1]) for frame in range(anim.frames) ]
        npyrotpos[:,i*6:i*6+3] = [ joint.rotation[frame] for frame in range(anim.frames) ]

    for frame in range(anim.frames):
        for i, joint in enumerate(anim.getlistofjoints()):
            pos = joint.getPosition(frame)
            npyrot6dpos[frame, i*9+6:i*9+9] = pos
            npyrotpos[frame, i*6+3:i*6+6] = pos
    
    return npyrot6dpos, npyrotpos

def bvh2representations1(anim: bvhsdk.Animation):
    # Converts bvh to three representations: 6d rotations, 3d positions (euler angles) and 3d positions
    njoints = len(anim.getlistofjoints())
    npyrot6d = np.empty(shape=(anim.frames, 6*njoints))
    npyrot = np.empty(shape=(anim.frames, 3*njoints))
    npypos = np.empty(shape=(anim.frames, 3*njoints))
    for i, joint in enumerate(anim.getlistofjoints()):
        npyrot6d[:,i*6:i*6+6] = [ np_matrix_to_rotation_6d(joint.getLocalTransform(frame)[:-1,:-1]) for frame in range(anim.frames) ]
        npyrot[:,i*3:i*3+3] = [ joint.rotation[frame] for frame in range(anim.frames) ]

    for frame in range(anim.frames):
        for i, joint in enumerate(anim.getlistofjoints()):
            npypos[frame, i*3:i*3+3] = joint.getPosition(frame)
    
    return npyrot6d, npyrot, npypos