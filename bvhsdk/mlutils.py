import numpy as np
import bvhsdk
import os
import scipy.io.wavfile as wav
import math
import librosa


def bvh2npy(path="F:\\Downloads\\Trinity\\FBXs\\", skiplegs=True, skiphands=False):
    to_skip = []
    if skiplegs:
        to_skip += ['RightUpLeg', 'LeftUpLeg']
    if skiphands:
        to_skip += ["RightHandThumb1", "RightHandIndex1", "RightHandMiddle1",
                    "RightHandRing1", "RightHandPinky1", "LeftHandThumb1",
                    "LeftHandIndex1", "LeftHandMiddle1", "LeftHandRing1",
                    "LeftHandPinky1"]
    np_path = os.path.join(path, 'Numpy\\')
    for root, directories, files in os.walk(path):
        for file in files:
            if '.bvh' in file:
                file_path = os.path.join(path, file)
                file_name = file[:-4]
                animation = bvhsdk.ReadFile(file_path)
                animation.downSample(100)
                trans, rot = animation.MLPreProcess(skipjoints=to_skip, local_rotation=True)
                np.save(os.path.join(np_path, file_name + '_trans.npy'), trans)
                np.save(os.path.join(np_path, file_name + '_rot.npy'), rot)
                print(str.format('%s processed' % file_name))


def wav2npy(path="F:\\Downloads\\Trinity\\Audio\\", fps=100):
    for root, directories, files in os.walk(path):
        for file in files:
            if '.wav' in file:
                print(file)
                file_path = os.path.join(path, file)
                file_name = file[:-4]
                rate, signal = wav.read(file_path)
                # from int16 to -1/1
                signal = signal.astype(float)/math.pow(2, 15)
                hop_len = int(rate/fps)
                n_fft = int(rate*0.13)
                C = librosa.feature.melspectrogram(y=signal, sr=rate, n_fft=2048, hop_length=hop_len, n_mels=27, fmin=0.0, fmax=8000)
                C = np.log(C)
                print("fs: " + str(rate))
                print("hop_len: " + str(hop_len))
                print("n_fft: " + str(n_fft))
                print(C.shape)
                print(np.min(C), np.max(C))
                np.save(os.path.join(path, file_name + '.npy'), np.transpose(C))
                print(str.format('%s processed' % file_name))


def wav_raw2npy(path="F:\\Downloads\\Trinity\\Audio\\", fps=100):
    for root, directories, files in os.walk(path):
        for file in files:
            if '.wav' in file:
                print(file)
                file_path = os.path.join(path, file)
                file_name = file[:-4]
                signal, sr = librosa.load(file_path, sr=100)
                print(np.min(signal), np.max(signal))
                np.save(os.path.join(path, file_name + '_raw.npy'), signal)
                print(str.format('%s processed' % file_name))


def load_data(path="F:\\Downloads\\Trinity", rotation=False, audio_path=None, mocap_path=None):
    if not audio_path:
        audio_path = os.path.join(path, "Audio\\Numpy\\")
    if not mocap_path:
        mocap_path = os.path.join(path, "FBXs\\Numpy\\")
    audios = []
    mocaps = []
    audios_test = []
    find_string = '_trans.npy'
    if rotation:
        find_string = '_rot.npy'
    #assures that the files are ordered:
    audio_files = os.listdir(audio_path)
    audio_files.sort()
    mocap_files = os.listdir(mocap_path)
    mocap_files.sort()
    for file in audio_files:
        if '_31.npy' not in file and '_29.npy' not in file:
            audios.append(np.load(os.path.join(audio_path, file)).astype(np.float32))
            #print(file, audios[-1].shape)
        if '_31.npy' in file or '_29.npy' in file:
            audios_test.append(np.load(os.path.join(audio_path, file)).astype(np.float32))
    for file in mocap_files:
        if file[-10:] == find_string or file[-8:] == find_string:
            mocaps.append(np.load(os.path.join(mocap_path, file)).astype(np.float32))
            #print(file, mocaps[-1].shape)
    return audios, mocaps, audios_test


def aligndata(np_audios, np_mocaps, fps, alignment_path="F:\\Downloads\\Trinity"):
    # times = []
    with open(os.path.join(alignment_path, 'AlignmentTimes.csv')) as file:
        for count, line in enumerate(file):
            if count != 0:
                starts = [int(float(i)*fps) for i in line.replace('\n', '').split(',')]
                # print(starts)
                i = count-1
                np_audios[i] = np.delete(np_audios[i], range(starts[1]), axis=0)
                np_mocaps[i] = np.delete(np_mocaps[i], range(starts[2]), axis=2)
                # Antes estava assim, verificar se ainda funciona:
                # minimun = min([audio[i].shape[0], mocap[i].shape[2]])
                minimun = min([np_audios[i].shape[0], np_mocaps[i].shape[2]])
                np_audios[i] = np_audios[i][:minimun]
                np_mocaps[i] = np_mocaps[i][:, :, :minimun]
                np_mocaps[i] = np.swapaxes(np.swapaxes(np_mocaps[i], 0, 2), 1, 2)
                # times.append([float(i) for i in line.replace('\n','').split(',')])
    return np_audios, np_mocaps


def normalizedata_OLD2(np_audios, np_audios_test, np_mocaps, rotation=True):
    #_OLD is oldest than _OLD2
    #Both are wrong
    means = []
    maxis = []

    for i, data in enumerate([np_audios+np_audios_test, np_mocaps]):
        mini = []
        maxi = []
        mean = []
        for m in data:
            mini.append(np.min(m))
            maxi.append(np.max(m))
            mean.append(np.mean(m))
        mini = np.min(mini)
        maxi = np.max(maxi)
        mean = np.mean(mean)
        maxi = np.max([np.abs(mini-mean), maxi-mean])
        if rotation and i == 1:
            if maxi < 180:
                maxi = 180
        means.append(mean)
        maxis.append(maxi)
        for m in data:
            m[:] = (m-mean)/maxi
    return np_audios, np_audios_test, np_mocaps, means, maxis

def normalizedata_OLD(np_audios, np_mocaps, rotation=False):
    means = []
    maxis = []
    for i, data in enumerate([np_audios, np_mocaps]):
        mini = []
        maxi = []
        mean = []
        for m in data:
            mini.append(np.min(m))
            maxi.append(np.max(m))
            mean.append(np.mean(m))
        mini = np.min(mini)
        maxi = np.max(maxi)
        mean = np.mean(mean)
        maxi = np.max([np.abs(mini-mean), maxi-mean])
        if rotation and i == 1:
            if maxi < 180:
                maxi = 180
        means.append(mean)
        maxis.append(maxi)
        for m in data:
            m[:] = (m-mean)/maxi
    return np_audios, np_mocaps, means, maxis


def concatenate(np_audios, np_audios_test, np_mocaps):
    _ms = np_mocaps[0].shape
    _as = np_audios[0].shape
    _ast = np_audios_test[0].shape

    if len(_as) > 1:
        new_np_audios = np.empty(shape=(0, _as[1]), dtype='float32')
        new_np_audios_test = np.empty(shape=(0, _ast[1]), dtype='float32')
    else:
        new_np_audios = np.empty(shape=(0), dtype='float32')
        new_np_audios_test = np.empty(shape=(0), dtype='float32')
    new_np_mocaps = np.empty(shape=(0, _ms[1], _ms[2]), dtype='float32')
    for i in range(len(np_audios)):
        new_np_audios = np.concatenate((new_np_audios, np_audios[i]))
        new_np_mocaps = np.concatenate((new_np_mocaps, np_mocaps[i]))
    for i in range(len(np_audios_test)):
        new_np_audios_test = np.concatenate((new_np_audios_test, np_audios_test[i]))
    return new_np_audios, new_np_audios_test, new_np_mocaps


def load_data_test(path="F:\\Downloads\\Trinity", mean_maxi=None):
    # LOAD NUMPY
    audio_path = os.path.join(path, "Audio\\Numpy\\")
    audios = []
    for file in os.listdir(audio_path):
        if '_31.npy' in file or '_29.npy' in file:
            audios.append(np.load(os.path.join(audio_path, file)).astype(np.float32))
    print('%i files found.' % len(audios))
    # NORMALIZE
    for data in audios:
        if not mean_maxi:
            mini = []
            maxi = []
            mean = []
            for m in data:
                mini.append(np.min(m))
                maxi.append(np.max(m))
                mean.append(np.mean(m))
            mini = np.min(mini)
            maxi = np.max(maxi)
            mean = np.mean(mean)
            maxi = np.max([np.abs(mini-mean), maxi-mean])
        else:
            mean = mean_maxi[0]
            maxi = mean_maxi[1]
        for m in data:
            m[:] = (m-mean)/maxi
    # CONCATENATE
    new_np_audios = np.empty(shape=(0, 27), dtype='float32')
    for i in range(len(audios)):
        new_np_audios = np.concatenate((new_np_audios, audios[i]))
    return new_np_audios


def predicted_rotations_to_bvh(base_animation, predicted, norm_maxi = 180, norm_mean = 0):
    frames, total_joints = predicted.shape[0], int(predicted.shape[1]/3)
    base_animation.expandFrames(frames, set_empty=True)
    joints = [[predicted[:,i]*norm_maxi + norm_mean, predicted[:,i+1]*norm_maxi + norm_mean, predicted[:,i+2]*norm_maxi + norm_mean] for i in range(0,len(predicted[0]), 3)]
    #joints = [[predicted[:,i]*180, predicted[:,i+1]*180, predicted[:,i+2]*180] for i in range(0,len(predicted[0]), 3)]
    joints_predicted_names = ["Hips",
                                "Spine",
                                "Spine1",
                                "Spine2",
                                "Spine3",
                                "Neck",
                                "Neck1",
                                "Head",
                                "RightShoulder",
                                "RightArm",
                                "RightForeArm",
                                "RightHand",
                                "RightHandThumb1",
                                "RightHandThumb2",
                                "RightHandThumb3",
                                "RightHandIndex1",
                                "RightHandIndex2",
                                "RightHandIndex3",
                                "RightHandMiddle1",
                                "RightHandMiddle2",
                                "RightHandMiddle3",
                                "RightHandRing1",
                                "RightHandRing2",
                                "RightHandRing3",
                                "RightHandPinky1",
                                "RightHandPinky2",
                                "RightHandPinky3",
                                "LeftShoulder",
                                "LeftArm",
                                "LeftForeArm",
                                "LeftHand",
                                "LeftHandThumb1",
                                "LeftHandThumb3",
                                "LeftHandIndex1",
                                "LeftHandIndex2",
                                "LeftHandIndex3",
                                "LeftHandMiddle1",
                                "LeftHandMiddle2",
                                "LeftHandMiddle3",
                                "LeftHandRing1",
                                "LeftHandRing2",
                                "LeftHandRing3",
                                "LeftHandPinky1",
                                "LeftHandPinky2",
                                "LeftHandPinky3"]
    joints_lowerbody_names = ["RightUpLeg",
                                "RightLeg",
                                "RightFoot",
                                "RightForeFoot",
                                "RightToeBase",
                                "LeftUpLeg",
                                "LeftLeg",
                                "LeftFoot",
                                "LeftForeFoot",
                                "LeftToeBase"]
    for joint_number, joint in enumerate(base_animation.getlistofjoints()):
        if joint_number < total_joints:
            cp = joints[joint_number]
            for frame in range(frames):
                array = np.asarray([cp[0][frame],cp[1][frame],cp[2][frame]])
                joint.setRotation(frame, array)
                joint.setTranslation(frame, joint.offset)
        else:
            for frame in range(frames):
                joint.setRotation(frame, np.asarray([0,0,0]))
                joint.setTranslation(frame, joint.offset)
    return base_animation


def predicted_rotations_to_bvh2(base_animation, predicted, norm_maxi = 180, norm_mean = 0):
    frames, total_joints = predicted.shape[0], int(predicted.shape[1]/3)
    base_animation.expandFrames(frames, set_empty=True)
    joints = [[predicted[:,i]*norm_maxi + norm_mean, predicted[:,i+1]*norm_maxi + norm_mean, predicted[:,i+2]*norm_maxi + norm_mean] for i in range(0,len(predicted[0]), 3)]
    print(len(joints))
    print(len(joints[0]))
    joints_predicted_names = ["Hips",
                                "Spine",
                                "Spine1",
                                "Spine2",
                                "Spine3",
                                "Neck",
                                "Neck1",
                                "Head",
                                "RightShoulder",
                                "RightArm",
                                "RightForeArm",
                                "RightHand",
                                "LeftShoulder",
                                "LeftArm",
                                "LeftForeArm",
                                "LeftHand"]
    for joint_number, joint in enumerate(base_animation.getlistofjoints()):
        try:
            joint_index = joints_predicted_names.index(joint.name)
            print(joint_index)
        except ValueError:
            joint_index = -1
        if joint_index >= 0:
            cp = joints[joint_index]
            print(joint.name)
            for frame in range(frames):
                array = np.asarray([cp[0][frame], cp[1][frame], cp[2][frame]])
                joint.setRotation(frame, array)
                joint.setTranslation(frame, joint.offset)
        else:
            for frame in range(frames):
                joint.setRotation(frame, np.asarray([0, 0, 0]))
                joint.setTranslation(frame, joint.offset)
    return base_animation, joints


def compute_velocity(mocaps):
    new_mocaps = []
    for i, mocap in enumerate(mocaps):
        velocities = np.asarray(mocap[1:, :, :] - mocap[:-1, :, :])
        new_mocaps.append( np.asarray( np.append(mocap[1:, :, :], velocities, axis=1) ) )
    return new_mocaps
