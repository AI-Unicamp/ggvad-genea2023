import glob

from pymo.parsers import BVHParser
from pymo.preprocessing import *
from sklearn.pipeline import Pipeline


def convert_bvh(bvhfile):
    parser = BVHParser()
    parsed_data = parser.parse(bvhfile)

    # use a subset of joints
    target_joints = ['body_world', 'b_root', 'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_l_pinky1', 'b_l_pinky2', 'b_l_pinky3', 'b_l_ring1', 'b_l_ring2', 'b_l_ring3', 'b_l_middle1', 'b_l_middle2', 'b_l_middle3', 'b_l_index1', 'b_l_index2', 'b_l_index3', 'b_l_thumb0', 'b_l_thumb1', 'b_l_thumb2', 'b_l_thumb3', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_r_thumb0', 'b_r_thumb1', 'b_r_thumb2', 'b_r_thumb3', 'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3', 'b_r_middle1', 'b_r_middle2', 'b_r_middle3', 'b_r_ring1', 'b_r_ring2', 'b_r_ring3', 'b_r_index1', 'b_r_index2', 'b_r_index3', 'b_neck0', 'b_head']

    pipe = Pipeline([
        ('param', MocapParameterizer('position')),
        ('jtsel', JointSelector(target_joints, include_root=False)),
        ('np', Numpyfier()),
    ])
    pos_data = pipe.fit_transform([parsed_data])[0]

    return pos_data


if __name__ == '__main__':
    bvhfiles = glob.glob('../dataset/genea2023_dataset/trn/main-agent/bvh/*.bvh')
    # bvhfiles = glob.glob('../dataset/genea2023_dataset/val/main-agent/bvh/*.bvh')

    for bvhfile in sorted(bvhfiles):
        print(bvhfile)
        npy_data = convert_bvh(bvhfile)
        print(npy_data.shape)

        out_file = bvhfile.replace('.bvh', '.npy').replace('/bvh/', '/npy/')
        np.save(out_file, npy_data)
