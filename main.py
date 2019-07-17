import utility
import take_io

import logging
import numpy as np
import quaternion

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)


def quaternion_to_euler(q):
    qv = quaternion.as_float_array(q)

    p0 = qv[:, 0]
    p1 = qv[:, 1]
    p2 = qv[:, 2]
    p3 = qv[:, 3]

    # Theta 2, test for singularity.
    t2 = 2 * (p0 * p2 + p1 * p3)

    t1_num = 2 * (p0 * p1 - p2 * p3)
    t1_den = 1 - 2 * (p1 * p1 + p2 * p2)

    t2_num = 2 * (p0 * p3 - p1 * p2)
    t2_den = 1 - 2 * (p2 * p2 + p3 * p3)

    rx = np.arctan2(t1_num, t1_den)
    ry = np.arcsin(t2)
    rz = np.arctan2(t2_num, t2_den)

    return np.column_stack((rx, ry, rz))


def take_to_bob_text(prefix):
    timer = utility.Timer()

    logger.info('loading take from "{prefix}": {delta:.4f}'.format(
        prefix=prefix, delta=timer.elapsed()))

    info, node_list, data = take_io.read('{}/data.mStream'.format(prefix))

    logger.info('read take stream: {delta:.4f}'.format(delta=timer.elapsed()))

    #
    # Read the mTake file to get the node string ids. Create a name
    # mapping for all of the nodes and the data fields that are present in the
    # take data stream. For example, create something like:
    #
    #   node_map['Hips']['Gq'] = (0, 4)
    #
    # Which can be used to index into the big array of data loaded from the
    # take.
    #
    node_map = take_io.make_node_map('{}/take.mTake'.format(prefix), node_list)

    logger.info('create named data base and bounds: {delta:.4f}'.format(
        delta=timer.elapsed()))

    #
    # The take stream data is arranged as a flat 1D array.
    #   [ax0 ay0 az0 ... axN ayN azN]
    # Use the reshape function access as 2D array with rows as time.
    #   [[ax0 ay0 az0],
    #    ...
    #    [axN ayN azN]]
    #
    stride = int(info['frame_stride'] / 4)
    num_frame = int(info['num_frame'])

    y = np.reshape(np.array(data), (num_frame, stride))

    # Time in seconds.
    h = info.get('h', 0.01)
    x = np.arange(0, (num_frame - 1) * h, h)
    # x = y[:, range(*node_map['Body']['timestamp'])].flatten()
    # x = x - x[0]

    logger.info('copy data to numpy.array x and y: {delta:.4f}'.format(
        delta=timer.elapsed()))

    # Name mapping from Shadow to BoB joint
    shadow_to_bob = {
        'Hips': 'pelvis',
        'Chest': 'lumbar_joint',  # thorax_l1
        'Head': 'neck_joint',  # skull_c1
        'LeftThigh': 'left_hip',
        'LeftLeg': 'left_knee',
        'LeftFoot': 'left_ankle',
        'LeftShoulder': 'left_sc_joint',
        'LeftArm': 'left_shoulder',
        'LeftForearm': 'left_elbow',
        'LeftHand': 'left_wrist',
        'RightThigh': 'right_hip',
        'RightLeg': 'right_knee',
        'RightFoot': 'right_ankle',
        'RightShoulder': 'right_sc_joint',
        'RightArm': 'right_shoulder',
        'RightForearm': 'right_elbow',
        'RightHand': 'right_wrist'
    }

    # Rotation of each BoB joint in the skeleton definition. Used to change
    # the rotation coordinate system of the Shadow data as we copy it to the
    # BoB skeleton.
    root_2_over_2 = np.sqrt(2) / 2
    pre_rotate = {
        'LeftLeg': [0, 1, 0, 0],
        'LeftFoot': [root_2_over_2, root_2_over_2, 0, 0],
        'RightLeg': [0, 1, 0, 0],
        'RightFoot': [root_2_over_2, root_2_over_2, 0, 0],
        'Chest': [0.9588, -0.2840, 0, 0],
        'Head': [0.9981, 0.0610, 0, 0],
        'LeftShoulder': [0.2179, -0.6727, -0.6942, 0.1342],
        'LeftArm': [0.0062,  0.7071, 0.7071, -0.0062],
        'LeftForearm': [0.0062,  0.7071, 0.7071, -0.0062],
        'LeftHand': [0.0062,  0.7071, 0.7071, -0.0062],
        'RightShoulder': [0.6942, -0.1342, -0.2179, 0.6727],
        'RightArm': [0.0062, 0.7071, -0.7071, 0.0062],
        'RightForearm': [0.0062, 0.7071, -0.7071, 0.0062],
        'RightHand': [0.0062, 0.7071, -0.7071, 0.0062]
    }

    # Name and index ordering to convert take data to BoB text format.
    channel_order = {
        'x': 2,
        'y': 0,
        'z': 1
    }

    # Store data as named channels.
    # data['pelvis']['rotx'] = {...}
    data = {}

    for key in shadow_to_bob:
        name = shadow_to_bob[key]
        joint = {}

        if key in pre_rotate:
            pre = quaternion.as_quat_array(
                np.full((num_frame, 4), pre_rotate[key]))

            # Local quaternion. In the joint coordinate frame.
            Lq = quaternion.as_quat_array(y[:, range(*node_map[key]['Lq'])])
            # Change rotation frame to match BoB skeleton.
            Lq = pre * Lq * pre.conjugate()
            # Convert to X-Y-Z Euler angle set.
            rot = np.rad2deg(quaternion_to_euler(Lq))
        else:
            # No need to change coordinate frame. Use the X-Y-Z Euler angle set
            # directly from the Shadow skeleton.
            rot = np.rad2deg(y[:, range(*node_map[key]['r'])])

        for channel_key in channel_order:
            channel = 'rot{}'.format(channel_key)
            joint[channel] = rot[:, channel_order[channel_key]]

        if key == 'Hips':
            # World space translation. Shadow is in cm, BoB in m.
            trans = y[:, range(*node_map[key]['c'])] * 0.01
            for channel_key in channel_order:
                channel = 'trans{}'.format(channel_key)
                joint[channel] = trans[:, 1 + channel_order[channel_key]]

        data[name] = joint

    logger.info('convert rotate and translate data: {delta:.4f}'.format(
        delta=timer.elapsed()))

    #
    # Write out data in plain text format.
    #
    time_str = ' '.join(map(str, x))
    with open('{}/data.txt'.format(prefix), 'w') as f:
        for name in data:
            f.write('% {}\n'.format(name))

            f.write('{}.time=[{}];\n'.format(name, time_str))

            for channel in data[name]:
                f.write('{}.{}=[{}];\n'.format(
                    name, channel, ' '.join(map(str, data[name][channel]))))

            f.write('\n')

    logger.info('total time: {delta:.4f}'.format(
        delta=timer.total()))


def main(path=None):
    path = take_io.find_newest_take(path)

    take_to_bob_text(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert a Shadow take to BoB text format')

    parser.add_argument('path', nargs='*')

    args = parser.parse_args()

    if len(args.path) == 0:
        main()
    else:
        for path in args.path:
            main(path)
