import utility
import take_io

import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)


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
    # The data is arranged as a flat 1D array.
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
    #x = y[:, range(*node_map['Body']['timestamp'])].flatten()
    #x = x - x[0]

    logger.info('copy data to numpy.array x and y: {delta:.4f}'.format(
        delta=timer.elapsed()))

    # Name mapping from Shadow to BoB joint
    shadow_to_bob = {
        'Hips': 'pelvis',
        'Chest': 'lumbar_joint',
        'Head': 'neck_joint',
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

    # Name and index ordering to convert take data to BoB text format.
    channel_order = {
        'x': 0,
        'y': 2,
        'z': 1
    }

    #
    # Write out data in plain text format.
    #
    time_str = ' '.join(map(str, x))
    with open('{}/data.txt'.format(prefix), 'w') as f:
        for key in shadow_to_bob:
            name = shadow_to_bob[key]

            f.write('% {} = {}\n'.format(key, name))

            f.write('{}.time=[{}]\n'.format(name, time_str))

            rot = np.rad2deg(y[:, range(*node_map[key]['r'])])
            for channel_key in channel_order:
                f.write('{}.rot{}=[{}]\n'.format(
                    name,
                    channel_key,
                    ' '.join(map(str, rot[:, channel_order[channel_key]].flatten()))))

            if key == 'Hips':
                trans = y[:, range(*node_map[key]['c'])] * 0.01
                for channel_key in channel_order:
                    f.write('{}.trans{}=[{}]\n'.format(
                        name,
                        channel_key,
                        ' '.join(map(str, trans[:, 1 + channel_order[channel_key]].flatten()))))

            f.write('\n')

    logger.info('total time: {delta:.4f}'.format(
        delta=timer.total()))


def main():
    prefix = take_io.find_newest_take()

    take_to_bob_text(prefix)


if __name__ == "__main__":
    main()
