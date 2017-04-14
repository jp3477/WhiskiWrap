import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.signal
import pandas
import os
from os import path


def find_angle(group):
    group = group.apply(             
        lambda x:
            angle_between((x.tip_x - x.fol_x,  x.tip_y - x.fol_y), (1,0) ), 
        axis = 1
    ).mean()

    return group

def get_angle_over_time(data, outfile=None, frame_rate=30.0):
    """
        Plot whisker angles from dataset saved in data
    """

    angles = data.groupby('time').apply(find_angle).to_frame('angle')
    angles['time'] = angles.index

    if outfile:
        angles.to_pickle(outfile)
        print 'Saved all angle data in {}'.format(path.join(os.getcwd(), outfile))

    return angles

def plot_angle_over_time(data, savefile=None, frame_rate=30.):
    """
        Plot whisker angles from dataset saved in data
    """
    angles = []
    times = []
    gb = data.groupby('time')

    dfs_by_second = [gb.get_group(x) for x in gb.groups]

    for df in dfs_by_second:
        average_angle = df.apply(
            lambda x:
                angle_between((x.tip_x - x.fol_x,  x.tip_y - x.fol_y), (1,0) ),
            axis = 1
        ).mean()

        time_point = (df.iloc[0]["time"] * frame_rate) / 1000

        times.append(time_point)
        angles.append(average_angle)

    mean = np.mean(angles)

    plt.plot(times,angles)
    plt.plot(times, np.ones(len(times)) * mean, 'r')
    plt.title('Angle behavior')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')

    if savefile:
        plt.savefig(savefile)
    else:
        plt.show()



def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    #Make sure to return negative/positive angles
    if v1_u[1] >= 0:
        return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    else:
        return -1 * np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def get_intervals(data, pickle_file, outfile=None, frame_rate=30.0):
    """
        Get average whisker angles at regular intervals determined by
        data in pickle_file

        data: Dataframe with whisker trace data
        pickle_file: File containing dataframe with trial times
        frame_rate: Frame rate of the video

    """
    angles_by_frame = get_angle_over_time(data, outfile='all_angles', frame_rate=frame_rate)
    rwin_vbase_times = pandas.read_pickle(pickle_file)['rwin_time_vbase']

    #max_time = data.iloc[-1].time / frame_rate
    max_time = max(data.time) / frame_rate

    i = 0
    total_times =  np.array([])
    total_angles = np.array([])


    while i < rwin_vbase_times.size and (rwin_vbase_times).iloc[i] < max_time:

        print 'Max Time: {}, Current Time: {}'.format(max_time, rwin_vbase_times.iloc[i])
        vbase_time = rwin_vbase_times.iloc[i]
        print vbase_time, max_time


        #Create a interval that will always stay the same length of 5 seconds
        start = int((vbase_time - 2) * frame_rate)
        interval = (start, start + (frame_rate * 5))


        chunk = angles_by_frame[(angles_by_frame.time >= interval[0]) & (angles_by_frame.time < interval[1])]

        times, angles = np.array(chunk.time / frame_rate), np.array(chunk.angle)

        # print "Chunk times: {}".format(interval[0])


        #Convert times back to frames, and subtract start time from all, and round to nearest integer
        normalized_times = np.rint(((times - min(times)) * frame_rate))

        total_times = np.concatenate((total_times, normalized_times))

        total_angles = np.concatenate((total_angles, angles))
        i += 1


    #Sort the times
    sortidx = np.argsort(total_times)
    sorted_times = total_times[sortidx]

    #Extract unique times and corresponding angles
    #See http://stackoverflow.com/questions/31878240/numpy-average-of-values-corresponding-to-unique-coordinate-positions
    unqID_mask = np.append(True, np.diff(sorted_times, axis=0)).astype(bool)
    ID = unqID_mask.cumsum() - 1

    unique_times = sorted_times[unqID_mask] / frame_rate
    average_angles = np.bincount(ID, total_angles[sortidx]) / np.bincount(ID)

    #Plot average angles at each unique time point


    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.plot(unique_times, average_angles)
    ax.axvline(x=2, color='r', ls='--')
    ax.set_title('Whisker Angle Near Trial Start')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_xlim(0, 5)
    fig.savefig('angle_plot')


    if outfile:
        trial_angles = pandas.DataFrame(data={'time':unique_times, 'average_angle':average_angles})
        trial_angles.to_pickle('trial_angles')
        print 'Saved angle data in {}'.format(path.join(os.getcwd(), 'trial_angles'))

    
    print 'Saved angle plot in {}'.format(path.join(os.getcwd(), 'angle_plot'))
    return unique_times, average_angles


def hilbert_transform(data):
    analytic_signal = scipy.signal.hilbert(data)
