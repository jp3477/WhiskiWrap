import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.signal
import pandas



def get_angle_over_time(data, frame_rate=30):
    """
        Plot whisker angles from dataset saved in data
    """
    data=data[:200]
    angles = []
    times = []
    gb = data.groupby('time')

    dfs_by_frame = [gb.get_group(x) for x in gb.groups]


    for i in range(len(dfs_by_frame)):
        df = dfs_by_frame[i]
        # print df.time
        average_angle = df.apply(
            lambda x:
                angle_between((x.tip_x - x.fol_x,  x.tip_y - x.fol_y), (1,0) ),
            axis = 1
        ).mean()

        # time_point = (df.iloc[0]["time"] / frame_rate)

        # times.append(time_point)
        # angles.append(average_angle)

        dfs_by_frame[i]['angle'] = average_angle







    # mean = np.mean(angles)

    # times = np.array(times)
    # angles = np.array(angles)

    # data_to_plot = np.array([times, angles])
    # idx = np.argsort(data_to_plot[0])
    # data_to_plot = data_to_plot[:, idx]

    # times, angles = data_to_plot[0, :], data_to_plot[1, :]

    pandas.concat(dfs_by_frame)
    data = pandas.concat(dfs_by_frame)

    return data

def plot_angle_over_time(data, savefile=None, frame_rate=30,):
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
    # times, angles = reject_outliers(np.array(times), np.array(angles))

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

def reject_outliers(xdata, ydata, m = 3.):
    """ Remove outliers from a set of data
        m: number of standard deviations away
    """
    d = np.abs(ydata - np.median(ydata))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return xdata[s<m], ydata[s<m]

def get_intervals(data, pickle_file, frame_rate=30.0):
    """
        Get average whisker angles at regular intervals determined by
        data in pickle_file

        data: Dataframe with whisker trace data
        pickle_file: File containing dataframe with trial times
        frame_rate: Frame rate of the video

    """
    rwin_vbase_times = pandas.read_pickle(pickle_file)['rwin_time_vbase']
    max_time = data.iloc[-1].time / frame_rate

    i = 0

    trials = []
    total_times =  np.array([])
    total_angles = np.array([])

    while i < rwin_vbase_times.size and (rwin_vbase_times).iloc[i] < max_time:

        print 'Max Time: {}, Current Time: {}'.format(max_time, rwin_vbase_times.iloc[i])
        vbase_time = rwin_vbase_times.iloc[i]

        #Create a interval that will always stay the same length of 5 seconds
        start = int((vbase_time - 2) * frame_rate)
        interval = (start, start + (frame_rate * 5))
        # print interval
        chunk = data[(data.time >= interval[0]) & (data.time < interval[1])]

        times, angles = get_angle_over_time(chunk, frame_rate=frame_rate)

        normalized_times = np.rint(((times - times[0]) * frame_rate))


        # trials.append(dict(zip(normalized_times, angles)))
        # trials.append(zip(normalized_times, angles))

        total_times = np.concatenate((total_times, normalized_times))

        total_angles = np.concatenate((total_angles, angles))
        i += 1


    sortidx = np.argsort(total_times)
    sorted_times = total_times[sortidx]

    unqID_mask = np.append(True, np.diff(sorted_times, axis=0)).astype(bool)

    ID = unqID_mask.cumsum() - 1

    unique_times = sorted_times[unqID_mask]
    average_angles = np.bincount(ID, total_angles[sortidx]) / np.bincount(ID)

    plt.plot(unique_times / frame_rate, average_angles)
    plt.axvline(x=2, color='r', ls='--')
    plt.plot()
    plt.title('Angle behavior')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.xlim(0, 5)
    plt.savefig('angle_plot')

    print 'Saved angle plot in {}'.format(path.join(os.getcwd(), 'angle_plot'))
    return unique_times, average_angles


def hilbert_transform(data):
    analytic_signal = scipy.signal.hilbert(data)