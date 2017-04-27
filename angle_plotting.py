import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import butter, lfilter, filtfilt
import pandas
import os
from os import path

from db_connect import Mouse, Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker



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


def get_angles_on_trial_intervals(angles_by_frame, vbase_times, max_time, savefile=None, frame_rate=30.0):
    """
        Get average whisker angles at regular intervals determined by
        data in pickle_file

        data: Dataframe with whisker trace data
        pickle_file: File containing dataframe with trial times
        frame_rate: Frame rate of the video

    """

    i = 0
    total_times =  np.array([])
    total_angles = np.array([])


    while i < vbase_times.size and (vbase_times).iloc[i] < max_time:

        vbase_time = vbase_times.iloc[i]
        print 'Max Time: {}, Current Time: {}'.format(max_time, vbase_time)


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


    if savefile:
        trial_angles = pandas.DataFrame(data={'time':unique_times, 'average_angle':average_angles})
        trial_angles.to_pickle(savefile)
        print 'Saved angle data in {}'.format(path.join(os.getcwd(), savefile))


    return unique_times, average_angles



def get_hit_and_error_angles_on_trial_intervals(angles_by_frame, pickle_file, save_data=True, save_figure=True, frame_rate=30.0):
    """
        Get average whisker angles at regular intervals determined by
        data in pickle_file

        data: Dataframe with whisker trace data
        pickle_file: File containing dataframe with trial times
        frame_rate: Frame rate of the video

    """
    #angles_by_frame = get_angle_over_time(data, outfile='all_angles', frame_rate=frame_rate)
    trials = pandas.read_pickle(pickle_file)
    rwin_vbase_times = trials['rwin_time_vbase']

    random_vbase_times = rwin_vbase_times[trials['isrnd'] == True]

    hit_vbase_times = random_vbase_times[trials['outcome'] == 'hit']
    error_vbase_times = random_vbase_times[trials['outcome'] == 'error']

    hit_pcnt = round(len(hit_vbase_times) / float(len(hit_vbase_times) + len(error_vbase_times)), 2)
    error_pcnt = 1 - hit_pcnt 

    #max_time = data.iloc[-1].time / frame_rate
    max_time = max(angles_by_frame.time) / frame_rate

    hits_savefile, errors_savefile = None, None
    if save_data:
        hits_savefile = 'hit_angles.pickle'
        errors_savefile = 'error_angles.pickle'


    print "Averaging hit angles over trials"
    hit_times, hit_angles = get_angles_on_trial_intervals(angles_by_frame, hit_vbase_times, max_time, savefile=hits_savefile, frame_rate=30.0)

    print "Averaging error angles over trials"
    error_times, error_angles = get_angles_on_trial_intervals(angles_by_frame, error_vbase_times, max_time, savefile=errors_savefile, frame_rate=30.0)


    cutoff = 1.0
    hit_angles_smoothed = butter_lowpass_filter(hit_angles, cutoff, frame_rate)
    error_angles_smoothed = butter_lowpass_filter(error_angles, cutoff, frame_rate)


    #Plot average angles at each unique time point


    fig = plt.figure()

    ax = fig.add_subplot(111)

    hit_label = 'hits ({}%)'.format(hit_pcnt * 100)
    error_label = 'errors ({}%)'.format(error_pcnt * 100)

    ax.plot(hit_times, hit_angles, 'b-', label=hit_label)
    ax.plot(error_times, error_angles, 'r-', label=error_label)

    ax.plot(hit_times, hit_angles_smoothed, 'b-', linewidth=4, label='smoothed hits')
    ax.plot(error_times, error_angles_smoothed,'r-', linewidth=4, label='smoothed errors')

    ax.axvline(x=2, color='k', ls='--')
    ax.set_title('Whisker Angle Near Trial Start')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_xlim(0, 5)
    ax.legend(loc='lower right')

    angle_savefile = 'angle_plot'
    if save_figure:
        fig.savefig(angle_savefile)
        print 'Saved angle plot in {}'.format(path.join(os.getcwd(), angle_savefile))


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b,a = butter(order, normal_cutoff, btype='low', analog=False)
    return b,a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b,a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def smooth_trial_angle(dataframe, cutoff, fs):
    time = dataframe.time
    angles = dataframe.average_angle
    filtered = butter_lowpass_filter(angles, cutoff, fs)

    plt.plot(time, angles, 'b-', label='data')
    plt.plot(time, filtered, 'g-', linewidth=4, label='filtered data')
    plt.xlabel('Time (s)')
    plt.ylabel('Mean angle (degrees)')
    plt.grid()
    plt.legend()
    plt.show()


def hilbert_transform(data):
    analytic_signal = scipy.signal.hilbert(data)


def plot_sessions_by_mouse(mouse_name):
    engine = create_engine('postgresql://jason:password@localhost/trace_db', echo=True)
    Sess = sessionmaker(bind=engine)
    session = Sess()

    mouse = session.query(Mouse).filter(Mouse.name == mouse_name).first()

    sessions = mouse.sessions

    print sessions

