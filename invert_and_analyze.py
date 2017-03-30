import matplotlib
matplotlib.use('Qt4Agg')


import ffmpy
import sys , getopt
import WhiskiWrap
from os import path
import os
from datetime import date

import numpy as np
import numpy.linalg as la
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal

import tables
import pandas
import whiskvid.output_video as ov
from base import FFmpegReader
from pymediainfo import MediaInfo

from make_pickle import make_vbase_pickle_file, get_session_from_video_filename

root_dir = os.getcwd()
nas_dir = path.expanduser('~/mnt/jason_nas2_home')
master_pickle = path.expanduser('~/jason_data/sbvdf.pickle')



def invert_video(infile, outfile, time=None):
    """ This function inverts the colors in video designated by infile
        and writes the results to outfile

        infile: input video filename
        outfile: inverted output video filename
        time: time chunk of video to invert
    """

    output_args = ['-vf',
                'lutrgb=r=negval:g=negval:b=negval,eq=2:0:0:1:1:1:1:1',
                ]

    if time:
        output_args += ['-ss', '00:00:00', '-t', time]

    ff = ffmpy.FFmpeg( global_options='-y',
        inputs={infile : None},
        outputs={outfile : output_args}
    )

    ff.run()

def invert_and_trace(video, time='00:00:20', results_file='trace.hdf5'):
    """ Inverts a video's colors and then runs a trace

        video : input video filename
        outdir : directory where trace products will be placed
        results_file : dataframe that holds the geometric data of trace
    """


    video_name, ext = path.splitext(video)
    inverted_video = path.basename(video_name) + "_inverted" + ".mp4"
    invert_video(video, inverted_video, time=time)

    #Trace
    WhiskiWrap.pipeline_trace(
        inverted_video,
        results_file,
        n_trace_processes=4
    )

    return inverted_video, results_file


def overlay_video_with_results(original_video, inverted_video, whiskers_file, whiskers_table):
    """
        Overlays whisker trace data onto a video.

        inverted_video: The filename of the inverted video output by invert_video
            Used to obtain frame_rate and video dimensions
        whiskers_file: The name of the hdf5 file with saved whisker data
        whisker_table: A pandas dataframe representation of whisker data
            Holds the filtered data if necessary
    """

    handle = tables.open_file(whiskers_file)

    input_reader = FFmpegReader(original_video)

    vid_info = MediaInfo.parse(inverted_video).tracks[1]

    width = vid_info.width
    height = vid_info.height
    frame_rate = int(float(vid_info.frame_rate)) # Frames per second
    video_length = int(vid_info.duration) / 1000 #Video duration in seconds

    frame_count = frame_rate * video_length

    overlayed_video = 'overlayed.mp4'

    print "Overlaying results onto video"
    ov.write_video_with_overlays(
        overlayed_video,
        input_reader,
        width,
        height,
        whiskers_table=whiskers_table,
        whiskers_file_handle=handle,
        frame_triggers=[0],
        trigger_dstart = 0,
        trigger_dstop = frame_count
    )

    print "Saved in {}".format(path.join(os.getcwd(), overlayed_video))



class PersistentRectangleSelector(RectangleSelector):
    """
        Allows a rectangle to be drawn on a matplotlib axis
    """
    def release(self, event):
        super(PersistentRectangleSelector, self).release(event)
        self.to_draw.set_visible(True)
        self.canvas.draw()

def onselect(eclick, erelease):
    """
        Registers data from an eclick event
    """
    global startpos
    global endpos

    startpos = (eclick.xdata, eclick.ydata)
    endpos = (erelease.xdata, erelease.ydata)

    print startpos

def select_region(image_file):
    """
        Show image on axis and allow selection of a region with rectangle

        image_file: The image onto which a rectangle will be traced
    """
    global startpos
    global endpos
    ax = plt.gca()
    img = mpimg.imread(image_file)
    imgplot = ax.imshow(img, cmap='Greys_r')

    selector = PersistentRectangleSelector(ax, onselect, drawtype='box')
    plt.show()
    plt.close()

    return {'startpos' : startpos, 'endpos' : endpos}



def read_hdf5(hdf5_filename):
    """
        Reads summary data from an hdf5 file

        hdf5_filename: name of input hdf5 file
    """
    with tables.open_file(hdf5_filename) as fi:
        results = pandas.DataFrame.from_records(fi.root.summary.read())

    return results

def write_hdf5(hdf5_filename, summary):
    """
        Writes data to a new hdf5 file

        hdf5_filename: name of output hdf5 file
        summary: pandas dataframe of traced whisker data
    """
    WhiskiWrap.setup_hdf5(hdf5_filename, 1000000)
    with tables.open_file(hdf5_filename, mode='a') as hdf5file:
        table = hdf5file.get_node('/summary')
        h5seg = table.row

        for index, row in summary.iterrows():
            h5seg['chunk_start'] = row['chunk_start']
            h5seg['time'] = row['time']
            h5seg['id'] = row['id']
            h5seg['fol_x'] = row['fol_x']
            h5seg['fol_y'] = row['fol_y']
            h5seg['tip_x'] = row['tip_x']
            h5seg['tip_y'] = row['tip_y']
            h5seg['pixlen'] = row['pixlen']

            h5seg.append()


        table.flush()




def get_filtered_results_by_position(results_file, selected_positions):
    """ Filters identififed follicles by location and angle

        results_file: name of file with traced whisker data
        selected_data: a 2 x 2 array of the rectangular coordinates where whisker follicles are expected

    """

    summary = read_hdf5(results_file)

    startpos = selected_positions['startpos']
    endpos = selected_positions['endpos']
    left_limit, right_limit = min(startpos[0], endpos[0]), max(startpos[0], endpos[0])
    up_limit, down_limit = min(startpos[1], endpos[1]), max(startpos[1], endpos[1])

    filtered_summary = summary[
        (summary.fol_x > left_limit) & (summary.fol_x < right_limit) &
        (summary.fol_y > up_limit) & (summary.fol_y < down_limit) &
        (summary.fol_x < summary.tip_x)
    ]


    # indices = []

    # for index, summary_row in summary.iterrows():
    #     if (summary_row.fol_x > left_limit) & (summary_row.fol_x < right_limit) & \
    #     (summary_row.fol_y > up_limit) & (summary_row.fol_y < down_limit) & \
    #     (summary_row.fol_x < summary_row.tip_x):
    #         indices.append(index)


    # summary = summary.iloc[indices]

    # xpixels = [xpixels[i] for i in indices]
    # ypixels = [ypixels[i] for i in indices]



    # for index, x in filtered_results.iterrows():
    #     vector = (x.tip_x - x.fol_x,  x.tip_y - x.fol_y)
    #     print angle_between(vector, (1,0)) > 1

    # filtered_results = filtered_results[
    #     filtered_results.apply(
    #         lambda x:
    #             np.absolute(angle_between((x.tip_x - x.fol_x,  x.tip_y - x.fol_y), (1,0) )) > 5,
    #         axis = 1
    #     )
    # ]

    filtered_filename = 'filtered_trace.hdf5'
    # write_hdf5(filtered_filename, filtered_summary)



    return filtered_summary

def get_filtered_results_from_tiff_files(results_file):
    """
        Look at a single tiff file and trace out a region for filtering results

        results_file: filename of hdf5 file with traced whisker data

    """
    tiff_file = [ fi for fi in os.listdir('.') if fi.endswith(".tif") ][0]

    region = select_region(tiff_file)
    results = get_filtered_results_by_position(results_file, region)

    return results

def extract_frame(video_file, frame_name):
    """
        Extract first frame from a video and save it as an image

        video_file: input video filename
        frame_name: name of output image file

    """
    ff = ffmpy.FFmpeg(
            global_options='-y',
            inputs={video_file : None},
            outputs={frame_name : [
                '-r', '1',
                '-vframes', '1',

                ]}
        )
    print ff.cmd
    ff.run()


def get_desired_region_from_video(video_file):
    """
        Extract frame from a video file and trace a region of interest
    """
    frame_name = 'frame.png'
    extract_frame(video_file, frame_name)

    region = select_region(frame_name)
    os.remove(frame_name)
    return region



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

def get_angle_over_time(data, frame_rate=30):
    """
        Plot whisker angles from dataset saved in data
    """
    angles = []
    times = []
    gb = data.groupby('time')

    dfs_by_second = [gb.get_group(x) for x in gb.groups]

    for df in dfs_by_second:
        # print df.time
        average_angle = df.apply(
            lambda x:
                angle_between((x.tip_x - x.fol_x,  x.tip_y - x.fol_y), (1,0) ),
            axis = 1
        ).mean()

        # time_point = (df.iloc[0]["time"] * frame_rate) / 1000
        time_point = (df.iloc[0]["time"] / frame_rate)

        times.append(time_point)
        angles.append(average_angle)


    # mean = np.mean(angles)
    # times, angles = reject_outliers(np.array(times), np.array(angles))
    times = np.array(times)
    angles = np.array(angles)

    data_to_plot = np.array([times, angles])
    idx = np.argsort(data_to_plot[0])
    data_to_plot = data_to_plot[:, idx]

    times, angles = data_to_plot[0, :], data_to_plot[1, :]

    return times, angles



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








def run_pipeline(video, outdir, time, region):
    #Create the output directory if it doesn't exists or clear it
    if not path.exists(outdir):
        os.makedirs(outdir)
    else:

        # delete = raw_input(
        #     "Directory '{}' already exists. Do you want to erase its contents (Enter y or n) ".format(outdir)
        # )

        #Temporarily force skip
        delete = 'n';
        if delete == 'y':
            map(os.remove, [ path.join(outdir,f) for f in os.listdir(outdir)])
        else:
            i = 2
            while path.exists(outdir + str(i)):
                i += 1

            outdir = outdir + str(i)
            os.makedirs(outdir)

    os.chdir(outdir)

    try:
        results_file = 'trace.hdf5'
        inverted_video, results_file = invert_and_trace(video, time=time)
        # filtered_summary = get_filtered_results_from_tiff_files('trace.hdf5')
        filtered_summary = get_filtered_results_by_position(results_file, region)
        overlay_video_with_results(video, inverted_video, 'trace.hdf5', filtered_summary)

        # session = get_session_from_video_filename(video)
        # pickle_file = make_vbase_pickle_file(master_pickle, session)

        # get_intervals(filtered_summary, pickle_file)

        #Change back to root directory at end
        os.chdir(nas_dir)

    except KeyboardInterrupt:
        print 'Program was closed prematurely'

        if len(os.listdir(outdir)):
            print "Deleting empty {} directory".format(outdir)
            os.rmdir(outdir)
    except:
        os.chdir(nas_dir)
        pass




if __name__ == "__main__":
    outdir = None
    time = None



    #Set up parameters
    try:
        opts, args = getopt.getopt(
            sys.argv[2:],
            'o:t:'
        )
    except getopt.GetoptError:
        print 'invert_and_analyze.py <video_name | video_direct_name> -o <output_directory> -t <time_interval> '
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-o':
            outdir = arg
        elif opt == '-t':
            time = arg




    video_argument = path.abspath(path.expanduser(sys.argv[1]))

    #Video Argument can either be a single video or a directory of videos

    if not path.isdir(video_argument):
        video = video_argument

        if not outdir:
            date_string = date.today().isoformat()
            outdir =  path.join(nas_dir, 'traces/' + date_string + '/' +  path.splitext(path.basename(video))[0] + '_trace')
        
        region = get_desired_region_from_video(video)
        run_pipeline(video, outdir, time, region)

    else:
        videos = [path.join(video_argument, fle) for fle in os.listdir(video_argument) if fle.endswith('mp4') or fle.endswith('mkv')]


        for idx, video in enumerate(videos):
            print "Preparing to trace {} ({} / {})".format(video, idx + 1, len(videos))
            if not outdir:
                date_string = date.today().isoformat()
                outdir =  path.join(nas_dir, 'traces/' + date_string + '/' +  path.splitext(path.basename(video))[0] + '_trace')
            
            region = get_desired_region_from_video(video)
            run_pipeline(video, outdir, time, regions)



