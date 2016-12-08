import matplotlib
matplotlib.use('Qt4Agg')


import ffmpy
import sys , getopt
import WhiskiWrap
from os import path
import os
import numpy as np
import numpy.linalg as la
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tables
import pandas
import whiskvid.output_video as ov
from base import FFmpegReader
from pymediainfo import MediaInfo





def invert_video(infile, outfile, time='00:00:20'):
    """ This function inverts the colors in video designated by infile
        and writes the results to outfile

        infile: input video filename
        outfile: inverted output video filename
    """
    # eq=1:0:3:1:1:1:1:1'
    ff = ffmpy.FFmpeg(
            global_options='-y',
            inputs={infile : None},
            outputs={outfile : ['-vf', 
                'lutrgb=r=negval:g=negval:b=negval' ,
                '-ss',
                '00:00:00',
                '-t',
                time,

                ]}
        )

    ff.run()

def invert_and_trace(video, outdir=None, time='00:00:20', results_file='trace.hdf5'):
    """ Inverts a video's colors and then runs a trace

        video : input video filename
        outdir : directory where trace products will be placed
        results_file : dataframe that holds the geometric data of trace
    """

    results_file = path.join(outdir, results_file)

    video_name, ext = path.splitext(video)
    inverted_video = path.join(outdir, path.basename(video_name) + "_inverted" + ".mp4")
    invert_video(video, inverted_video, time=time)

    #Trace
    WhiskiWrap.pipeline_trace(
        inverted_video, 
        results_file,
        n_trace_processes=4
    )

    tiff_file = [ fi for fi in os.listdir(outdir) if fi.endswith(".tif") ][0]
    tiff_file = path.join(outdir, tiff_file)

    region = select_region(tiff_file)
    results = get_filtered_results_by_position(results_file, region)


    handle = tables.open_file(results_file)
    input_reader = FFmpegReader(video)

    vid_info = MediaInfo.parse(inverted_video).tracks[1]

    width = vid_info.width
    height = vid_info.height
    frame_rate = int(float(vid_info.frame_rate)) # Frames per second
    video_length = int(vid_info.duration) / 1000 #Video duration in seconds

    frame_count = frame_rate * video_length

    overlayed_video = path.join(outdir, 'overlayed.mp4')

    print "Overlaying results onto video"
    ov.write_video_with_overlays(
        overlayed_video,
        input_reader,
        width,
        height,
        whiskers_table=results,
        whiskers_file_handle=handle,
        frame_triggers=[0],
        trigger_dstart = 0,
        trigger_dstop = frame_count
    )

    print "Saved in {}".format(overlayed_video)

    plot_angle_over_time(results, path.join(outdir, 'angle_plot.png'), frame_rate)


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
        Show image on axis and allow regional selection with rectangle
    """
    global startpos
    global endpos
    ax = plt.gca()
    img = mpimg.imread(image_file)
    imgplot = ax.imshow(img, cmap='Greys_r')

    selector = PersistentRectangleSelector(ax, onselect, drawtype='box')
    plt.show()

    return {'startpos' : startpos, 'endpos' : endpos}

def get_results_from_hdf5(hdf5_file):
    with tables.open_file(hdf5_file) as fi:
        results = pandas.DataFrame.from_records(fi.root.summary.read())

    return results

def get_filtered_results_by_position(results_file, selected_positions):
    """ Filters identififed follicles by location and angle

    """
    results = get_results_from_hdf5(results_file)

    startpos = selected_positions['startpos']
    endpos = selected_positions['endpos']
    left_limit, right_limit = min(startpos[0], endpos[0]), max(startpos[0], endpos[0])
    up_limit, down_limit = min(startpos[1], endpos[1]), max(startpos[1], endpos[1])

    filtered_results = results[
        (results.fol_x > left_limit) & (results.fol_x < right_limit) &
        (results.fol_y > up_limit) & (results.fol_y < down_limit) &
        (results.fol_x < results.tip_x)
    ]
    
    # for index, x in filtered_results.iterrows():
    #     vector = (x.tip_x - x.fol_x,  x.tip_y - x.fol_y)
    #     print angle_between(vector, (1,0)) > 1

    filtered_results = filtered_results[
        filtered_results.apply(
            lambda x: 
                np.absolute(angle_between((x.tip_x - x.fol_x,  x.tip_y - x.fol_y), (1,0) )) > 5,
            axis = 1
        )
    ]


    return filtered_results

def plot_angle_over_time(data, savefile, frame_rate=30,):
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

    median = np.median(angles)
    times, angles = reject_outliers(np.array(times), np.array(angles))

    plt.plot(times,angles)
    plt.plot(times, np.ones(len(times)) * median, 'r')
    plt.title('Angle behavior')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.savefig(savefile)

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


    median = np.median(angles)
    times, angles = reject_outliers(np.array(times), np.array(angles))

    data_to_plot = np.array([times, angles])
    idx = np.argsort(data_to_plot[0])
    data_to_plot = data_to_plot[:, idx]

    times, angles = data_to_plot[0, :], data_to_plot[1, :]

    return times, angles




# def angle_between(v1, v2):
#     """ Returns the angle in radians between vectors 'v1' and 'v2'    """
#     cosang = np.dot(v1, v2)
#     sinang = la.norm(np.cross(v1, v2))

#     return np.rad2deg(np.arctan2(sinang, cosang))

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
    rwin_vbase_times = pandas.read_pickle(pickle_file)['rwin_time_vbase']
    max_time = data.iloc[-1].time / frame_rate

    current_frame = 0
    i = 0

    while (rwin_vbase_times).iloc[i] < max_time:
        
        vbase_time = rwin_vbase_times.iloc[i]

        #Create a interval that will always stay the same length
        start = int((vbase_time - 2) * frame_rate)
        interval = (start, start + (frame_rate * 5))
        chunk = data[(data.time >= interval[0]) & (data.time < interval[1])]

        times, angles = get_angle_over_time(chunk, frame_rate=frame_rate)
        print (times - times[0]) * frame_rate
        i += 1

        






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
        print 'invert_and_analyze.py <video_name> -o <output_directory> -t <time_interval> '
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-o':
            outdir = arg
        elif opt == '-t':
            time = arg


    video = sys.argv[1]

    #Create the output directory if it doesn't exists or clear it
    if not outdir:
        outdir = path.basename(video) + '_trace'
    if not path.exists(outdir):
        os.makedirs(outdir)
    else:
        map(os.remove, [ path.join(outdir,f) for f in os.listdir(outdir)])

    if not time:
        time = '00:00:40'
    

    invert_and_trace(video, outdir=outdir, time=time)


