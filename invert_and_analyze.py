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

    #Create the output directory if it doesn't exists or clear it
    if outdir == None:
        outdir = path.basename(video) + '_trace'
    if not path.exists(outdir):
        os.makedirs(outdir)
    else:
        map(os.remove, [ path.join(outdir,f) for f in os.listdir(outdir)])

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
                angle_between((x.tip_x - x.fol_x,  x.tip_y - x.fol_y), (1,0) ) > 10,
            axis = 1
        )
    ]


    return filtered_results

def plot_angle_over_time(data, frame_rate=30):
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

    plt.plot(times,angles)
    plt.title('Angle behavior')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.show()







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


if __name__ == "__main__":
    outdir = None
    time = ''

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
    
    if time:
        invert_and_trace(video, outdir=outdir, time=time)
    else:
        invert_and_trace(video, outdir=outdir)

    # select_region('output2/chunk00000000.tif')
    

