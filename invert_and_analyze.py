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


import tables
import pandas
import whiskvid.output_video as ov
from base import FFmpegReader
from pymediainfo import MediaInfo

from angle_plotting import *
from make_pickle import make_vbase_pickle_file, get_session_from_video_filename

import logging

#Set up error logger
logging.basicConfig(filename='/tmp/trace.log', level=logging.DEBUG,
                            format='%(asctime)s %(levelname)s %(name)s %(message)s')

logger = logging.getLogger(__name__)

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

def invert_and_trace(video, time=None, results_file='trace.hdf5'):
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
    with tables.open_file(whiskers_file) as handle:

        #Lowered buffer size, look into standard error output
        input_reader = FFmpegReader(original_video, bufsize=10**6)

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





# os.chdir(path.expanduser('/home/jason/mnt/jason_nas2_home/traces/2017-03-24/CR2-20170303122448_trace'))
# original_video = path.expanduser('~/dev/to_trace4/CR2-20170303122448.mkv')
# inverted_video = path.expanduser('CR2-20170303122448_inverted.mp4')
# whiskers_file = path.expanduser('trace.hdf5')
# whiskers_table = get_filtered_results_from_tiff_files(whiskers_file)



# overlay_video_with_results(original_video, inverted_video, whiskers_file, whiskers_table)





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
        #First create symlink to original video to make accessing it easier
        #os.symlink(video, path.basename(video))

        results_file = 'trace.hdf5'
        inverted_video, results_file = invert_and_trace(video, time=time)

        #Filter traced data
        filtered_summary = get_filtered_results_by_position(results_file, region)

        #Get plots
        session = get_session_from_video_filename(video)
        pickle_file = make_vbase_pickle_file(master_pickle, session)
        angles_by_frame = get_angle_over_time(filtered_summary, outfile='all_angles.pickle', frame_rate=30.0)
        get_hit_and_error_angles_on_trial_intervals(angles_by_frame, pickle_file, save_data=True, save_figure=True, frame_rate=30.0)

        #Make new video with overlayed traces
        # overlay_video_with_results(video, inverted_video, 'trace.hdf5', filtered_summary)

        #Change back to root directory at end
        os.chdir(nas_dir)

    except KeyboardInterrupt:
        print 'Program was closed prematurely'

        if len(os.listdir(outdir)):
            print "Deleting empty {} directory".format(outdir)
            os.rmdir(outdir)
    except Exception as e:
        logger.error(e)
        os.chdir(nas_dir)
        pass




if __name__ == "__main__":
    time = None
    mouse = None



    #Set up parameters
    try:
        opts, args = getopt.getopt(
            sys.argv[2:],
            'o:t:m:'
        )
    except getopt.GetoptError:
        print 'invert_and_analyze.py <video_name | video_direct_name> -o <output_directory> -t <time_interval> '
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-o':
            outdir = arg
        elif opt == '-t':
            time = arg
        elif opt == '-m':
            mouse = arg




    video_argument = path.abspath(path.expanduser(sys.argv[1]))

    #Video Argument can either be a single video or a directory of videos

    if not path.isdir(video_argument):
        video = video_argument

        # If optional outdir command line argument is not given
        if not outdir:
            if mouse:
                mouse = mouse.lower()
                outdir = path.join(nas_dir, 'traces/' + mouse + '/' + path.splitext(path.basename(video))[0] + '_trace')
            else:
                date_string = date.today().isoformat()
                outdir =  path.join(nas_dir, 'traces/' + date_string + '/' +  path.splitext(path.basename(video))[0] + '_trace')

        region = get_desired_region_from_video(video)
        run_pipeline(video, outdir, time, region)

    else:
        videos = [path.expanduser(path.join(video_argument, fle)) for fle in os.listdir(video_argument) if fle.endswith('mp4') or fle.endswith('mkv')]
        regions = []

        for idx, video in enumerate(videos):
            regions.append(get_desired_region_from_video(video))

        for idx, video in enumerate(videos):
            print "Preparing to trace {} ({} / {})".format(video, idx + 1, len(videos))
            if mouse:
                mouse = mouse.lower()
                outdir = path.join(nas_dir, 'traces/' + mouse + '/' + path.splitext(path.basename(video))[0] + '_trace')
            else:
                date_string = date.today().isoformat()
                outdir =  path.join(nas_dir, 'traces/' + date_string + '/' +  path.splitext(path.basename(video))[0] + '_trace')

            run_pipeline(video, outdir, time, regions[idx])


    print "Error logs stored in /tmp/trace.log"



