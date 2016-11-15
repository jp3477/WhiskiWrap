import ffmpy
import sys
import WhiskiWrap
from os import path

def invert_video(infile, outfile):
    """ This function inverts the colors in video designated by infile
        and writes the results to outfile
    """
    ff = ffmpy.FFmpeg(
            global_options='-y',
            inputs={infile : None},
            outputs={outfile : ['-vf', 'lutrgb=r=negval:g=negval:b=negval']}
        )

    ff.run()

def invert_and_trace(video, outdir):
    """ Inverts a video's colors and then runs a trace """
    video_name, ext = path.splitext(video)
    inverted_video_path = path.join(outdir, video_name + "_inverted" + ".mp4")
    invert_video(video, inverted_video_path)

    #Trace
    WhiskiWrap.pipeline_trace(
        video, 
        path.join(outdir,'trace.hdf5'), 
        n_trace_processes=4
    )







if __name__ == "__main__":
    video = sys.argv[1]
    outdir = sys.argv[2]
    
    invert_and_trace(video, outdir)
    

