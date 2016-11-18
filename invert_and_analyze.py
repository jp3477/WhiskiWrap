import ffmpy
import sys
import WhiskiWrap
from os import path

def invert_video(infile, outfile):
    """ This function inverts the colors in video designated by infile
        and writes the results to outfile
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
                '00:00:20' 
                ]}
        )

    ff.run()

def invert_and_trace(video, outdir):
    """ Inverts a video's colors and then runs a trace """
    video_name, ext = path.splitext(video)
    inverted_video_path = path.join(outdir, path.basename(video_name) + "_inverted" + ".mp4")
    invert_video(video, inverted_video_path)

    #Trace
    WhiskiWrap.pipeline_trace(
        inverted_video_path, 
        'trace.hdf5',
        n_trace_processes=4
    )

def classify():
    pass







if __name__ == "__main__":
    video = sys.argv[1]
    outdir = sys.argv[2]
    
    invert_and_trace(video, outdir)
    

