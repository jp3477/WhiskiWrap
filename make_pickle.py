# Sync a session

import MCwatch.behavior.syncing
import my
import numpy as np
import ArduFSM
import pandas
import sys
import os
import re
from IPython import embed

# Get the synced behavior and video files

def make_vbase_pickle_file(master_pickle, session):
    sbvdf = pandas.read_pickle(master_pickle)
    session = sbvdf.loc[sbvdf.session.str.startswith(session)].session.values[0]
    print session

    #Had to make a symlink to my home directory named 'jack'

    # Choose a session
    session_params = sbvdf.set_index('session').ix[session]

    # Get the video file for this
    video_file = session_params['filename_video']
    print video_file

    # Get the behavior file
    bfile = session_params['filename']
    print bfile

    # Get the trial matrix filename
    trial_matrix_filename = os.path.join('/home/jack/jason_data/trial_matrix', session)

    # Get the mean luminances
    print "loading luminances ... this will take a while"
    lums = my.video.process_chunks_of_video(video_file, n_frames=np.inf)

    # Get onsets and durations, lowered delta from 75 to 60
    onsets, durations = MCwatch.behavior.syncing.extract_onsets_and_durations(-lums,
        delta=60, diffsize=2, refrac=50)

    # Convert to seconds in the spurious timebase
    v_onsets = onsets / 30.
    v_onsets = v_onsets[1:]

    # Get the data from Ardulines
    lines = ArduFSM.TrialSpeak.read_lines_from_file(bfile)
    parsed_df_by_trial = \
        ArduFSM.TrialSpeak.parse_lines_into_df_split_by_trial(lines)

    # Find the time of transition into state 1
    backlight_times = ArduFSM.TrialSpeak.identify_state_change_times(
        parsed_df_by_trial, state1=1, show_warnings=True)

    # Find the fit
    b2v_fit = MCwatch.behavior.syncing.longest_unique_fit(v_onsets, backlight_times)

    ## Put the times in the trial matrix
    tm = pandas.read_csv(trial_matrix_filename)
    tm = ArduFSM.TrialMatrix.add_rwin_and_choice_times_to_trial_matrix(
        tm, bfile)

    # Calculate rwin time in vbase
    tm['rwin_time_vbase'] = np.polyval(b2v_fit, tm.rwin_time)

    embed()


    # Save the tm
    outfile = 'tm_%s.pickle' % session
    tm.to_pickle(outfile)

    return outfile


def get_session_from_video_filename(video_filename):
    session = re.search('-(\d*)', video_filename).group(1)
    return session

#master_pickle = os.path.expanduser('~/jason_data/sbvdf.pickle')
#session = '20170303122448'
#make_vbase_pickle_file(master_pickle, session)
