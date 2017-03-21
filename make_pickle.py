# Sync a session

import MCwatch.behavior
import my
import numpy as np
import ArduFSM
import pandas
import sys

# Get the synced behavior and video files

def make_vbase_pickle_file(master_pickle, session):
    sbvdf = pandas.read_pickle(master_pickle)
    session = sbvdf.loc[sbvdf.session.str.startswith(session)].session

    # Choose a session
    session_params = sbvdf.set_index('session').ix[s]

    # Get the video file for this
    video_file = session_params['filename_video']

    # Get the behavior file
    bfile = session_params['filename']

    # Get the trial matrix filename
    trial_matrix_filename = os.path.join('/home/jack/jason_data/trial_matrix', session)

    # Get the mean luminances
    print "loading luminances ... this will take a while"
    lums = my.video.process_chunks_of_video(video_file, n_frames=np.inf)

    # Get onsets and durations
    onsets, durations = MCwatch.behavior.syncing.extract_onsets_and_durations(-lums,
        delta=75, diffsize=2, refrac=50)

    # Convert to seconds in the spurious timebase
    v_onsets = onsets / 30.

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

    # Save the tm
    tm.to_pickle('tm_%s.pickle' % session_name)



