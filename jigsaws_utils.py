import os
import numpy as np

from video_loader import *

i2task = {0: 'Knot_Tying',
        1: 'Needle_Passing',
        2: 'Suturing'}
ntask = len(i2task.keys())
task2i = {v: k for k, v in i2task.items()}

# Gesture index to description
gi2descr = {
    1: 'Reaching for needle with right hand',
    2: 'Positioning needle',
    3: 'Pushing needle through tissue',
    4: 'Transferring needle from left to right',
    5: 'Moving to center with needle in grip',
    6: 'Pulling suture with left hand',
    7: 'Pulling suture with right hand',
    8: 'Orienting needle',
    9: 'Using right hand to help tighten suture',
    10: 'Loosening more suture',
    11: 'Dropping suture at end and moving to end points',
    12: 'Reaching for needle with left hand',
    13: 'Making C loop around right hand',
    14: 'Reaching for suture with right hand',
    15: 'Pulling suture with both hands'
}

root_path = 'data/JIGSAWS_converted'
def load_video_data(tasks=None, subjects=None, trials=None, captures=None, gestures=None, verbose=True):
    if tasks is None:
        tasks = np.array(list(task2i.values()))
    else:
        tasks = np.array(tasks).ravel()

    #or_tasks = '\|'.join([i2task[task] for task in tasks])
    stream = os.popen("find %s -name '*.avi' | sed 's:^.*/\([^/]\+_[A-Z][0-9]\{3\}\)_.*$:\\1:'" %'data/JIGSAWS_converted')
    video_meta = stream.read().split('\n')
    video_meta = [name[-4:] for name in video_meta for task in tasks if i2task[task] in name]
    if subjects is None:
        subjects = np.unique([x[0] for x in video_meta])
    else:
        subjects = np.array(subjects).ravel()

    if trials is None:
        trials = np.unique([x[-1] for x in video_meta])
    else:
        trials = np.array(trials).ravel()

    if captures is None:
        captures = np.array([1,2])
    else:
        captures = np.array(captures).ravel()

    if gestures is None:
        gestures = np.array(list(gi2descr.keys()))
    else:
        gestures = np.array(gestures).ravel()

    X = []
    y = []
    #print(tasks, subjects, trials, captures)
    for task in tasks:
        task_name = i2task[task]
        for subject in subjects:
            for trial in trials:
                transcr_filename = f'{root_path}/{task_name}/transcriptions/{task_name}_{subject}00{trial}.txt'
                try:
                    with open(transcr_filename, 'r') as fp:
                        for l in fp.readlines():
                            start_frame, end_frame, gesture = l.split()
                            start_frame = int(start_frame)
                            end_frame = int(end_frame)
                            gesture = int(gesture[1:])
                            if gesture not in gestures:
                                continue
                            for capt in captures:
                                video_filename = f'{root_path}/{task_name}/video/{task_name}_{subject}00{trial}_capture{capt}.avi'
                                fragment = VideoLoader(video_filename, gray=True, start_frame=start_frame,
                                                       duration_frames=end_frame-start_frame+1)
                                fragment.trial = trial
                                fragment.jig_capture = capt
                                fragment.subject = subject
                                fragment.task = task
                                fragment.gesture = gesture

                                X.append(fragment)
                                y.append(gesture)
                except FileNotFoundError:
                    if verbose:
                        print(f"Missing file for: task '{task_name}', subject '{subject}', trial {trial}")
                    continue
    y = np.array(y)
    return X, y
