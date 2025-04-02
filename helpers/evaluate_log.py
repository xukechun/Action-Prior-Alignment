#!/usr/bin/env python

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


# Parse session directories
parser = argparse.ArgumentParser(description='Calculating performance of testing logs.')
parser.add_argument('--session_directory', dest='session_directory', action='store', type=str, help='path to session directory for which to measure performance')
args = parser.parse_args()
session_directory = args.session_directory

# Parse data from session (task completions, episode steps, reward values) 
results_directory = os.path.join(session_directory, 'results')
result_files = os.listdir(results_directory)
result_files.sort(key=lambda x:int(x[4:6].split('.')[0]))

avg_success = []
avg_step = []
avg_success_step = []
avg_reward = []
for file in result_files:
    lang_goal = ''
    file = os.path.join(results_directory, file)
    with open(file, "r") as f:
        file_content = f.readlines()
        result = file_content[0].split()
        for x in result[:-4]:
            lang_goal += x
            lang_goal += " "
        success = float(result[-4])
        step = float(result[-3])
        success_step = float(result[-2])
        reward = float(result[-1])

        avg_success.append(success*100)
        avg_step.append(step)
        if success_step != 1000:
            avg_success_step.append(success_step)
        avg_reward.append(reward)

    print("Language goal: %s, average steps: %.2f, average success: %.2f" % (lang_goal, success_step, success*100))

print('Testing Result of Experiment: %s' % results_directory)
print('Average Task Success: %.2f' % np.array(avg_success).mean())
print('Average Step: %.2f' % np.array(avg_step).mean())
print('Average Success Step: %.2f' % np.array(avg_success_step).mean())
print('Average Reward: %.2f' % np.array(avg_reward).mean())