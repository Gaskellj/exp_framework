"""
Given a genome, runs a simulation of a walking robot in evogym, using an SNN controlled robot,
providing a fitness score corresponding to how far the robot walked.

Author: Thomas Breimer, James Gaskell
January 29th, 2025
"""

import os
import sys
import itertools
from pathlib import Path
import time
import cv2
import numpy as np
from evogym import EvoWorld, EvoSim, EvoViewer
from evogym import WorldObject

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from snn_sim.robot.morphology import Morphology
from snn.snn_controller import SNNController

# Simulation constants
ROBOT_SPAWN_X = 2
ROBOT_SPAWN_Y = 0
ACTUATOR_MIN_LEN = 0.6
ACTUATOR_MAX_LEN = 1.6
NUM_ITERS = 1000
FPS = 50
MODE = "v" # "headless", "screen", or "video"

FITNESS_OFFSET = 100

# Files
ENV_FILENAME = "bigger_platform.json"
ROBOT_FILENAME = "bestbot.json"
THIS_DIR = os.path.dirname(os.path.realpath(__file__))

def create_video(source, output_name, vid_path, fps=FPS):
    """
    Saves a video from a list of frames

    Parameters:
        source (list): List of cv2 frames.
        output_name (string): Filename of output video.
        vid_path (string): Filepath of output video.
        fps (int): Frames per second of video to save.
    """

    Path(vid_path).mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(os.path.join(vid_path, output_name + ".mp4"),
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (source[0].shape[1], source[0].shape[0]))
    for frame in source:
        out.write(frame)
    out.release()

def group_list(flat_list: list, n: int) -> list:
    """
    Groups flat_array into a list of list of size n.

    Parameters:
        flat_list (list): List to groups.
        n: (int): Size of sublists.
    
    Returns:
        list: Grouped list.
    """
    return [list(flat_list[i:i+n]) for i in range(0, len(flat_list), n)]

def run(iters, genome, mode, vid_name=None, vid_path=None):
    """
    Runs a single simulation of a given genome.

    Parameters:
        iters (int): How many iterations to run.
        genome (ndarray): The genome of the robot.
        mode (string): How to run the simulation. 
                       "h" runs without any video or visual output.
                       "v" outputs the simulation as a video in the "./videos folder.
                       "s" shows the simulation on screen as a window.
                       "b: shows the simulation on a window and saves a video.
        vid_name (string): If mode is "v" or "b", this is the name of the saved video.
        vid_path (string): If mode is "v" or "b", this is the path the video will be saved.
    Returns:
        float: The fitness of the genome.
    """

    # Create world
    world = EvoWorld.from_json(os.path.join(THIS_DIR, 'robot', 'world_data', ENV_FILENAME))

    # Add robot
    robot = WorldObject.from_json(os.path.join(THIS_DIR, 'robot', 'world_data', ROBOT_FILENAME))

    world.add_from_array(
        name='robot',
        structure=robot.get_structure(),
        x=ROBOT_SPAWN_X + 1,
        y=ROBOT_SPAWN_Y + 1,
        connections=robot.get_connections())

    # Create simulation
    sim = EvoSim(world)
    sim.reset()

    # Set up viewer
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')

    video_frames = []

    # Get position of all robot point masses
    init_raw_pm_pos = sim.object_pos_at_time(sim.get_time(), "robot")

    morphology = Morphology(ROBOT_FILENAME)

    robot_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'robot', 'world_data', ROBOT_FILENAME)

    snn_controller = SNNController(2, 2, 1, robot_config=robot_file_path)
    snn_controller.set_snn_weights(genome)

    action_log = []

    for i in range(iters):
        if i % 12 == 0:
            # Get point mass locations
            raw_pm_pos = sim.object_pos_at_time(sim.get_time(), "robot")

            if i == 0:
                init_corner_distances = np.array(morphology.get_corner_distances(raw_pm_pos))

            # Get current corner distances
            corner_distances = np.array(morphology.get_corner_distances(raw_pm_pos))

            epsilon = 1e-10  # Prevent division by zero

            # Step 1: Compute the difference between current and initial corner distances
            delta_distances = corner_distances - init_corner_distances

            # Step 2: Scale the difference by the inverse of initial corner distances
            scaled_differences = delta_distances / (init_corner_distances + epsilon)

            # Step 3: Normalize scaled differences between [-1, 1]
            arr_min = np.min(scaled_differences)
            arr_max = np.max(scaled_differences)

            if arr_max != arr_min:  # Avoid division by zero in normalization
                normalized_distances = 2 * (scaled_differences - arr_min) / (arr_max - arr_min) - 1
            else:
                normalized_distances = np.zeros_like(scaled_differences)

            # Use the normalized distances as input
            action = snn_controller.get_lengths(corner_distances)

            time.sleep(0.01)

            # action = [[1.6] if x[0] > 1 else [0.6] for x in action]
            #action = np.array(action)

            # Clip actuator target lengths to be between 0.6 and 1.6 to prevent buggy behavior
            action = np.clip(action[0], ACTUATOR_MIN_LEN, ACTUATOR_MAX_LEN)
            action_log.append(action)

            # Set robot action to the action vector. Each actuator corresponds to a vector
            # index and will try to expand/contract to that value
            sim.set_action('robot', action)

            # Execute step
        sim.step()

        if mode == "v":
            video_frames.append(viewer.render(verbose=False, mode="rgb_array"))
        elif mode == "s":
            viewer.render(verbose=True, mode="screen")
        elif mode == "b":
            viewer.render(verbose=True, mode="screen")
            video_frames.append(viewer.render(verbose=False, mode="rgb_array"))

    action_log_iter = []
    for x in action_log:
        action_log_iter.append(list(itertools.chain.from_iterable(x)))

    levels_log = snn_controller.get_levels_log()

    viewer.close()

    # Get robot point mass position position afer sim has run
    final_raw_pm_pos = sim.object_pos_at_time(sim.get_time(), "robot")

    fitness = np.mean(final_raw_pm_pos[0]) - np.mean(init_raw_pm_pos[0])

    bottom_pos = final_raw_pm_pos[1][-4:]
    for val in bottom_pos: # Fix falling over in fitness
        if val > 1.6:
            if not np.mean(final_raw_pm_pos[1]) - np.mean(init_raw_pm_pos[1]) > 0.6: # Checks if robot is airborne so we don't get rid of jumping bots
                fitness = 0


    if mode in ["v", "b"]:
        create_video(video_frames, vid_name, vid_path, FPS)

    return FITNESS_OFFSET - fitness, action_log_iter, levels_log # Turn into a minimization problem
