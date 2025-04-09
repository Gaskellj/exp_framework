"""
Runs cma-es on `run_simulation.py` as a fitness function.
Creates output.csv and updates it continuously with the best individual from each generation.
Whether to show the simulation or save as video, number of generations, sigma can be passed as
command line arguments. Example: `python3 run_cmaes.py --gens 50 --sigma 2 --mode h` 
runs cma-es for 50 generations
in headless mode with a sigma of 2. Replacing "--mode h" with "--mode s" makes the simulation 
output to the screen, and replacing it with "--mode v" saves each simulation 
as a video in `./videos`. 
"--mode b" shows on screen and saves a video.

Authors: Thomas Breimer, James Gaskell
February 4th, 2025
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
from cmaes import CMA
import numpy as np
from snn_sim import run_simulation
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor
import os
import sys

def is_windows():
    """
    Checks if the operating system is Windows.

    Returns:
        bool: True if the OS is Windows, False otherwise.
    """
    return os.name == 'nt' or sys.platform.startswith('win')

# Shape of the genome
SNN_INPUT_SHAPE = 72

# Mean genome
MEAN_ARRAY = [0.0] * SNN_INPUT_SHAPE

# Num of sim time steps
NUM_ITERS = 1000

VERBOSE = False

# Index of genome & fitness in tuple
GENOME_INDEX = 0
FITNESS_INDEX = 1

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATE_TIME = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def run(mode, gens, sigma_val):
    """
    Runs the cma_es algorithm on the robot locomotion problem,
    with sin-like robot actuators. Saves a csv file to ./output
    with each robot's genome & fitness for every generation.

    Parameters:
        mode (string): How to run the simulation. 
                       "h" runs without any video or visual output.
                       "v" outputs the simulation as a video in the "./videos folder.
                       "s" shows the simulation on screen as a window.
                       "b: shows the simulation on a window and saves a video.
        gens (int): How many generations to run.
        sigma_val (float): The standard deviation of the normal distribution
        used to generate new candidate solutions
    """

    # Generate output.csv file
    csv_header = ['generation', 'best_fitness', "best_so_far"]
    csv_header.extend([f"weight{i}" for i in range(SNN_INPUT_SHAPE)])

    Path(os.path.join(ROOT_DIR, "data")).mkdir(parents=True, exist_ok=True)

    csv_path = os.path.join(ROOT_DIR, "data", f"{DATE_TIME}.csv")

    # Set up latest.csv symlink
    if os.path.exists(os.path.join("cmaes_framework", "latest.csv")):
        os.remove(os.path.join("cmaes_framework", "latest.csv"))

    if is_windows():
        os.symlink(csv_path, os.path.join("cmaes_framework", "latest.csv"))
    else:
        os.system("ln -s " + csv_path + " latest.csv")

    pd.DataFrame(columns=csv_header).to_csv(csv_path, index=False)

    # Perhaps try bounds again? doesn't seem to be doing anything
    # bounds = [(-1, 1)] * SNN_INPUT_SHAPE
    # for i in range(len(bounds)):
    #    if (i-1) % 3 == 0:
    #        bounds[i] = (0, 100)


    # Init CMA
    optimizer = CMA(
    mean=np.array(MEAN_ARRAY),
    sigma=sigma_val,
    population_size=12,
    lr_adapt=True
    )

    best_fitness_so_far = run_simulation.FITNESS_OFFSET

    # Run generations
    for generation in range(gens):
        solutions = []

        # Run individuals
        for _ in range(optimizer.population_size):
            x = optimizer.ask() # Ask cmaes for a genome
            fitness, _, _ = run_simulation.run(NUM_ITERS, x, "h") # get fitness
            solutions.append((x, fitness))

        optimizer.tell(solutions) # Tell cmaes about population

        sorted_solutions = sorted(solutions, key=lambda x: x[FITNESS_INDEX])

        best_sol = sorted_solutions[0]

        if best_sol[FITNESS_INDEX] < best_fitness_so_far:
            print("Found new best! Old:", best_fitness_so_far, "New:", best_sol[FITNESS_INDEX])
            best_fitness_so_far = best_sol[FITNESS_INDEX]

        if VERBOSE:
            print([i[1] for i in sorted_solutions])

        print("Generation", generation, "Best Fitness:", best_sol[FITNESS_INDEX])

        # Add a new row to output.csv file with cols: generation#, fitness, and genome
        new_row = [generation, best_sol[FITNESS_INDEX], best_fitness_so_far] + \
            best_sol[GENOME_INDEX].tolist()

        new_row_df = pd.DataFrame([new_row], columns=csv_header)

        # Append the new row to the CSV file using pandas in append mode (no header this time).
        new_row_df.to_csv(csv_path, mode='a', index=False, header=False)

        # If --mode s, v, or b show/save best individual from generation
        if mode in ["s", "b", "v"]:
            vid_name = DATE_TIME + "_gen" + str(generation)
            vid_path = os.path.join(ROOT_DIR, "videos", DATE_TIME)

            run_simulation.run(NUM_ITERS, best_sol[GENOME_INDEX], mode, vid_name, vid_path)

def task(name, mode, gens, sigma):
    """Task function that runs multiple threads within a multiprocessing process."""
    print(f"Process {name}: starting")

    # Create and start multiple threads
    threads = []
    for i in range(3):  # 3 threads per process
        thread = threading.Thread(target=run, args=(mode, gens, sigma))
        thread.start()
        threads.append(thread)

    # Ensure all threads complete before the process exits
    for thread in threads:
        thread.join()

    print(f"Process {name}: finishing")
    return name  # Return the process name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--mode',  # headless, screen, video, both h, s, v, b
        help='mode for output. h-headless , s-screen, v-video, b-both',
        default="h")
    parser.add_argument('--gens',
                        type=int,
                        help='number of generations to run',
                        default=100)
    parser.add_argument('--sigma',
                        type=float,
                        default=3,
                        help='sigma value for cma-es')
    args = parser.parse_args()

    num_workers = multiprocessing.cpu_count()

    # Use ProcessPoolExecutor instead of multiprocessing.Pool
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(task, name, args.mode, args.gens, args.sigma): name
                   for name in ["A", "B", "C"]}

        for future in futures:
            try:
                result = future.result()  # Wait for process completion
                print(f"Completed process: {result}")
            except Exception as e:
                print(f"Process {futures[future]} failed with error: {e}")

    print("All processes completed.")