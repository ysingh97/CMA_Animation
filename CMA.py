import Creature_Simulators
import pydart2 as pydart
from cma import *
import numpy as np

import copy
import multiprocessing as mp
from functools import partial
import math

general_option = {'maxiter': 70, 'popsize': 30}

DURATION = 5
SIMULATOR = Creature_Simulators.BaseSimulator
OPTIONS = general_option
CMA_STEP_SIZE = 0.6
NUM_RESTART = 1


def fitnessFunction(inputVector):
    simulator = SIMULATOR(inputVector)
    initialPos = simulator.skeletons[1].q
    endPos = episode(simulator)
    result = 10
    if endPos is not None:
        result = endPos[0] - initialPos[0]
    return result

FITNESS_FUNC =  fitnessFunction



def episode(current_simulator):

    terminal_flag = False
    while current_simulator.t < DURATION and not terminal_flag:
        current_simulator.step()
        curr_q = current_simulator.skeletons[1].q
        if abs(curr_q.any()) > 10 ** 3:
            print(curr_q)
            print(current_simulator.skeletons[1].controller.compute())
            print(current_simulator.skeletons[1].controller.pd_controller_target_compute())
            terminal_flag = True
            print("NAN")
    res = current_simulator.skeletons[1].q
    if terminal_flag:
        res = None
    return res


def run_CMA(x0):
    global OPTIONS, FITNESS_FUNC
    es = CMAEvolutionStrategy(x0, CMA_STEP_SIZE, OPTIONS)
    es.optimize(FITNESS_FUNC)
    res = es.result
    return res


def writeToFile(res):
    s = "MaxIterations: " + str(general_option['maxiter']) + "; "
    s += "Population: " + str(general_option['popsize']) + "; "
    s += "Iterations: " + str(res.iterations) + "; "
    s += "DPhase: " + str(res.xbest[0]) + "; "
    s += "Amplitude: " + str(res.xbest[1]) + "; "
    s += "Period: " + str(res.xbest[2]) + "\n"
    file = open("results.txt", "a+")
    file.write(s)


if __name__=='__main__':
    pydart.init()
    print(fitnessFunction([math.pi/2, 1, 2]))

    lb = [-10, -10, -10]
    hb = [10, 10, 10]
    OPTIONS['bounds'] = lb, hb

    x0 = [0, 0, 0]

    res = run_CMA(x0)
    writeToFile(res)

    print(res.xbest)

    testSimulator = SIMULATOR(res.xbest)
    pydart.gui.viewer.launch(testSimulator)







