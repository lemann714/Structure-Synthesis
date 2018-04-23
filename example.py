#!/usr/bin/env python

import matplotlib.pyplot as plt
from copy import deepcopy, copy
from synthesis_repo import BaseGenetics
from synthesis_repo import StructureGenetics
from synthesis_repo import NetworkOperator
from synthesis_repo import addition, multiplication, maximum, minimum, relu, identity, pow_two, negative, irer, reverse, exp, natlog, logic, cosinus, sinus, cubicroot, atan, cubic
from synthesis_repo.models import Robot

if __name__=='__main__':
    # 1) Instantiate model
    time = 2.3
    x0 = 10
    y0 = 10
    tetta0 = 0
    xf = 0
    yf = 0
    tettaf = 0
    robot = Robot(time, x0, y0, tetta0, xf, yf, tettaf)
    # 2) Instantiate network operator
    unaries = [identity, negative, pow_two, negative, sinus, logic, cosinus, atan, exp, natlog, irer, cubic]#, cubicroot]#reverse, 
    binaries = [addition, multiplication, maximum, minimum]
    input_nodes = [0, 1, 2, 3, 4, 5]
    output_nodes = [9, 10]
    nop = NetworkOperator(unaries, binaries, input_nodes, output_nodes)
    q = [1.68258017, 0.68013064, 1.72779495]
    nop.set_q(q)
    nop.update_base_q()
    #psi = [[(0, 0), (3, 0), (5, 6), (6, 1)], [(1, 0), (4, 0), (7, 0)], [(2, 0), (5, 5), (0, 8), (8, 1)], [(6, 4), (7, 0), (8, 0), (9, 0)], [(6, 0), (7, 0), (8, 5), (6, 8), (10, 0)]]
    #with obsticle
    psi = [[(0, 7), (4, 7), (4, 1), (6, 1)], [(0, 8), (1, 5), (7, 3)], [(2, 3), (5, 0), (3, 10), (2, 4), (8, 0)], [(6, 10), (8, 11), (0, 11), (7, 10), (9, 3)], [(6, 0), (7, 10), (8, 0), (1, 3), (1, 4), (1, 1), (2, 6), (2, 0), (4, 10), (4, 0), (4, 0), (4, 0), (7, 1), (9, 6), (5, 5), (10, 0)]]
    nop.set_psi(psi)
    nop.update_base_psi()
    # 3) Instantiate structure genetic and pass model with nop to it
    sg = StructureGenetics(nop, robot)
    # 4) Set synthesis parameters

    # psi_change_epoch <= individuals
    # ksearch <= individuals
    # variations_per_individuals >= 1
    # g > 0
    # crossings > 0

    sg.optimize(qmin=-5, qmax=5,
                individuals=100,
                generations=10,
                psi_change_epoch=2,
                variations_per_individual=10,
                crossings=30,
                ksearch=15)

    robot.simulate()
    x, y = robot.get_coords()
    plt.axes()
    circle = plt.Circle((5, 5), radius=2.5, lw=2.0, fc='y', edgecolor='black')
    plt.gca().add_patch(circle)
    plt.plot(x, y, 'b')
    plt.axis('scaled')
    plt.show()
