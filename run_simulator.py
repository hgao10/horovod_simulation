# horovod_simulator test runs
import horovod_event_simulator
import collections
import matplotlib.pyplot as plt
import numpy as np
import datetime
from enum import Enum
from horovod_simulator_config import SimulatorConfig, SchedulingDisc
import typing
'''
Simulation setup

input: 
    scheduling discipline: 
                                        FIFO or PerfectPQ
                                        default = FIFO
    packet_size_MB: 
                                        "packet" size in MB, see below details on the definition of packet
                                        An abstraction of packets to shorten simulation time, 
                                        with 10MB and current resnet model size 100MB, each layer
                                        has at least 2 "packets", to ensure each layer can be interrupted at least once
                                        Can be viewed as the smallest unit that needs to be transmitted or the minimal delay of network transmission if 
                                        higher priority packet is to be inserted into the front of the queue
                                        should be kept under 20MB if num_layers = 10 and model_size = 100 MB
                                        Higheset allowed value assuming per layer distributino is the same: layer_size[0] // packet_size >= 1

                                        default = 0.44
    transmission_rate_Gbit_per_sec: 
                                        Unidirectional network bandwidth in Gbit/s
                                        default = 10 
    propagation_delay_ms: 
                                        One way propogation delay in ms
                                        default = 10 us = 0.01 ms
    num_layers: 
                                        Number of layers in the target model, default assumes resnet50 v1.5
                                        default = 50 
    compute_time_per_iteration_ms: 
                                        per_layer computation time is then computed automatically based on an estimation of the generation distribution per layer
                                        default = 900 ms, execution time on a P100 GPU 
    num_workers:
                                        Num of workers that participe in allreduce, >= 2
                                        default = 2
    credit_size:                        
                                        Number of packets(tensors) must be transmitted before preemption
                                        default = 1
    num_prirority_queues:
                                        Number of priority queues
                                        default = 1 if PerfectPQ

    TotalIteration:
                                        Number of iterations to be executed per test
                                        default = 2, only need to capture 1 as each iteration takes the same amount of time assuming no disturbance is injected
output:
    iteration time in ms
    Single run:
                                        timestamp and events
    PerfectPQ:
                                        slack time per layer                                      
'''

def run_test(simulator: horovod_event_simulator.HorovodSimulator) -> typing.DefaultDict:
    simulator.run()
    print(horovod_event_simulator.compute_iteration_time(simulator.record, simulator))
    print(horovod_event_simulator.compute_slack_time_FIFO(simulator.record, simulator))
    return simulator.record

# TODO make timeline a member of HorovodSimulator
def build_timeline(record, horovod_simulator):
    timeline_event = collections.defaultdict(list)
    timeline_annotate = collections.defaultdict(list)
    # calculate per_event duration
    for key, recs in horovod_simulator.record.items():
        if key == "FP_computation_done" or key == "Gradients_received":
            for event in recs:
                if event.iteration == 1:
                    if key == "FP_computation_done":
                        # timeline_annotate[key].append(f"FP[{event.layer}]")
                        timeline_event[key].append((event.start_time, event.duration))
                    
                    if key == "Gradients_received":
                        # key = key + f"[{event.layer}]"
                        timeline_event[key+f"[{event.layer}]"].append((event.start_time, event.duration))
                        # timeline_annotate[key].append(f"GR[{event.layer}]")
        if key == "BP_computation_done" or key == "Tensor_transimission_done":
            for event in recs:
                if event.iteration == 0:
                    # timeline_event[key].append((event.start_time, event.duration))
                    if key == "BP_computation_done":
                        timeline_event[key].append((event.start_time, event.duration))
                        timeline_annotate[key].append(f"BP[{event.layer}]")
                    if key == "Tensor_transimission_done":
                        # key = key + f"[{event.layer}]"
                        timeline_event[key+f"[{event.layer}]"].append((event.start_time, event.duration))
                        timeline_annotate[key].append(f"L[{event.layer}]P[{event.packet_idx}]")
    return timeline_event    

def plot_timeline(timeline_event,horovod_simulator, plt_block=True, savefig=True):    
    print(timeline_event)
    fig, ax = plt.subplots()

    # colors = ('tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:grey', 'tab:pink')
    colors = ('tab:blue','tab:orange', 'tab:green', 'tab:purple', 'tab:cyan')

    len_c = len(colors)
    idx = 0

    num_timelines = len(timeline_event)
    ylim = [5, (num_timelines + 1) * 10]
    y_ticks = [ 15 + i*10 for i in range(num_timelines)]
    # ypos = [(10 * i, 9) for i in range(1, num_timelines+1)]
    bar_width = 8
    ypos = [(tick - bar_width/2, bar_width) for tick in y_ticks]
    yticklabels = []
    for key, value in timeline_event.items():
        print(f"barh values [{key}]: {value}")
        num_intervals = len(value)
        c = num_intervals//len_c * colors + colors[:num_intervals % len_c] 
        # ax.broken_barh(value, ypos[idx], facecolors = colors[idx])
        ax.broken_barh(value, ypos[idx], facecolors = c)
        idx += 1
        yticklabels.append(key)
    ax.set_ylim(ylim)

    timeline_start_time = timeline_event["BP_computation_done"][0][0] - 10
    timeline_finish_time = timeline_event["FP_computation_done"][-1][0] +  timeline_event["FP_computation_done"][-1][1] + 10
    xlim = (timeline_start_time, timeline_finish_time)
    print(f"xlim: {xlim}, ylim: {ylim}")
    ax.set_xlim(xlim)

    ax.set_yticks(y_ticks)
    # ax.set_yticklabels(yticklabels, fontsize='small')
    ax.set_yticklabels(yticklabels, fontsize=5)
    ax.grid(True)
    fig.set_size_inches(20.5, 12.5)
    plt.show(block=plt_block)

    plot_name = f"event_timeline_qdisc_{horovod_simulator.config.qdisc.name}_{horovod_simulator.config}_{curr_time}"
    print(f"config: {horovod_simulator.config}, curr_time: {curr_time}")
    
    if savefig:
        plt.savefig(f'./simulation_result/{plot_name}')

def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%d' % int(height),
                ha='center', va='bottom')
# common input 
curr_time = str(datetime.datetime.now()).replace(" ","_")[:19]

'''
Test 1: 
        network bandwidth vs iteration time
        FIFO and PerfectPQ
'''
def test1():
    curr_time = str(datetime.datetime.now()).replace(" ","_")
    curr_time = curr_time.split(".")[0]
    # test inputs variables
    num_runs = 10

    base_network_bandwidth_Gbit_per_sec = 1 
    bandwidth_increment_Gbit_per_sec = 1
    # key: run_idx, value: network_bandwidth 
    test_network_bandwidth = [base_network_bandwidth_Gbit_per_sec  + i * bandwidth_increment_Gbit_per_sec for i in range(num_runs)]

    # key: qdisc mode, FIFO or PerfectPQ, values: records collected from simulation runs
    iteration_result = collections.defaultdict(list)

    for mode in SchedulingDisc:
        for network_bd in test_network_bandwidth:
            if mode == SchedulingDisc.PerfectPQ:
                config = SimulatorConfig(**{"iteration_barrier": False, "qdisc": SchedulingDisc.PerfectPQ, "transmission_rate_Gbit_per_sec": network_bd }) 
            else:
                config = SimulatorConfig(**{"qdisc": SchedulingDisc.FIFO, "transmission_rate_Gbit_per_sec": network_bd }) 
            simulator = horovod_event_simulator.HorovodSimulator(config)
            simulator.run()
            iteration = horovod_event_simulator.compute_iteration_time(simulator.record, simulator)
            print(f"iteration: {iteration}")
            iteration_result[mode].append(iteration)

    # plot results
    fig, ax = plt.subplots()
    #This will create the bar graph for poulation
    ind = np.arange(num_runs) # the x locations 
    width = 0.35 # the width of the bars

    rects1 = ax.bar(ind, iteration_result[SchedulingDisc.FIFO], width)
    rects2 = ax.bar(ind+width, iteration_result[SchedulingDisc.PerfectPQ], width)

    ax.set_ylabel('Iteration time in ms')
    ax.set_xticks(ind + width/2)
    ax.set_xticklabels((str(x)+" Gbit/s" for x in test_network_bandwidth))
    ax.legend((rects1[0], rects2[0]), ("FIFO", "PerfectPQ"))
    ax.set_title("Network bandwidth vs Iteration time")

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    fig.set_size_inches(18.5, 10.5)
    plt.show(block=False)

    # plot_name = f"Network_bandwidth_vs_iteration_time_default_layer_{str(default_num_layers)}_packet_size_{str(default_packet_size_MB)}_{timestamp_str}"
    plot_name = f"Network_bandwidth_vs_iteration_time_{simulator.config}_{curr_time}"
    print(f"config: {simulator.config}, curr_time: {curr_time}")
    plt.savefig(f'./simulation_result/{plot_name}')

'''
Test 2: 
        network bandwidth vs slack time
        FIFO
'''

def test2():
    # key: layer index, val: slack time
    num_runs = 10
    slack_result = []
    base_network_bandwidth_Gbit_per_sec = 5 
    bandwidth_increment_Gbit_per_sec = 5
    # key: run_idx, value: network_bandwidth 
    test_network_bandwidth = [base_network_bandwidth_Gbit_per_sec  + i * bandwidth_increment_Gbit_per_sec for i in range(num_runs)]

    for network_bd in test_network_bandwidth:
        config = SimulatorConfig(**{"transmission_rate_Gbit_per_sec": network_bd})
        simulator = horovod_event_simulator.HorovodSimulator(config)
        # set network bandwidth
        simulator.run()
        slack = horovod_event_simulator.compute_slack_time_FIFO(simulator.record, simulator)
        slack_result.append(slack)

    # plot results
    fig, ax = plt.subplots()
    #This will create the bar graph for poulation
    ind = np.arange(simulator.config.num_layers) # the x locations 
    width = 0.08 # the width of the bars

    rects = {}
    for i in range(num_runs):
        # print(f"slack_result_per_layer[{i}], {slack_result[i]} ")
        print(f"slack_result {i}: {slack_result[i]}")
        rects[i] = ax.bar(ind + i*width, slack_result[i].values(), width)

    ax.set_ylabel('Slack time in ms')
    ax.set_xticks(ind + width * num_runs/2)
    ax.set_xticklabels(("L" + str(x) for x in range(simulator.config.num_layers)))

    ax.legend((rects[i][0] for i in range(num_runs)), (str(x)+"Gbit/s" for x in test_network_bandwidth))
    ax.set_title("FIFO Network bandwidth vs Slack time")
    fig.set_size_inches(20.5, 12.5)
    plt.show(block=False)

    # TODO fix savefig, empty graphs saved somehow
    plot_name = f"Network_bandwidth_vs_slack_time__layer_{simulator.config}_{curr_time}"
    plt.savefig(f'./simulation_result/{plot_name}')

'''
Test 3: 
        packet_size_MB vs iteration time
        PerfectPQ
'''
def test3():
    num_runs = 10

    test_min_num_packets_per_layer = [4 * runs for runs in range(1, num_runs+1)]
    iteration_result = []
    for num_packets in test_min_num_packets_per_layer:
        config = SimulatorConfig(**{"iteration_barrier": False, "qdisc": SchedulingDisc.PerfectPQ, "min_packet_per_layer":num_packets })
        simulator = horovod_event_simulator.HorovodSimulator(config)
        simulator.run()
        iteration_time = horovod_event_simulator.compute_iteration_time(simulator.record, simulator)
        iteration_result.append(iteration_time)

    # plot results
    fig, ax = plt.subplots()
    #This will create the bar graph for poulation
    ind = np.arange(num_runs) # the x locations 
    width = 0.35 # the width of the bars

    rects1 = ax.bar(ind, iteration_result, width)

    ax.set_ylabel('Iteration time in ms')
    ax.set_xticks(ind)

    ax.set_xticklabels(test_min_num_packets_per_layer)

    # ax.legend(rects1[0], "PerfectPQ")
    autolabel(rects1, ax)

    ax.set_title(f"PerfectPQ Packet Size vs Iteration time ")
    fig.set_size_inches(18.5, 10.5)
    plt_block = False
    plt.show(block=plt_block)

    if not plt_block:
        plot_name = f"Packet_size_vs_iteration_time__layer_{simulator.config}_{curr_time}"
        plt.savefig(f'./simulation_result/{plot_name}')

'''
Test 4: 
        compute_time vs iteration time
        FIFO and PerfectPQ
'''

'''
Test 5: 
        timeline events using Broken Barh
        FIFO and PerfectPQ
'''
config1 = SimulatorConfig(**{"num_layers":10, "propagation_delay_ms":5})
config2 = SimulatorConfig(**{"iteration_barrier": False, "qdisc": SchedulingDisc.PerfectPQ, "num_layers":10, "propagation_delay_ms":5})

config_FIFO = []
config_PerfectPQ = []
for network_bd in [1, 3]:
    config_FIFO.append(SimulatorConfig(**{"transmission_rate_Gbit_per_sec": 1 }))
    config_PerfectPQ.append(SimulatorConfig(**{"iteration_barrier": False, "qdisc": SchedulingDisc.PerfectPQ, "transmission_rate_Gbit_per_sec": network_bd}))

def test_timeline(config, plt_block, savefig):

    horovod_simulator = horovod_event_simulator.HorovodSimulator(config)
    r = run_test(horovod_simulator)
    timeline = build_timeline(r, horovod_simulator)
    plot_timeline(timeline, horovod_simulator, plt_block=plt_block, savefig=savefig)


if __name__ == "__main__":
    # test1()
    # test2()
    # test3()
    test_timeline(config_PerfectPQ[0], True, False)