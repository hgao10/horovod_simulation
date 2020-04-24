# horovod_simulator test runs
import horovod_event_simulator
import collections
import matplotlib.pyplot as plt
import numpy as np

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

                                        default = 10 
    transmission_rate_Gbit_per_sec: 
                                        Unidirectional network bandwidth in Gbit/s
                                        default = 10 
    propagation_delay_ms: 
                                        One way propogation delay in ms
                                        default = 5
    num_layers: 
                                        Number of layers in the target model, default assumes resnet50 v1.5
                                        default = 182
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

'''
Test 1: 
        network bandwidth vs iteration time
        FIFO and PerfectPQ
'''
# default values for inputs
default_num_layers = 50
default_packet_size_MB = 10

num_runs = 10
base_network_bandwidth_Gbit_per_sec = 5 
bandwidth_increment_Gbit_per_sec = 5
# key: run_idx, value: network_bandwidth 
test_network_bandwidth = [base_network_bandwidth_Gbit_per_sec * i for i in range(1, num_runs+1)]
# key: qdisc mode, FIFO or PerfectPQ, values: records collected from simulation runs
iteration_result = collections.defaultdict(list)

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%d' % int(height),
                ha='center', va='bottom')
for mode in ["FIFO", "PerfectPQ"]:
    for network_bd in test_network_bandwidth:
        simulator = horovod_event_simulator.HorovodSimulator(default_num_layers, default_packet_size_MB)
        # set network bandwidth
        simulator.set_transmission_rate_Gbit_per_sec(network_bd)
        if mode == "PerfectPQ":
            print("MODE == PERFECTPQ")
            simulator.set_PerfectPQ_qdisc()
        simulator.run()
        iteration = horovod_event_simulator.compute_iteration_time(simulator.record, simulator)
        iteration_result[mode].append(iteration)

# plot results
fig, ax = plt.subplots()
#This will create the bar graph for poulation
ind = np.arange(num_runs) # the x locations 
width = 0.35 # the width of the bars

rects1 = ax.bar(ind, iteration_result["FIFO"], width)
rects2 = ax.bar(ind+width, iteration_result["PerfectPQ"], width)

ax.set_ylabel('Iteration time in ms')
ax.set_xticks(ind + width/2)
ax.set_xticklabels((str(x)+" Gbit/s" for x in test_network_bandwidth))
ax.legend((rects1[0], rects2[0]), ("FIFO", "PerfectPQ"))
ax.set_title("Network bandwidth vs Iteration time")

autolabel(rects1)
autolabel(rects2)

fig.set_size_inches(18.5, 10.5)
# plt.show()

plot_name = f"Network_bandwidth_vs_iteration_time_default_layer_{str(default_num_layers)}_packet_size_{str(default_packet_size_MB)}"
plt.savefig(f'./simulation_result/{plot_name}')

'''
Test 2: 
        network bandwidth vs slack time
        FIFO
'''
# key: layer index, val: slack time
slack_result = []
for network_bd in test_network_bandwidth:
    simulator = horovod_event_simulator.HorovodSimulator(default_num_layers, default_packet_size_MB)
    # set network bandwidth
    simulator.set_transmission_rate_Gbit_per_sec(network_bd)
    simulator.run()
    slack = horovod_event_simulator.compute_slack_time_FIFO(simulator.record, simulator)
    slack_result.append(slack)

# plot results
fig, ax = plt.subplots()
#This will create the bar graph for poulation
ind = np.arange(default_num_layers) # the x locations 
width = 0.08 # the width of the bars

rects = {}
for i in range(num_runs):
    # print(f"slack_result_per_layer[{i}], {slack_result[i]} ")
    print(f"slack_result {i}: {slack_result[i]}")
    rects[i] = ax.bar(ind + i*width, slack_result[i].values(), width)

ax.set_ylabel('Slack time in ms')
ax.set_xticks(ind + width * num_runs/2)
ax.set_xticklabels(("L" + str(x) for x in range(default_num_layers)))
# ax.legend(rects1[0], "PerfectPQ")
# autolabel(rects1)

ax.legend((rects[i][0] for i in range(num_runs)), (str(x)+"Gbit/s" for x in test_network_bandwidth))
ax.set_title("FIFO Network bandwidth vs Slack time")
fig.set_size_inches(20.5, 12.5)
# plt.show()

# TODO fix savefig, empty graphs saved somehow
# plot_name = f"Network_bandwidth_vs_slack_time__layer_{str(default_num_layers)}_packet_size_{str(default_packet_size_MB)}"
# plt.savefig(f'./simulation_result/{plot_name}')

'''
Test 3: 
        packet_size_MB vs iteration time
        PerfectPQ
'''
test_packet_size_MB = [0.1 * runs for runs in range(1, num_runs+1)]
iteration_result = []
default_network_tranmission_rate_Gbit_per_sec = 50
for packet_size_MB in test_packet_size_MB:
    simulator = horovod_event_simulator.HorovodSimulator(default_num_layers, packet_size_MB)
    simulator.set_transmission_rate_Gbit_per_sec(default_network_tranmission_rate_Gbit_per_sec)
    simulator.set_PerfectPQ_qdisc()
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

str_test_packet_sizes = [f'{x:.3f}' for x in test_packet_size_MB]
ax.set_xticklabels(str_test_packet_sizes)

file_name_packet_sizes = [x.replace(".", "_") for x in str_test_packet_sizes]
print(file_name_packet_sizes)
# ax.legend(rects1[0], "PerfectPQ")
autolabel(rects1)

ax.set_title(f"PerfectPQ Packet Size vs Iteration time (Bandwidth: {default_network_tranmission_rate_Gbit_per_sec} Gbit/s)")
fig.set_size_inches(18.5, 10.5)
plt.show()

plot_name = f"packet_size_MB_vs_iteration_time__layer_{str(default_num_layers)}_network_{str(default_network_tranmission_rate_Gbit_per_sec)}_packet_size_{file_name_packet_sizes}"
plt.savefig(f'./simulation_result/{plot_name}')

'''
Test 4: 
        compute_time vs iteration time
        FIFO and PerfectPQ
'''
