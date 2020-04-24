# horovod_simulator test runs
import horovod_event_simulator
import collections
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

num_runs = 10
base_network_bandwidth_Gbit_per_sec = 5 
bandwidth_increment_Gbit_per_sec = 5
# key: run_idx, value: network_bandwidth 
test_network_bandwidth = {run_idx: base_network_bandwidth_Gbit_per_sec * run_idx for run_idx in range(1, num_runs+1)}

# key: qdisc mode, FIFO or PerfectPQ, values: records collected from simulation runs
result = collections.defaultdict(list)
for mode in ["FIFO", "PerfectPQ"]:
    for run_idx, network_bd in test_network_bandwidth.items():
        record = horovod_event_simulator.simulator(transmission_rate_Gbit_per_sec=network_bd, qdisc = "FIFO")
        result[mode].append(record)

# plot results

'''
Test 2: 
        network bandwidth vs slack time
        FIFO
'''

'''
Test 3: 
        compute_time vs iteration time
        FIFO and PerfectPQ
'''

'''
Test 4: 
        packet_size_MB vs iteration time
        PerfectPQ
'''
