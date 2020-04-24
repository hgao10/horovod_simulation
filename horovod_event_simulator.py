import collections
import time
import heapq

# architecuture
IterationBarrier = True
FIFO = IterationBarrier
PerfectPQ = not FIFO
transmission_queue = collections.deque()
PerfectPQ_transmission_queue = [] # minheap sorted by priority

event_queue = []
transmission_rate_Gbit_per_sec = 10 
num_layers = 10 #make it an argument

# Resnet50 on one P100: 900 ms 
compute_time_per_iteration_ms = 900
fp_total_time_ms = (1/3) * compute_time_per_iteration_ms
bp_total_time_ms = (2/3) * compute_time_per_iteration_ms

# To simplify computation time in FP: assum each layer takes less then d ms to compute than previous layer and the last layer takes 0 ms
fp_diff_per_layer_ms = 2 * fp_total_time_ms // (num_layers * (num_layers-1))
fp_first_layer_ms = 2 * fp_total_time_ms // num_layers
# Same simplification applies to BP except its in ascending order
bp_diff_per_layer_ms = 2 * bp_total_time_ms // (num_layers * (num_layers-1))

fp_layers = {layer: fp_first_layer_ms - layer * fp_diff_per_layer_ms for layer in range(num_layers)}
bp_layers = {layer: layer * bp_diff_per_layer_ms for layer in range(num_layers)}

# tracks computation completion status per layer according to data dependency
# the first layer is always true as it doesn't depend on any previous layer
previous_FP_layer_status = {layer: False for layer in range(num_layers)}
previous_FP_layer_status[0] = True

# total model is 100MB
model_size_MB = 100 #MB 
# To simplify layer size, first half of the layers are size S and the next 1/4 half of the layers are of size 4s and the last 1/4 of size 12s
min_layer_size_MB = 2 * model_size_MB // 9  

layer_size = {}
for layer in range(num_layers):
    if layer <= num_layers//2:
        layer_size[layer] = min_layer_size_MB
    elif  num_layers//2 <layer <= 3*num_layers//4:
        layer_size[layer] = 4 * min_layer_size_MB
    else:
        layer_size[layer] = 12 * min_layer_size_MB
# layer_size = {layer: model_size/num_layers * (layer +1) for layer in range(num_layers) }

num_priority_queues = 4
priority_queues = {}
for i in range(num_priority_queues):
    priority_queues[i] = collections.deque()

# smallest transmission unit tensor, could also be packet
packet_size_MB = 10  

# number of packets to be sent/received per layer 
layer_size_in_packets = {} 
for layer in range(num_layers):
    layer_size_in_packets[layer] = int(layer_size[layer]//packet_size_MB) # gradient is always multiples of tensors
    print(f'layer_size_in_packets[{layer}]: {layer_size_in_packets[layer]}')
# TODO incorperate credit_size in non perfect priority queue situation where packets can only be pre-empted if there is enough credit left 
credit_size = 1
TotalIteration = 2
increment_iteration_status = {i: False for i in range(TotalIteration+1)}

gradient_received = {layer: False for layer in range(num_layers)}
# key: event name value: event obj
record = collections.defaultdict(list)
received_tensor_count = {layer: 0 for layer in range(num_layers)}

InTransit = False
allReduceComputeTime = 0
ApplyLayerGradient = 0 

# The transmission delay is the amount of time required for the router to push out the packet.
# The propagation delay, is the time it takes a bit to propagate from one router to the next.
tensor_transmittion_time_ms = packet_size_MB * 8 /transmission_rate_Gbit_per_sec 
print(f'tensor_transmittion_time_ms: {tensor_transmittion_time_ms}')
propagation_delay_ms = 5 # ms

#TODO simplied version, each worker sends the entire amount of gradient per layer at once instead of gradient/num_workers for num_workers times, refer to ring allreduce paper
TotalAllReduceTime = allReduceComputeTime + ApplyLayerGradient + 2* (tensor_transmittion_time_ms + propagation_delay_ms) # compute + network roundtrip time

#TODO create an event class! and pass event.str() to record function

def enque_FP(curr_time, iteration):
    for layer, compute_time in fp_layers.items():
        next_event = Compute_Event(compute_time + curr_time, "FP", layer, iteration, "done")
        heapq.heappush(event_queue, next_event)
        # heapq.heappush(event_queue, [compute_time + curr_time, "FP_computation_done", layer,  iteration])
        curr_time += compute_time

# transmission queue: comprised of packet_id (iteration_idx, layer_idx, packet_idx)
def transmit_tensor(curr_time): # tensor_layer, iteration):
    if FIFO and transmission_queue:
        packet = transmission_queue.popleft()
    elif PerfectPQ and PerfectPQ_transmission_queue:
        packet = heapq.heappop(PerfectPQ_transmission_queue)
    else:
        return
    # print(f'transimitting packet: iter:{packet.iteration_idx}, layer: {packet.layer_idx}, id: {packet.packet_idx}')
    next_event = Transmit_Event(tensor_transmittion_time_ms+curr_time, "done", packet.iteration_idx, packet.layer_idx, packet.packet_idx)
    heapq.heappush(event_queue, next_event)
    # heapq.heappush(event_queue, [tensor_transmittion_time_ms+curr_time, "Tensor_transimission_done",  packet.layer_idx, packet.iteration_idx, packet.packet_idx])
    if packet.packet_idx == layer_size_in_packets[packet.layer_idx] - 1: # last packet in the layer, assume that there is no OOO transmission
        if not increment_iteration_status[packet.iteration_idx+1]: # any layer that finishes transmitting all gradients will increament the iteration for that layer
            packet.iteration_idx += 1
        next_event = Gradients_Event(TotalAllReduceTime + curr_time, packet.iteration_idx, packet.layer_idx)
        heapq.heappush(event_queue, next_event)
        # heapq.heappush(event_queue, [TotalAllReduceTime + curr_time, "Gradients_received", packet.layer_idx, packet.iteration_idx])
    #to_be_transmitted = tensor_size
    global InTransit 
    InTransit = True

def add_to_transmission_queue(num_packets, layer, iteration):
    for i in range(num_packets):
        p = Packet(iteration, layer, i)
        if FIFO:
            # print(f'FIFO: add packets to transmission queue')
            transmission_queue.append(p)
        elif PerfectPQ:
            # print(f'PerfectPQ: add packets to transmission queue')
            heapq.heappush(PerfectPQ_transmission_queue, p)
        else:
            print(f'Error: packet isnt added')

class Packet():
    def __init__(self, iteration_idx, layer_idx, packet_idx):
        # global packet_size_MB
        self.iteration_idx = iteration_idx
        self.layer_idx = layer_idx
        self.packet_idx = packet_idx
        self.priority = self.layer_idx
        self.size = packet_size_MB

    def __lt__(self, other):
        return self.priority < other.priority or ((self.priority == other.priority) and self.packet_idx < other.packet_idx)

    def __str__(self):
        return (f'Packet.priority, {self.priority}, Packet.id, {self.packet_idx}, Packet.iteration, {self.iteration_idx}, Packet.layer, {self.layer_idx}')
    
    def set_priority(self, priority):
        self.priority = priority


class Event():
    def __init__(self, name, time):
        self.name = name
        self.time = time

    def __lt__(self, other):
        return self.time < other.time
    
    def __str__(self):
        return (f'Time_ms, {self.time}, Event, {self.name}')


class Compute_Event(Event):
    def __init__(self, time, direction, layer, iteration, state):
        # Forward or Backward
        name = direction + '_computation_' + state
        super().__init__(name, time)    
        self.direction = direction 
        self.iteration = iteration
        self.layer = layer
        # start or done
        self.state = state
    
    def __str__(self):
        return (f'Time_ms, {self.time}, Event, {self.name}, Iter, {self.iteration}, Layer, {self.layer}')


class Transmit_Event(Event):
    def __init__(self, time, state, iteration, layer, packet_idx):
        # start or done
        self.state = state
        name = 'Tensor_transimission_' + state 
        super().__init__(name, time)
        self.iteration = iteration
        self.layer = layer
        self.packet_idx = packet_idx
        # Start or Finish
        self.state = state
    
    def __str__(self):
        return (f'Time_ms, {self.time}, Event, {self.name}, Iter, {self.iteration}, Layer, {self.layer}, Packet_idx, {self.packet_idx}')


class Gradients_Event(Event):
    def __init__(self, time, iteration, layer):
        super().__init__("Gradients_received", time)
        self.iteration = iteration
        self.layer = layer
    
    def __str__(self):
        return (f'Time_ms, {self.time}, Event, {self.name}, Iter, {self.iteration}, Layer, {self.layer}')

# enque all FP events for the first iteration where there is no blocking
curr_time = 0
record["Start FP"].append(Event("Start FP", curr_time))
enque_FP(curr_time, 0)

''' main event loop '''
while event_queue:
    event = heapq.heappop(event_queue)
    timestamp, layer, iteration = event.time, event.layer, event.iteration
    record[event.name].append(event)
    print(f'event: {event}')
    if event.name == "FP_computation_done":
        curr_time = timestamp
        if PerfectPQ:
            if iteration != 0: # all FP events have been pushed for iteration 0
                # 2nd iteration onwards
                # restore previous FP compute status to not ready for next iteration
                if layer != 0: # first layer is execluded because it's always ready to compute once gradients are received
                    previous_FP_layer_status[layer] = False            
                if layer < num_layers-1: # unblock the compute for next FP layer
                    previous_FP_layer_status[layer+1] = True
                    if gradient_received[layer+1]:
                        next_event = Compute_Event(fp_layers[layer+1] + curr_time, "FP", layer+1, iteration, "done")
                        heapq.heappush(event_queue, next_event)
                        # heapq.heappush(event_queue, [fp_layers[layer+1] + curr_time, "FP_computation_done", layer+1,  iteration])
                gradient_received[layer] = False
        # no need to handle FIFO case cause all FP events have been pushed once at the start of the new iteration 
        if layer == num_layers - 1: #last layer
            # record.append([curr_time, "Start BP"])
            record["Start BP"].append(Event("Start BP", curr_time))
            next_event = Compute_Event(bp_layers[layer]+curr_time, "BP", layer, iteration, "done")
            heapq.heappush(event_queue, next_event)
            # heapq.heappush(event_queue,[bp_layers[layer]+curr_time,"BP_computation_done", layer, iteration] )

    elif (event.name == "BP_computation_done"):
        curr_time = timestamp
        # ready to send gradient
        num_packets = layer_size_in_packets[layer]
        # transmission_queue.append([num_packets, layer])
        add_to_transmission_queue(num_packets, layer, iteration)
        # print(PerfectPQ_transmission_queue)
        if not InTransit: # nothing is being transimitted 
            transmit_tensor(curr_time)
        # start BP for next layer
        if layer > 0:
            next_event = Compute_Event(bp_layers[layer]+curr_time, "BP", layer-1, iteration, "done")
            heapq.heappush(event_queue, next_event)
            # heapq.heappush(event_queue,[bp_layers[layer]+curr_time,"BP_computation_done", layer-1, iteration] )

    elif event.name == "Tensor_transimission_done":
        InTransit = False
        curr_time = timestamp
        transmit_tensor(curr_time)
    
    elif event.name == "Gradients_received":
        curr_time = timestamp
        gradient_received[layer] = True
        # Barrier between each iteration, current implementation
        if iteration == TotalIteration:
            print(f'break out of while loop : iteration: {iteration}')
            # exit while loops
            break
        if IterationBarrier == True:
            if sum(gradient_received.values()) == num_layers: # all gradients have received
                print(f'{curr_time},Start FP computation in new iteration in FIFO mode,{iteration}')
                record["Start FP computation in new iteration in FIFO mode"].append(Event("Start FP computation in new iteration in FIFO mode", curr_time))
                enque_FP(curr_time, iteration)
            else:
                print(f'Have not received all gradients')
        else: # start FP whenever previous FP layer has finished computation and gradients have been received and updated this layer 
            print(f'previous_FP_layer_status[{layer}]: {previous_FP_layer_status[layer]}')
            if previous_FP_layer_status[layer]:
                # start computation of FP layer
                compute_time = fp_layers[layer]
                if layer == 0:
                    print(f'{curr_time},Start FP computation in new iteration in Perfect PQ mode,{iteration}')
                    record["Start FP computation in new iteration in Perfect PQ mode"].append(Event("Start FP computation in new iteration in Perfect PQ mode", curr_time))
                next_event = Compute_Event(compute_time+curr_time, "FP", layer, iteration, "done")
                heapq.heappush(event_queue, next_event)
                # heapq.heappush(event_queue, [compute_time+curr_time, "FP_computation_done", layer, iteration])
    else:
        print(f"Error: Non-existing Event: {event}")
        break

#print(record)

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

# compute iteration time from records
def compute_iteration_time(record):
    iteration_time_ms = 0
    iteration_start_time = 0
    for event in record["FP_computation_done"]:
        if  event.layer == num_layers -1:
            if event.iteration == 0:
                iteration_start_time = event.time
            if event.iteration == 1:
                iteration_time_ms = event.time - iteration_start_time
                break
    print(f'iteration_time_ms: {iteration_time_ms}') 
    return iteration_time_ms
    
def compute_slack_time_FIFO(record, fp_layers):
    # compute slack per layer for FIFO
    slack_per_layer_in_ms = {layer: 0 for layer in range(num_layers)}
    # Time difference between when gradients are computed to when gradients are needed
    # gradients_received_timestamp = {layer: 0 for layer in range(num_layers)}
    BP_computation_done_timestamp = {layer: 0 for layer in range(num_layers)}
    for event in record["BP_computation_done"]:
        if event.iteration == 0:
            BP_computation_done_timestamp[event.layer] = event.time
    for event in record["FP_computation_done"]:
        if event.iteration == 1:
            # print(f'layer: {event.layer}, FP_computation_done, {event.time}, fp_layers, {fp_layers[event.layer]}, BP compute done: { BP_computation_done_timestamp[event.layer]}')
            slack_per_layer_in_ms[event.layer] = event.time - fp_layers[event.layer] - BP_computation_done_timestamp[event.layer]

    print(f'slack_per_layer_in_ms: {slack_per_layer_in_ms}')
    return slack_per_layer_in_ms

#compute_iteration_time(record)
#compute_slack_time_FIFO(record, fp_layers)



