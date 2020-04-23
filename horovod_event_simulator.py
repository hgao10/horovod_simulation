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
num_layer = 10 #make it an argument

# Resnet50 on one P100: 900 ms 
iteration_time_P100_ms = 900
fp_total_time_ms = (1/3) * iteration_time_P100_ms
bp_total_time_ms = (2/3) * iteration_time_P100_ms

# To simplify computation time in FP: assum each layer takes less then d ms to compute than previous layer and the last layer takes 0 ms
fp_diff_per_layer_ms = 2 * fp_total_time_ms // (num_layer * (num_layer-1))
fp_first_layer_ms = 2 * fp_total_time_ms // num_layer
# Same simplification applies to BP except its in ascending order
bp_diff_per_layer_ms = 2 * bp_total_time_ms // (num_layer * (num_layer-1))

fp_layers = {layer: fp_first_layer_ms - layer * fp_diff_per_layer_ms for layer in range(num_layer)}
bp_layers = {layer: layer * bp_diff_per_layer_ms for layer in range(num_layer)}
# fp_layers = {layer: num_layer*2-layer for layer in range(num_layer)}
# bp_layers = {layer: layer*3 + 1 for layer in range(num_layer)}

# tracks computation completion status per layer according to data dependency
# the first layer is always true as it doesn't depend on any previous layer
previous_FP_layer_status = {layer: False for layer in range(num_layer)}
previous_FP_layer_status[0] = True

# total model is 100MB
model_size_MB = 100 #MB 
# To simplify layer size, first half of the layers are size S and the next 1/4 half of the layers are of size 4s and the last 1/4 of size 12s
min_layer_size_MB = 2 * model_size_MB // 9  

layer_size = {}
for layer in range(num_layer):
    if layer <= num_layer//2:
        layer_size[layer] = min_layer_size_MB
    elif  num_layer//2 <layer <= 3*num_layer//4:
        layer_size[layer] = 4 * min_layer_size_MB
    else:
        layer_size[layer] = 12 * min_layer_size_MB
# layer_size = {layer: model_size/num_layer * (layer +1) for layer in range(num_layer) }

num_priority_queues = 4
priority_queues = {}
for i in range(num_priority_queues):
    priority_queues[i] = collections.deque()

# smallest transmission unit tensor, could also be packet
packet_size_MB = 10  

# number of packets to be sent/received per layer 
layer_size_in_packets = {} 
for layer in range(num_layer):
    layer_size_in_packets[layer] = int(layer_size[layer]//packet_size_MB) # gradient is always multiples of tensors
    print(f'layer_size_in_packets[{layer}]: {layer_size_in_packets[layer]}')
# TODO incorperate credit_size in non perfect priority queue situation where packets can only be pre-empted if there is enough credit left 
credit_size = 1
now_pico_seconds = 0
TotalIteration = 2
increment_iteration_status = {i: False for i in range(TotalIteration+1)}

gradient_received = {layer: False for layer in range(num_layer)}
record = []
received_tensor_count = {layer: 0 for layer in range(num_layer)}

InTransit = False
allReduceComputeTime = 0
ApplyLayerGradient = 0 

# The transmission delay is the amount of time required for the router to push out the packet.
# The propagation delay, is the time it takes a bit to propagate from one router to the next.
tensor_transmittion_time_ms = packet_size_MB * 8 /transmission_rate_Gbit_per_sec 
print(f'tensor_transmittion_time_ms: {tensor_transmittion_time_ms}')
propagation_delay_ms = 5 # ms

#TODO simplied version, each worker sends the entire amount of gradient per layer at once instead of gradient/num_worker for num_worker times, refer to ring allreduce paper
TotalAllReduceTime = allReduceComputeTime + ApplyLayerGradient + 2* (tensor_transmittion_time_ms + propagation_delay_ms) # compute + network roundtrip time

#TODO create an event class! and pass event.str() to record function

def enque_FP(curr_time, iteration):
    for layer, compute_time in fp_layers.items():
        heapq.heappush(event_queue, [compute_time + curr_time, "FP_computation_done", layer,  iteration])
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
    heapq.heappush(event_queue, [tensor_transmittion_time_ms+curr_time, "Done_transmitting_tensors",  packet.layer_idx, packet.iteration_idx, packet.packet_idx])
    if packet.packet_idx == layer_size_in_packets[packet.layer_idx] - 1: # last packet in the layer, assume that there is no OOO transmission
        if not increment_iteration_status[packet.iteration_idx+1]: # any layer that finishes transmitting all gradients will increament the iteration for that layer
            packet.iteration_idx += 1
        heapq.heappush(event_queue, [TotalAllReduceTime + curr_time, "Received_gradients_update", packet.layer_idx, packet.iteration_idx, packet.packet_idx])
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
    # if layer == 3:
    #     while PerfectPQ_transmission_queue:
    #         print(f'Pop packets: {heapq.heappop(PerfectPQ_transmission_queue)}')


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

# enque all FP events for the first iteration where there is no blocking
curr_time = 0
record.append([curr_time, "Start FP"])
enque_FP(curr_time, 0)

''' main event loop '''
while event_queue:
    event = heapq.heappop(event_queue)
    timestamp, layer, iteration = event[0], event[2], event[3]
    record.append(event)
    print(f'event: {event}')
    if event[1] == "FP_computation_done":
        iteration = event[3]
        curr_time = timestamp
        if PerfectPQ:
            if iteration != 0: # all FP events have been pushed for iteration 0
                # 2nd iteration onwards
                # restore previous FP compute status to not ready for next iteration
                if layer != 0: # first layer is execluded because it's always ready to compute once gradients are received
                    previous_FP_layer_status[layer] = False            
                if layer < num_layer-1: # unblock the compute for next FP layer
                    previous_FP_layer_status[layer+1] = True
                    if gradient_received[layer+1]:
                        heapq.heappush(event_queue, [fp_layers[layer+1] + curr_time, "FP_computation_done", layer+1,  iteration])
                gradient_received[layer] = False
        # no need to handle FIFO case cause all FP events have been pushed once at the start of the new iteration 
        if layer == num_layer - 1: #last layer
            record.append([curr_time, "Start BP"])
            heapq.heappush(event_queue,[bp_layers[layer]+curr_time,"BP_computation_done", layer, iteration] )

    elif (event[1] == "BP_computation_done"):
        iteration = event[3]
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
            heapq.heappush(event_queue,[bp_layers[layer]+curr_time,"BP_computation_done", layer-1, iteration] )

    elif event[1] == "Done_transmitting_tensors":
        InTransit = False
        curr_time = timestamp
        transmit_tensor(curr_time)
    
    elif event[1] == "Received_gradients_update":
        curr_time = timestamp
        gradient_received[layer] = True
        # Barrier between each iteration, current implementation
        if iteration == TotalIteration:
            print(f'break out of while loop : iteration: {iteration}')
            # exit while loops
            break
        if IterationBarrier == True:
            if sum(gradient_received.values()) == num_layer: # all gradients have received
                print(f'{curr_time},Start FP computation in new iteration in FIFO mode,{iteration}')
                enque_FP(curr_time, iteration)
            else:
                print(f'Have not received all gradients')
        else: # start FP whenever previous FP layer has finished computation and gradients have been received and updated this layer 
            print(f'previous_FP_layer_status[layer]: {previous_FP_layer_status[layer]}')
            if previous_FP_layer_status[layer]:
                # start computation of FP layer
                compute_time = fp_layers[layer]
                if layer == 0:
                    print(f'{curr_time},Start FP computation in new iteration in Perfect PQ mode,{iteration}')
                heapq.heappush(event_queue, [compute_time+curr_time, "FP_computation_done", layer, iteration])
    else:
        print(f"Error: Non-existing Event: {event}")
        break

print(record)

            



