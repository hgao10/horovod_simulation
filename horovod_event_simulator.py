import collections
import time
import heapq

# architecuture
IterationBarrier = True
FIFO = True
PerfectPQ = not FIFO

event_queue = []
transmission_rate_Gbit_per_sec = 10 
num_layer = 100 #make it an argument

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

transmission_queue = collections.deque()

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
def transmit_tensor(curr_time, transmission_queue ): # tensor_layer, iteration):
    # transmit whatever is in front of the queue FIFO
    if transmission_queue: 
        iteration_idx, layer_idx, packet_idx = transmission_queue.popleft()
        # if transmission_queue[0][0]  == 0: #> 0:
        #     # no packets left to be transmitted in this layer, pop and move to next layer
        #     transmission_queue.popleft()
        heapq.heappush(event_queue, [tensor_transmittion_time_ms+curr_time, "Done_transmitting_tensors",  layer_idx, iteration_idx, packet_idx])
        # if packet_idx == layer_size_in_packets[layer_idx] - 1:
        #     if layer_idx == 0:
        #         iteration_idx += 1
        #         print(f'increment interation: {iteration_idx}')
            # create a future event to receive gradients for this layer
        if packet_idx == layer_size_in_packets[layer_idx] - 1: # last packet in the layer, assume that there is no OOO transmission
            if not increment_iteration_status[iteration_idx+1]: # any layer that finishes transmitting all gradients will increament the iteration for that layer
                iteration_idx += 1
            heapq.heappush(event_queue, [TotalAllReduceTime + curr_time, "Received_gradients_update", layer_idx, iteration_idx, packet_idx])
        #to_be_transmitted = tensor_size
        global InTransit 
        InTransit = True

def add_to_transmission_queue(queue, num_packets, layer, iteration):
    for i in range(num_packets):
        packet_id = (iteration, layer, i)
        queue.append(packet_id)
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
        if iteration != 0:
            # 2nd iteration onwards
            # restore previous FP compute status to not ready for next iteration
            previous_FP_layer_status[layer] = False
            if layer < num_layer-1: # before last layer
                previous_FP_layer_status[layer+1] = True

        if layer == num_layer - 1: #last layer
            record.append([curr_time, "Start BP"])
            heapq.heappush(event_queue,[bp_layers[layer]+curr_time,"BP_computation_done", layer, iteration] )

    elif (event[1] == "BP_computation_done"):
        iteration = event[3]
        curr_time = timestamp
        # ready to send gradient
        num_packets = layer_size_in_packets[layer]
        # transmission_queue.append([num_packets, layer])
        add_to_transmission_queue(transmission_queue, num_packets, layer, iteration)

        if not InTransit: # nothing is being transimitted 
            transmit_tensor(curr_time, transmission_queue)
        # start BP for next layer
        if layer > 0:
            heapq.heappush(event_queue,[bp_layers[layer]+curr_time,"BP_computation_done", layer-1, iteration] )

    elif event[1] == "Done_transmitting_tensors":
        InTransit = False
        curr_time = timestamp
        transmit_tensor(curr_time, transmission_queue)
    
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
                print(f'{curr_time},Start FP computation in new iteration,{iteration}')
                enque_FP(curr_time, iteration)
            else:
                print(f'Have not received all gradients')
        else: # start FP whenever previous FP layer has finished computation and gradients have been received and updated this layer 
            print(f'previous_FP_layer_status[layer]: {previous_FP_layer_status[layer]}')
            if previous_FP_layer_status[layer]:
                # start computation of FP layer
                compute_time = fp_layers[layer]
                heapq.heappush(event_queue, [compute_time+curr_time, "FP_computation_done", layer, iteration])
    else:
        print(f"Error: Non-existing Event: {event}")
        break

print(record)

            



