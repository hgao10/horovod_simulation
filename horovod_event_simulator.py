import collections
import time
import heapq

class Packet():
    def __init__(self, iteration_idx, layer_idx, packet_idx, packet_size_MB):
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


class HorovodSimulator():
    def __init__(self):
        # key: event name value: event obj
        self.record = collections.defaultdict(list)
        
        self.iteration_barrier = True
        self.FIFO_set = self.iteration_barrier
        self.PerfectPQ_set = not self.FIFO_set
        
        # FIFO queue
        self.transmission_queue = collections.deque()
        # minheap sorted by priority
        self.PerfectPQ_transmission_queue = []

        # event queue is used as a minheap sorted by timestamp
        self.event_queue = []
        self.curr_time = 0
        # model specific
        self.num_layers = 10
        
        # Resnet50 on one P100: 900 ms 
        self.compute_time_per_iteration_ms = 900
        self.fp_total_time_ms = (1/3) * self.compute_time_per_iteration_ms
        self.bp_total_time_ms = (2/3) * self.compute_time_per_iteration_ms
        # To simplify computation time in FP: assum each layer takes less then d ms to compute than previous layer and the last layer takes 0 ms
        self.fp_diff_per_layer_ms = 2 * self.fp_total_time_ms // (self.num_layers * (self.num_layers-1))
        self.fp_first_layer_ms = 2 * self.fp_total_time_ms // self.num_layers
        # Same simplification applies to BP except its in ascending order
        self.bp_diff_per_layer_ms = 2 * self.bp_total_time_ms // (self.num_layers * (self.num_layers-1))
        self.fp_layers = {layer: self.fp_first_layer_ms - layer * self.fp_diff_per_layer_ms for layer in range(self.num_layers)}
        self.bp_layers = {layer: layer * self.bp_diff_per_layer_ms for layer in range(self.num_layers)}
        # total model is 100MB
        self.model_size_MB = 100 #MB 
        # To simplify layer size, first half of the layers are size S and the next 1/4 half of the layers are of size 4s and the last 1/4 of size 12s
        self.min_layer_size_MB = 2 * self.model_size_MB // 9  
        self.layer_size = {}
        self.calculate_layer_size()        

        # Test run specs
        self.TotalIteration = 2

        # horovod simulator status registers
        self.gradient_received = {layer: False for layer in range(self.num_layers)}
        self.received_tensor_count = {layer: 0 for layer in range(self.num_layers)}
        # tracks computation completion status per layer according to data dependency
        # the first layer is always true as it doesn't depend on any previous layer
        self.previous_FP_layer_status = {layer: False for layer in range(self.num_layers)}
        self.previous_FP_layer_status[0] = True
        self.increment_iteration_status = {i: False for i in range(self.TotalIteration+1)}

        self.InTransit = False

        # non-essential compute time
        self.allReduceComputeTime = 0
        self.ApplyLayerGradient = 0


        # network specs
        self.transmission_rate_Gbit_per_sec = 10
        # smallest transmission unit tensor, could also be packet
        self.packet_size_MB = 10 
        # number of packets to be sent/received per layer  
        self.layer_size_in_packets = {}
        self._init_layer_size_in_packets()
        # The transmission delay is the amount of time required for the router to push out the packet.
        # The propagation delay, is the time it takes a bit to propagate from one router to the next.
        self.tensor_transmittion_time_ms = self.packet_size_MB * 8 /self.transmission_rate_Gbit_per_sec 
        self.propagation_delay_ms = 5 # ms
        #TODO simplied version, each worker sends the entire amount of gradient per layer at once instead of gradient/num_workers for num_workers times, refer to ring allreduce paper
        self.TotalAllReduceTime = self.allReduceComputeTime + self.ApplyLayerGradient + 2* (self.tensor_transmittion_time_ms + self.propagation_delay_ms) # compute + network roundtrip time


        # TODO future feature: multiple priority queues
        self.num_priority_queues = 4
        self.priority_queues = {}
        # TODO incorperate credit_size in non perfect priority queue situation where packets can only be pre-empted if there is enough credit left 
        self.credit_size = 1

    def calculate_layer_size(self):
        for layer in range(self.num_layers):
            if layer <= self.num_layers//2:
                self.layer_size[layer] = self.min_layer_size_MB
            elif  self.num_layers//2 <layer <= 3*self.num_layers//4:
                self.layer_size[layer] = 4 * self.min_layer_size_MB
            else:
                self.layer_size[layer] = 12 * self.min_layer_size_MB

    def _init_priority_queues(self):
            for i in range(self.num_priority_queues):
                self.priority_queues[i] = collections.deque() 
    
    def _init_layer_size_in_packets(self):
        for layer in range(self.num_layers):
            self.layer_size_in_packets[layer] = int(self.layer_size[layer]//self.packet_size_MB) # gradient is always multiples of tensors
            print(f'layer_size_in_packets[{layer}]: {self.layer_size_in_packets[layer]}')

    def enque_FP(self, curr_time, iteration):
        for layer, compute_time in self.fp_layers.items():
            next_event = Compute_Event(compute_time + curr_time, "FP", layer, iteration, "done")
            heapq.heappush(self.event_queue, next_event)
            curr_time += compute_time

    # transmission queue: comprised of packet_id (iteration_idx, layer_idx, packet_idx)
    def transmit_tensor(self, curr_time):
        if self.FIFO_set and self.transmission_queue:
            packet = self.transmission_queue.popleft()
        elif self.PerfectPQ_set and self.PerfectPQ_transmission_queue:
            packet = heapq.heappop(self.PerfectPQ_transmission_queue)
        else:
            return
        # print(f'transimitting packet: iter:{packet.iteration_idx}, layer: {packet.layer_idx}, id: {packet.packet_idx}')
        next_event = Transmit_Event(self.tensor_transmittion_time_ms+curr_time, "done", packet.iteration_idx, packet.layer_idx, packet.packet_idx)
        heapq.heappush(self.event_queue, next_event)
        if packet.packet_idx == self.layer_size_in_packets[packet.layer_idx] - 1: # last packet in the layer, assume that there is no OOO transmission
            if not self.increment_iteration_status[packet.iteration_idx+1]: # any layer that finishes transmitting all gradients will increament the iteration for that layer
                packet.iteration_idx += 1
            next_event = Gradients_Event(self.TotalAllReduceTime + curr_time, packet.iteration_idx, packet.layer_idx)
            heapq.heappush(self.event_queue, next_event)
        self.InTransit = True


    def add_to_transmission_queue(self, num_packets, layer, iteration):
        for i in range(num_packets):
            p = Packet(iteration, layer, i, self.packet_size_MB)
            if self.FIFO_set:
                # print(f'self.FIFO_set: add packets to transmission queue')
                self.transmission_queue.append(p)
            elif self.PerfectPQ_set:
                # print(f'PerfectPQ: add packets to transmission queue')
                heapq.heappush(self.PerfectPQ_transmission_queue, p)
            else:
                print(f'Error: packet isnt added')
    
    def run(self):
        # enque all FP events for the first iteration where there is no blocking
        self.curr_time = 0
        self.record["Start FP"].append(Event("Start FP", self.curr_time))
        self.enque_FP(self.curr_time, 0)

        ''' main event loop '''
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            timestamp, layer, iteration = event.time, event.layer, event.iteration
            self.record[event.name].append(event)
            print(f'event: {event}')
            self.curr_time = timestamp
            if event.name == "FP_computation_done":
                if self.PerfectPQ_set:
                    if iteration != 0: # all FP events have been pushed for iteration 0
                        # 2nd iteration onwards
                        # restore previous FP compute status to not ready for next iteration
                        if layer != 0: # first layer is execluded because it's always ready to compute once gradients are received
                            self.previous_FP_layer_status[layer] = False            
                        if layer < self.num_layers-1: # unblock the compute for next FP layer
                            self.previous_FP_layer_status[layer+1] = True
                            if self.gradient_received[layer+1]:
                                next_event = Compute_Event(self.fp_layers[layer+1] + self.curr_time, "FP", layer+1, iteration, "done")
                                heapq.heappush(self.event_queue, next_event)
                                # heapq.heappush(self.event_queue, [self.fp_layers[layer+1] + self.curr_time, "FP_computation_done", layer+1,  iteration])
                        self.gradient_received[layer] = False
                # no need to handle self.FIFO_set case cause all FP events have been pushed once at the start of the new iteration 
                if layer == self.num_layers - 1: #last layer
                    # self.record.append([self.curr_time, "Start BP"])
                    self.record["Start BP"].append(Event("Start BP", self.curr_time))
                    next_event = Compute_Event(self.bp_layers[layer]+self.curr_time, "BP", layer, iteration, "done")
                    heapq.heappush(self.event_queue, next_event)
                    # heapq.heappush(self.event_queue,[self.bp_layers[layer]+self.curr_time,"BP_computation_done", layer, iteration] )

            elif (event.name == "BP_computation_done"):
                # ready to send gradient
                num_packets = self.layer_size_in_packets[layer]
                # transmission_queue.append([num_packets, layer])
                self.add_to_transmission_queue(num_packets, layer, iteration)
                # print(self.PerfectPQ_set_transmission_queue)
                if not self.InTransit: # nothing is being transimitted 
                    self.transmit_tensor(self.curr_time)
                # start BP for next layer
                if layer > 0:
                    next_event = Compute_Event(self.bp_layers[layer]+self.curr_time, "BP", layer-1, iteration, "done")
                    heapq.heappush(self.event_queue, next_event)
                    # heapq.heappush(self.event_queue,[self.bp_layers[layer]+self.curr_time,"BP_computation_done", layer-1, iteration] )

            elif event.name == "Tensor_transimission_done":
                self.InTransit = False
                self.transmit_tensor(self.curr_time)
            
            elif event.name == "Gradients_received":
                self.gradient_received[layer] = True
                # Barrier between each iteration, current implementation
                if iteration == self.TotalIteration:
                    print(f'break out of while loop : iteration: {iteration}')
                    # exit while loops
                    break
                if self.iteration_barrier == True:
                    if sum(self.gradient_received.values()) == self.num_layers: # all gradients have received
                        print(f'{self.curr_time},Start FP computation in new iteration in FIFO mode,{iteration}')
                        self.record["Start FP computation in new iteration in FIFO mode"].append(Event("Start FP computation in new iteration in FIFO mode", self.curr_time))
                        self.enque_FP(self.curr_time, iteration)
                    else:
                        print(f'Have not received all gradients')
                else: # start FP whenever previous FP layer has finished computation and gradients have been received and updated this layer 
                    print(f'self.previous_FP_layer_status[{layer}]: {self.previous_FP_layer_status[layer]}')
                    if self.previous_FP_layer_status[layer]:
                        # start computation of FP layer
                        compute_time = self.fp_layers[layer]
                        if layer == 0:
                            print(f'{self.curr_time},Start FP computation in new iteration in Perfect PQ mode,{iteration}')
                            self.record["Start FP computation in new iteration in Perfect PQ mode"].append(Event("Start FP computation in new iteration in Perfect PQ mode", self.curr_time))
                        next_event = Compute_Event(compute_time+self.curr_time, "FP", layer, iteration, "done")
                        heapq.heappush(self.event_queue, next_event)
                        # heapq.heappush(self.event_queue, [compute_time+self.curr_time, "FP_computation_done", layer, iteration])
            else:
                print(f"Error: Non-existing Event: {event}")
                break

        # print(self.record)


# compute iteration time from records
def compute_iteration_time(record, simulator):
    iteration_time_ms = 0
    iteration_start_time = 0
    for event in record["FP_computation_done"]:
        if  event.layer == simulator.num_layers -1:
            if event.iteration == 0:
                iteration_start_time = event.time
            if event.iteration == 1:
                iteration_time_ms = event.time - iteration_start_time
                break
    print(f'iteration_time_ms: {iteration_time_ms}') 
    return iteration_time_ms
    
def compute_slack_time_FIFO(record, simulator):
    # compute slack per layer for FIFO
    slack_per_layer_in_ms = {layer: 0 for layer in range(simulator.num_layers)}
    # Time difference between when gradients are computed to when gradients are needed
    # gradients_received_timestamp = {layer: 0 for layer in range(num_layers)}
    BP_computation_done_timestamp = {layer: 0 for layer in range(simulator.num_layers)}
    for event in record["BP_computation_done"]:
        if event.iteration == 0:
            BP_computation_done_timestamp[event.layer] = event.time
    for event in record["FP_computation_done"]:
        if event.iteration == 1:
            # print(f'layer: {event.layer}, FP_computation_done, {event.time}, fp_layers, {fp_layers[event.layer]}, BP compute done: { BP_computation_done_timestamp[event.layer]}')
            slack_per_layer_in_ms[event.layer] = event.time - simulator.fp_layers[event.layer] - BP_computation_done_timestamp[event.layer]

    print(f'slack_per_layer_in_ms: {slack_per_layer_in_ms}')
    return slack_per_layer_in_ms

horovod_simulator = HorovodSimulator()
horovod_simulator.run()

compute_iteration_time(horovod_simulator.record, horovod_simulator)
compute_slack_time_FIFO(horovod_simulator.record, horovod_simulator)



