import collections
import time
import heapq
from horovod_simulator_config import SimulatorConfig, SchedulingDisc
from utils.logger import get_logger
import typing
from queue import PriorityQueue

class Packet():
    def __init__(self, iteration_idx, layer_idx, packet_idx, packet_size_MB):
        # global packet_size_MB
        self.logger = get_logger("Packet", "DEBUG")
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
    def __init__(self, name, end_time, start_time):
        self.name = name
        self.time = end_time
        self.start_time = start_time
        self.duration = self.time - self.start_time

    def __lt__(self, other):
        return self.time < other.time
    
    def __str__(self):
        return (f'Time_ms, {self.time}, Event, {self.name}')


class Compute_Event(Event):
    def __init__(self, time, start_time, direction, layer, iteration, state):
        # Forward or Backward
        name = direction + '_computation_' + state
        super().__init__(name, time, start_time)    
        self.direction = direction 
        self.iteration = iteration
        self.layer = layer
        # start or done
        self.state = state
    
    def __str__(self):
        return (f'Time_ms, {self.time}, Event, {self.name}, Iter, {self.iteration}, Layer, {self.layer}')


class Transmit_Event(Event):
    def __init__(self, time,start_time, state, iteration, layer, packet_idx):
        # start or done
        self.state = state
        name = 'Tensor_transimission_' + state 
        super().__init__(name, time, start_time)
        self.iteration = iteration
        self.layer = layer
        self.packet_idx = packet_idx
    
    def __str__(self):
        return (f'Time_ms, {self.time}, Event, {self.name}, Iter, {self.iteration}, Layer, {self.layer}, Packet_idx, {self.packet_idx}')

class RingAllReduce_Event(Event):
    def __init__(self, time, start_time, state, iteration, priority):
        self.state = state
        name = "RingAllReduce_" + state
        super().__init__(name, time, start_time)
        self.iteration = iteration
        self.priority = priority
        self.layer = priority

    def __str__(self):
        return (f'Time_ms, {self.time}, Event, {self.name}, Iter, {self.iteration}, Priority, {self.priority}')


class Gradients_Event(Event):
    def __init__(self, time, start_time, iteration, layer):
        super().__init__("Gradients_received", time, start_time)
        self.iteration = iteration
        self.layer = layer
    
    def __str__(self):
        return (f'Time_ms, {self.time}, Event, {self.name}, Iter, {self.iteration}, Layer, {self.layer}')

class Tensor():
    def __init__(self, layer, size):
        self.layer = layer
        self.size = size
        
    def __lt__(self, other):
        return self.layer < other.layer 
        
class SingleReduce():
    def __init__(self):
        self.tensors = [] # a list of tensors
        self.priority = 0
        self.size = 0
        self.progress = 0 # 2(n-1) times where n is the number of workers
        self.gradient_computed_status = {}
        self.logger = get_logger("SingleReduce", "DEBUG")
        self.iteration = 0
    def add_tensor(self, tensor):
        self.tensors.append(tensor)
        self.size += tensor.size
        # update priority
        self.priority = min(self.tensors, key=lambda k:k.layer).layer
        self.logger.debug(f"add tensor {tensor.layer}")
        self.logger.debug(f"SingleReduce priority: {self.priority}")
        self.gradient_computed_status[tensor.layer] = False
    
    def set_gradient_available(self, layer):
        self.gradient_computed_status[layer] = True

    def ready_to_be_sent(self):
        if sum(self.gradient_computed_status.values()) == len(self.gradient_computed_status):
            return True
        return False
    
    def clear_compute_status(self):
        for key in self.gradient_computed_status.keys():
            self.gradient_computed_status[key] = False
        self.logger.debug("Clearing gradient compute status")
    
    def __lt__(self, other):
        return self.priority < other.priority 


class RingAllReduce():
    # tensors: key layer, value = size in MB
    def __init__(self, num_partitions) -> None:
        self.num_partitions = num_partitions #num of workers
        # key: priority of each reduce operation, value: layer of tensors
        self.reducelists = {} 
        self.logger = get_logger("RingAllReduce", "DEBUG")
        self.fusion_reduce_lookup = {}

    def map_tensors(self, tensors: typing.List, fusion_buffer_size_MB):
        # map tensor i to one ring allreduce operation j 
        one_reduce = SingleReduce()
        added_t_count = 0
        # TODO: refactor to be more straightforward! 
        while added_t_count < len(tensors):
            tensor = tensors[added_t_count]
            if one_reduce.size + tensor.size < fusion_buffer_size_MB:
                one_reduce.add_tensor(tensor)
                added_t_count += 1
                if added_t_count < len(tensors) - 1:
                    # add last reduce operation to the list if its the last tensor       
                    continue

            self.logger.debug(f"one_reduce size {one_reduce.size} tensor.size {tensor.size}")
            if one_reduce.size == 0 or one_reduce.size == 0.0:
                self.logger.error("Reduce operation is of size ZERO")
            for t in one_reduce.tensors:
                self.logger.debug(f"Tensor {t.layer} in reduce {one_reduce.priority} ")
                self.fusion_reduce_lookup[t.layer] = one_reduce
            self.reducelists[one_reduce.priority] = one_reduce
            one_reduce = SingleReduce()

class HorovodSimulator():
    def __init__(self, config):
        self.logger = get_logger("HorovodSimulator", "DEBUG")
        # key: event name value: event obj
        self.record = collections.defaultdict(list)
        
        self.config = config
        
        # FIFO queue
        self.transmission_queue = collections.deque()
        # minheap sorted by priority
        self.PerfectPQ_transmission_queue = []

        # event queue is used as a minheap sorted by timestamp
        self.event_queue = []
        self.curr_time = 0

        self.fp_total_time_ms = (1/3) * self.config.compute_time_per_iteration_ms
        self.bp_total_time_ms = (2/3) * self.config.compute_time_per_iteration_ms
        # To simplify computation time in FP: assum each layer takes less then d ms to compute than previous layer and the last layer takes 0 ms
        self.fp_diff_per_layer_ms = 2 * self.fp_total_time_ms / (self.config.num_layers * (self.config.num_layers-1))
        self.logger.debug(f"self.fp_diff_per_layer_ms: {self.fp_diff_per_layer_ms}")
        self.fp_first_layer_ms = 2 * self.fp_total_time_ms / self.config.num_layers
        # Same simplification applies to BP except its in ascending order
        self.bp_diff_per_layer_ms = 2 * self.bp_total_time_ms / (self.config.num_layers * (self.config.num_layers-1))
        self.logger.debug(f"self.bp_diff_per_layer_ms: {self.bp_diff_per_layer_ms}")
        self.fp_layers = {layer: self.fp_first_layer_ms - layer * self.fp_diff_per_layer_ms for layer in range(self.config.num_layers)}
        self.fp_layers[self.config.num_layers -1] = self.fp_diff_per_layer_ms
    
        self.logger.debug(f"self.fp_layers: {self.fp_layers}")

        self.bp_layers = {layer: layer * self.bp_diff_per_layer_ms for layer in range(self.config.num_layers)}
        self.bp_layers[0] = self.bp_diff_per_layer_ms
        self.logger.debug(f"self.bp_layers:{self.bp_layers}")
        self.check_computation_time_per_layer()

        # To simplify layer size, first half of the layers are size S and the next 1/4 half of the layers are of size 4s and the last 1/4 of size 12s
        self.min_layer_size_MB = 2 * self.config.model_size_MB / ( 9 * self.config.num_layers)
        if self.min_layer_size_MB == 0.0:
            self.logger.warn("Min layer size in MB is zero")
        self.config.packet_size_MB = self.min_layer_size_MB/self.config.min_packet_per_layer
        self.logger.debug(f"min_layers in MB: {self.min_layer_size_MB}, packet_size_MB: {self.config.packet_size_MB}")
        self.layer_size = {}
        # number of packets to be sent/received per layer  
        self.layer_size_in_packets = {}
        self.calculate_layer_size() 
        self.tensors = []
        self.construct_tensors()   
        self.curr_fusion = []

        # self._init_layer_size_in_packets()
        self.logger.debug(f"layer_size_in packets: {self.layer_size_in_packets}")
        self.check_layer_size_in_packets()

        # initialize ring all reduce operations
        self.ringallreduce = RingAllReduce(self.config.num_workers)
        self.ringallreduce.map_tensors(self.tensors, self.config.fusion_buffer_size_MB)
        self.ringallreduce_pq = PriorityQueue()
        self.ringallreduce_fifo = collections.deque()

        # Test run specs
        self.config.TotalIteration = 2

        # horovod simulator status registers
        self.gradient_received = {layer: False for layer in range(self.config.num_layers)}
        self.received_tensor_count = {layer: 0 for layer in range(self.config.num_layers)}
        # tracks computation completion status per layer according to data dependency
        # the first layer is always true as it doesn't depend on any previous layer
        self.previous_FP_layer_status = {layer: False for layer in range(self.config.num_layers)}
        self.previous_FP_layer_status[0] = True
        self.increment_iteration_status = {i: False for i in range(self.config.TotalIteration+1)}

        self.InTransit = False

        # non-essential compute time
        self.allReduceComputeTime_ms = 0
        self.ApplyLayerGradient_ms = 0

        # The transmission delay is the amount of time required for the router to push out the packet.
        # The propagation delay, is the time it takes a bit to propagate from one router to the next.
        self.tensor_transmittion_time_ms = self.config.packet_size_MB * 8 /self.config.transmission_rate_Gbit_per_sec 
        self.logger.debug(f"tensor transmission time: {self.tensor_transmittion_time_ms}")
        #TODO simplied version, each worker sends the entire amount of gradient per layer at once instead of gradient/num_workers for num_workers times, refer to ring allreduce paper
         
        # parameter server model, send all tensors at once and wait for the PS respond back thus * 2 
        self.TotalAllReduceTime = self.allReduceComputeTime_ms + self.ApplyLayerGradient_ms + 2* (self.tensor_transmittion_time_ms + self.config.propagation_delay_ms) # compute + network roundtrip time
        self.logger.debug(f"totalallreducetime: {self.TotalAllReduceTime}")

        # TODO future feature: multiple priority queues
        self.config.num_priority_queues = 1
        self.priority_queues = {}
        # TODO incorperate credit_size in non perfect priority queue situation where packets can only be pre-empted if there is enough credit left 
        self.config.credit_size = 1


    def check_layer_size_in_packets(self):
        for layer, num in self.layer_size_in_packets.items():
            if num == 0 or num == 0.0:
                self.logger.warn(f"Layer {layer} contains zero transmission packets")

    def check_computation_time_per_layer(self):
        for layer, time in self.fp_layers.items():
            if time == 0.0:
                self.logger.warn(f"FP layer[{layer}] is zero")
        for layer, time in self.bp_layers.items():
            if time == 0.0:
                self.logger.warn(f"BP layer[{layer}] is zero")
        
    def set_model_compute_time_per_iteration_ms(self, time):
        self.config.compute_time_per_iteration_ms = time
        self.fp_total_time_ms = (1/3) * self.config.compute_time_per_iteration_ms
        self.bp_total_time_ms = (2/3) * self.config.compute_time_per_iteration_ms
        # To simplify computation time in FP: assum each layer takes less then d ms to compute than previous layer and the last layer takes 0 ms
        self.fp_diff_per_layer_ms = 2 * self.fp_total_time_ms // (self.config.num_layers * (self.config.num_layers-1))
        self.fp_first_layer_ms = 2 * self.fp_total_time_ms // self.config.num_layers
        # Same simplification applies to BP except its in ascending order
        self.bp_diff_per_layer_ms = 2 * self.bp_total_time_ms // (self.config.num_layers * (self.config.num_layers-1))
        self.fp_layers = {layer: self.fp_first_layer_ms - layer * self.fp_diff_per_layer_ms for layer in range(self.config.num_layers)}
        self.bp_layers = {layer: layer * self.bp_diff_per_layer_ms for layer in range(self.config.num_layers)}

    def remove_iteration_barrier(self):
        self.config.iteration_barrier = False
    def calculate_layer_size(self):
        for layer in range(self.config.num_layers):
            if layer <= self.config.num_layers//2:
                self.layer_size[layer] = self.min_layer_size_MB
                self.layer_size_in_packets[layer] = self.config.min_packet_per_layer
            elif  self.config.num_layers//2 <layer <= 3*self.config.num_layers//4:
                self.layer_size[layer] = 4 * self.min_layer_size_MB
                self.layer_size_in_packets[layer] = 4 * self.config.min_packet_per_layer
            else:
                self.layer_size[layer] = 12 * self.min_layer_size_MB
                self.layer_size_in_packets[layer] = 12 * self.config.min_packet_per_layer
    
    def construct_tensors(self):
        for layer, tensor_size in self.layer_size.items():
            self.logger.debug(f"construct tensor of layer {layer}, size {tensor_size}")
            t = Tensor(layer, tensor_size)
            self.tensors.append(t)

    def _init_priority_queues(self):
            for i in range(self.config.num_priority_queues):
                self.priority_queues[i] = collections.deque() 
    
    def enque_FP(self, curr_time, iteration):
        for layer, compute_time in self.fp_layers.items():
            next_event = Compute_Event(compute_time + curr_time, curr_time, "FP", layer, iteration, "done")
            heapq.heappush(self.event_queue, next_event)
            self.gradient_received[layer] = False
            curr_time += compute_time

    def transmit_tensor_fusion(self, fusion_reduce):
        self.logger.debug(f"fusion_reduce.size {fusion_reduce.size}")
        partion_size_MB = fusion_reduce.size/self.config.num_workers
        self.logger.debug(f"going to send fusion reduce with priority {fusion_reduce.priority}, p size: {partion_size_MB}")
        # assuming there is zero computation time to apply the partial gradients
        # artifically included propagation delay to transmission time to indicate the next transmission can't start until gradients from 
        # neighbors has been received which includes propagation time
        one_iteration_transmit_partition_duration_ms = partion_size_MB * 8/self.config.transmission_rate_Gbit_per_sec + self.config.propagation_delay_ms
        # receive_one_partition_time_ms = one_iteration_transmit_partition_duration_ms + self.config.propagation_delay_ms 
        total_transmit_fusion_time_ms = 2 * (self.config.num_workers - 1) * one_iteration_transmit_partition_duration_ms 
        
        self.logger.debug(f"add next ringallreduce event {total_transmit_fusion_time_ms + self.curr_time}")
        next_event = RingAllReduce_Event(total_transmit_fusion_time_ms + self.curr_time, self.curr_time, "done", fusion_reduce.iteration, fusion_reduce.priority )        
        heapq.heappush(self.event_queue, next_event)
        self.InTransit = True
        self.logger.debug(f"transmit_tensor in RingAllReduce")
        return 

    def transmit_packet(self, packet):
        # self.logger.debug(f'transimitting packet: iter:{packet.iteration_idx}, layer: {packet.layer_idx}, id: {packet.packet_idx}')
        next_event = Transmit_Event(self.tensor_transmittion_time_ms + self.curr_time, self.curr_time,"done", packet.iteration_idx, packet.layer_idx, packet.packet_idx)
        heapq.heappush(self.event_queue, next_event)
        if packet.packet_idx == self.layer_size_in_packets[packet.layer_idx] - 1: # last packet in the layer, assume that there is no OOO transmission
            if not self.increment_iteration_status[packet.iteration_idx+1]: # any layer that finishes transmitting all gradients will increament the iteration for that layer
                packet.iteration_idx += 1
            next_event = Gradients_Event(self.TotalAllReduceTime + self.curr_time, self.curr_time,packet.iteration_idx, packet.layer_idx)
            heapq.heappush(self.event_queue, next_event)
        self.InTransit = True        
        return 

    # transmission queue: comprised of packet_id (iteration_idx, layer_idx, packet_idx)
    def transmit_tensor(self):
        # if self.FIFO_set and self.transmission_queue:
        if self.config.qdisc == SchedulingDisc.FIFO and self.transmission_queue:
            packet = self.transmission_queue.popleft()
            self.transmit_packet(packet)
        elif self.config.qdisc == SchedulingDisc.PerfectPQ and self.PerfectPQ_transmission_queue:
            packet = heapq.heappop(self.PerfectPQ_transmission_queue)
            self.logger.debug(f"Debug, pop packet off PerfectPQ_transmission_queue: {packet}")
            self.transmit_packet(packet)
        elif self.config.qdisc == SchedulingDisc.RingAllReducePQ and not self.ringallreduce_pq.empty():
            self.logger.debug(f"transmit tensor for RingallReducePQ")
            # if not self.ringallreduce_pq.empty():
            self.logger.debug(f"ringallreduce_pq is not empty: {self.ringallreduce_pq}")
            fusion_reduce = self.ringallreduce_pq.get(block=False)           
            self.transmit_tensor_fusion(fusion_reduce)
        elif self.config.qdisc == SchedulingDisc.RingAllReduceFIFO and self.ringallreduce_fifo:
            fusion_reduce = self.ringallreduce_fifo.popleft()
            self.transmit_tensor_fusion(fusion_reduce)

    def add_to_transmission_queue(self, num_packets, layer, iteration):
        for i in range(num_packets):
            p = Packet(iteration, layer, i, self.config.packet_size_MB)
            # if self.FIFO_set:
            if self.config.qdisc == SchedulingDisc.FIFO:
                self.logger.debug(f'self.FIFO_set: add packets to transmission queue')
                self.transmission_queue.append(p)
            elif self.config.qdisc == SchedulingDisc.PerfectPQ:
                # self.logger.debug(f'PerfectPQ: add packets to transmission queue')
                heapq.heappush(self.PerfectPQ_transmission_queue, p)
            else:
                self.logger.error(f'Packets are not being added to the transmission queue')
    
    def run(self):
        # enque all FP events for the first iteration where there is no blocking
        self.curr_time = 0
        self.record["Start FP"].append(Event("Start FP", self.curr_time, self.curr_time))
        self.enque_FP(self.curr_time, 0)

        ''' main event loop '''
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            timestamp, layer, iteration = event.time, event.layer, event.iteration
            self.record[event.name].append(event)
            self.logger.debug(f'event: {event}')
            self.curr_time = timestamp
            if event.name == "FP_computation_done":
                # if self.PerfectPQ_set:
                if self.config.iteration_barrier == False:
                    if self.config.qdisc == SchedulingDisc.PerfectPQ or self.config.qdisc == SchedulingDisc.RingAllReducePQ:
                        if iteration != 0: # all FP events have been pushed for iteration 0
                            # 2nd iteration onwards
                            # restore previous FP compute status to not ready for next iteration
                            if layer != 0: # first layer is execluded because it's always ready to compute once gradients are received
                                self.previous_FP_layer_status[layer] = False            
                            if layer < self.config.num_layers-1: # unblock the compute for next FP layer
                                self.logger.debug(f"FP layer {layer} done, check if gradients received for {layer+1}")
                                self.previous_FP_layer_status[layer+1] = True
                                if self.gradient_received[layer+1]:
                                    self.logger.debug(f"gradient_received[{layer+1}]: {self.gradient_received[layer+1]}")
                                    next_event = Compute_Event(self.fp_layers[layer+1] + self.curr_time, self.curr_time, "FP", layer+1, iteration, "done")
                                    heapq.heappush(self.event_queue, next_event)
                                    # heapq.heappush(self.event_queue, [self.fp_layers[layer+1] + self.curr_time, "FP_computation_done", layer+1,  iteration])
                            self.gradient_received[layer] = False
                # no need to handle self.FIFO_set case cause all FP events have been pushed once at the start of the new iteration 
                if layer == self.config.num_layers - 1: #last layer
                    # self.record.append([self.curr_time, "Start BP"])
                    self.record["Start BP"].append(Event("Start BP", self.curr_time, self.curr_time))
                    next_event = Compute_Event(self.bp_layers[layer]+self.curr_time, self.curr_time,"BP", layer, iteration, "done")
                    heapq.heappush(self.event_queue, next_event)
                    # heapq.heappush(self.event_queue,[self.bp_layers[layer]+self.curr_time,"BP_computation_done", layer, iteration] )

            elif (event.name == "BP_computation_done"):
                # ready to send gradient
                # look up which reduce operation this layer belongs to
                if self.config.qdisc == SchedulingDisc.RingAllReducePQ or self.config.qdisc == SchedulingDisc.RingAllReduceFIFO:
                    if iteration == self.config.TotalIteration - 1 :
                        self.logger.debug(f'break out of while loop : iteration: {iteration}')
                        # exit while loops
                        break
                    fusion_reduce = self.ringallreduce.fusion_reduce_lookup[layer]
                    fusion_reduce.iteration = iteration
                    self.logger.debug(f"BP computation done for ring all reduce {fusion_reduce.priority} layer {layer}")
                    fusion_reduce.set_gradient_available(layer)
                    # if all tensors in the reduce operation are computed, move it to allreduce priority queue
                    if fusion_reduce.ready_to_be_sent():
                        if self.config.qdisc == SchedulingDisc.RingAllReducePQ:
                            self.ringallreduce_pq.put(fusion_reduce)
                        else:
                            self.ringallreduce_fifo.append(fusion_reduce)
                        fusion_reduce.clear_compute_status()


                elif self.config.qdisc == SchedulingDisc.PerfectPQ or self.config.qdisc == SchedulingDisc.FIFO:
                    num_packets = self.layer_size_in_packets[layer]
                    self.add_to_transmission_queue(num_packets, layer, iteration)

                if not self.InTransit: # nothing is being transimitted 
                    self.transmit_tensor()

                # start BP for next layer
                if layer > 0:
                    self.logger.debug(f"Debug: add next BP layer to the queue: {self.bp_layers[layer-1]+self.curr_time}")
                    next_event = Compute_Event(self.bp_layers[layer-1]+self.curr_time, self.curr_time, "BP", layer-1, iteration, "done")
                    heapq.heappush(self.event_queue, next_event)

            elif event.name == "Tensor_transimission_done":
                self.InTransit = False
                self.transmit_tensor()
            
            elif event.name == "RingAllReduce_done":
                self.InTransit = False
                # set gradients received for layers included in the ringallreduce to ready
                received_layers = []
                for tensor in self.ringallreduce.reducelists[event.priority].tensors:
                    self.logger.debug(f"RingAllReduce done: set gradient received [{tensor.layer}]")
                    self.gradient_received[tensor.layer] = True
                    received_layers.append(tensor.layer)

                received_layers.sort()
                # only need to kickstart the lowest layer of FP in the received fusion, the rest will be evoked once the lower FP is done
                layer = received_layers[0]
                if self.config.iteration_barrier == True:
                    if sum(self.gradient_received.values()) == self.config.num_layers: # all gradients have received
                        self.logger.debug(f'{self.curr_time},Start FP computation in new iteration in RingAllReduce mode,{iteration}')
                        self.record["Start FP computation in new iteration in RingAllReduce mode"].append(Event("Start FP computation in new iteration in RingAllReduce mode", self.curr_time, self.curr_time))
                        self.enque_FP(self.curr_time, iteration+1)
                    else:
                        self.logger.debug(f'Have not received all gradients')
                else: # start FP whenever previous FP layer has finished computation and gradients have been received and updated this layer                 
                    if self.previous_FP_layer_status[layer]:
                        self.logger.debug(f"start FP layer computation: {layer}")
                        compute_time = self.fp_layers[layer]
                        if layer == 0:
                            self.logger.debug(f'{self.curr_time},Start FP computation in new iteration in Perfect PQ mode,{iteration}')
                            self.record["Start FP computation in new iteration in RingAllReduce mode"].append(Event("Start FP computation in new iteration in RingAllReduce mode", self.curr_time,self.curr_time))
                        next_event = Compute_Event(compute_time+self.curr_time, self.curr_time,"FP", layer, iteration + 1, "done")
                        heapq.heappush(self.event_queue, next_event)                    

                self.transmit_tensor()
            
            elif event.name == "Gradients_received":
                self.gradient_received[layer] = True
                # Barrier between each iteration, current implementation
                if iteration == self.config.TotalIteration:
                    self.logger.debug(f'break out of while loop : iteration: {iteration}')
                    # exit while loops
                    break
                if self.config.iteration_barrier == True:
                    if sum(self.gradient_received.values()) == self.config.num_layers: # all gradients have received
                        self.logger.debug(f'{self.curr_time},Start FP computation in new iteration in FIFO mode,{iteration}')
                        self.record["Start FP computation in new iteration in FIFO mode"].append(Event("Start FP computation in new iteration in FIFO mode", self.curr_time, self.curr_time))
                        self.enque_FP(self.curr_time, iteration)
                    else:
                        self.logger.debug(f'Have not received all gradients')
                else: # start FP whenever previous FP layer has finished computation and gradients have been received and updated this layer 
                    self.logger.debug(f'self.previous_FP_layer_status[{layer}]: {self.previous_FP_layer_status[layer]}')
                    if self.previous_FP_layer_status[layer]:
                        # start computation of FP layer
                        self.logger.debug(f"start FP layer computation: {layer}")
                        compute_time = self.fp_layers[layer]
                        if layer == 0:
                            self.logger.debug(f'{self.curr_time},Start FP computation in new iteration in Perfect PQ mode,{iteration}')
                            self.record["Start FP computation in new iteration in Perfect PQ mode"].append(Event("Start FP computation in new iteration in Perfect PQ mode", self.curr_time,self.curr_time))
                        next_event = Compute_Event(compute_time+self.curr_time, self.curr_time,"FP", layer, iteration, "done")
                        heapq.heappush(self.event_queue, next_event)
                        # heapq.heappush(self.event_queue, [compute_time+self.curr_time, "FP_computation_done", layer, iteration])
            else:
                self.logger.error(f"Error: Non-existing Event: {event}")
                break

        # self.logger.debug(self.record)

# compute iteration time from records
def compute_iteration_time(record, simulator):
    logger = get_logger("compute_iteration_time", "DEBUG")
    iteration_time_ms = 0
    iteration_start_time = 0
    for event in record["FP_computation_done"]:
        if  event.layer == simulator.config.num_layers -1:
            if event.iteration == 0:
                iteration_start_time = event.time
            if event.iteration == 1:
                iteration_time_ms = event.time - iteration_start_time
                break
    logger.debug(f'iteration_time_ms: {iteration_time_ms}') 
    return iteration_time_ms
    
def compute_slack_time_FIFO(record, simulator):
    '''
        compute slack per layer for FIFO
        Time difference between when gradients are computed to when gradients are needed
        Gradients computed timestamp @ layer i  = BP computation time done @ layer i
        Gradients consumed timestamp @ layer i = FP computation start @ layer i
                                            = FP computation done @ layer i - FP computation duration @ layer i 
    '''
    logger = get_logger("compute_slack_time_FIFO", "DEBUG")
    slack_per_layer_in_ms = {layer: 0 for layer in range(simulator.config.num_layers)}
    BP_computation_done_timestamp = {layer: 0 for layer in range(simulator.config.num_layers)}
    for event in record["BP_computation_done"]:
        if event.iteration == 0:
            BP_computation_done_timestamp[event.layer] = event.time
    for event in record["FP_computation_done"]:
        if event.iteration == 1:
            # print(f'layer: {event.layer}, FP_computation_done, {event.time}, fp_layers, {fp_layers[event.layer]}, BP compute done: { BP_computation_done_timestamp[event.layer]}')
            slack_per_layer_in_ms[event.layer] = event.time - simulator.fp_layers[event.layer] - BP_computation_done_timestamp[event.layer]

    logger.debug(f'slack_per_layer_in_ms: {slack_per_layer_in_ms}')
    return slack_per_layer_in_ms

def compute_iteration_and_slack(record, simulator):
    compute_iteration_time(record, simulator)
    compute_slack_time_FIFO(record, simulator)

def test_run(config):
    horovod_simulator = HorovodSimulator(config)
    horovod_simulator.run()
    compute_iteration_and_slack(horovod_simulator.record, horovod_simulator)


if __name__ == "__main__":
    def test1():
        test_FIFO_s = SimulatorConfig(**{"num_layers":10, "propagation_delay_ms":5})
        horovod_simulator = HorovodSimulator(test_FIFO_s)
        horovod_simulator.run()
        compute_iteration_and_slack(horovod_simulator.record, horovod_simulator)

    def test2():
        test_PerfectPQ_s = SimulatorConfig(**{"iteration_barrier": False, "qdisc": SchedulingDisc.PerfectPQ, "num_layers":10, "propagation_delay_ms":5})
        horovod_simulator = HorovodSimulator(test_PerfectPQ_s)
        horovod_simulator.run()
        compute_iteration_and_slack(horovod_simulator.record, horovod_simulator)


    def test3():
        network_bd = 50

        test_FIFO_s = SimulatorConfig(**{"qidsc": SchedulingDisc.FIFO, "transmission_rate_Gbit_per_sec": network_bd})
        horovod_simulator = HorovodSimulator(test_FIFO_s)
        horovod_simulator.run()
        compute_iteration_and_slack(horovod_simulator.record, horovod_simulator)


    def test4():
        network_bd = 50

        test_PerfectPQ_s = SimulatorConfig(**{"iteration_barrier": False, "qdisc": SchedulingDisc.PerfectPQ, "transmission_rate_Gbit_per_sec": network_bd })    
        horovod_simulator = HorovodSimulator(test_PerfectPQ_s)
        horovod_simulator.run()
        compute_iteration_and_slack(horovod_simulator.record, horovod_simulator)

    
    def test_ring_allreduce_pq():
        config = SimulatorConfig(**{"iteration_barrier": False, "qdisc": SchedulingDisc.RingAllReducePQ,"num_layers":10, "propagation_delay_ms":5})
        test_run(config)
    
    def test_ring_allreduce_fifo():
        # fifo explicitly has iteration barrier in place
        config = SimulatorConfig(**{"iteration_barrier": True, "qdisc": SchedulingDisc.RingAllReduceFIFO,"num_layers":10, "propagation_delay_ms":5})
        test_run(config)

    # test1()
    test_ring_allreduce_fifo()    


