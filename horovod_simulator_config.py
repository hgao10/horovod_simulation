from enum import Enum
from utils.logger import get_logger
class SchedulingDisc(Enum):
    PerfectPQ = 0
    FIFO = 1

class SimulatorConfig():
    def __init__(self, **kwargs):
        self.logger = get_logger("SimulatorConfig", "DEBUG")
        self._init_configs()
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Provided a onfig attribute that doesn's exist: {key}")

    def _init_configs(self):
        self.iteration_barrier = True
        self.qdisc = SchedulingDisc.FIFO
        # model specific 
        # smallest transmission unit tensor, could also be packet
        self.min_packet_per_layer = 10
        self.packet_size_MB = 0.44/self.min_packet_per_layer
        self.num_layers = 50
        # total model is 100MB
        self.model_size_MB = 100 #MB 

        # network specs
        self.transmission_rate_Gbit_per_sec = 10
        self.propagation_delay_ms = 10**(-2) # 10 us
        
        # Resnet50 on one P100: 900 ms 
        self.compute_time_per_iteration_ms = 900
        self.num_workers = 2
        self.credit_size = 1
        self.num_priority_queues = 1

        # Test run specs
        self.TotalIteration = 2
    
    def __str__(self):
        prop_delay = f"{self.propagation_delay_ms:.3f}".replace(".", "_")
        # print(f'prop_delay:{prop_delay}')
        # return f"pkt_{self.packet_size_MB}_layer_{self.num_layers}_msize_{self.model_size_MB}_prop_delay_{prop_delay}"
        packet_size_MB_str = f"{self.packet_size_MB:.4f}".replace(".", "_")
        return f"pkt_{packet_size_MB_str}_layer_{self.num_layers}_msize_{self.model_size_MB}_prop_delay_{prop_delay}"

if __name__ == "__main__":
    s = SimulatorConfig(**{"qdisc": SchedulingDisc.FIFO})
    s.logger.debug(s.qdisc.name)
    s.logger.debug(s)
