from enum import Enum

class SchedulingDisc(Enum):
    PerfectPQ = 0
    FIFO = 1

class SimulatorConfig():
    def __init__(self, **kwargs):
        self._init_configs()
        for key, value in kwargs.items():
            setattr(self, key, value)

        # self.configs = {}
        # self._init_configs()

        # for key, value in kwargs.items():
        #     self.configs[key] = value

    def _init_configs(self):
        self.qdisc = SchedulingDisc.FIFO
        self.packet_size_MB = 10
            # self.["transmission_rate_Gbit_per_sec"] = 10
            # self.configs[ConfigVariables.propagation_delay_ms] = 10**(-2) # 10 us
            # self.configs[ConfigVariables.num_layers] = 50
            # self.configs[ConfigVariables.compute_time_per_iteration_ms] = 900
            # self.configs[ConfigVariables.num_workers] = 2
            # self.configs[ConfigVariables.credit_size] = 1
            # self.configs[ConfigVariables.num_prirority_queues] = 1
            # self.configs[ConfigVariables.TotalIteration] = 2
    
    def __str__(self):
        return f"qdisc_{self.qdisc.name}_pkt_{self.packet_size_MB}"

if __name__ == "__main__":
    s = SimulatorConfig(**{"qdisc": SchedulingDisc.FIFO})
    print(s.qdisc.name)
    print(s)
