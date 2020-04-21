import collections
import threading
#TODO num of priority queues -> argument
num_pq = 1
priority_queue = {pq:collections.deque() for pq in range(num_pq)}
print(priority_queue)

exitFlag = 0 

#TODO arguments
num_worker = 2
num_iterations = 3
num_layer = 2
fp_time = {layer: 3 for layer in range(num_layer)} #no_layer: execution time
bp_time = {layer:5 for layer in range(num_layer)}
# while num_iterations:
#     # forwardpropagation
#     for layer, ctime in fp_compute.items():

networkQueue = []
class networkThread(threading.Thread):

class workerThread(threading.Thread):
    def __init__(self, threadID, name, q, iters):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q
        self.iteration = iters
        self.locks = {layer: threading.Lock() for layer in range(num_layer)}  
        # self.priority_queue = {pq:collections.deque() for pq in range(num_pq)}
        self.queue = []      
    def run(self):
        print ("Starting " + self.name)
        training(self.name, self.q)
        print ("Exiting " + self.name)

    def training(self, q):
        while not exitFlag:
            #queueLock.acquire()
            for i in range(self.iteration):
                for layer in fp_time:
                    if self.locks[layer].acquire() or i == 0:
                        print(f'{self.name}: compute FP in layer {layer}')                    
                        time.sleep(fp_time[layer])

                for layer in bp_time:
                    print(f'{self.name}: comptue BP in layer {layer}')
                    time.sleep(bp_time[layer])
                    self.queue.append([self.name, self.threadID, layer, self.iteration])

    def network_agent(self):



        


