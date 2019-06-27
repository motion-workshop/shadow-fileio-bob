import time


class Timer:
    def __init__(self):
        self.time_0 = time.time()
        self.time_n = self.time_0

    def elapsed(self):
        tmp = time.time()
        dt = tmp - self.time_n
        self.time_n = tmp

        return dt

    def total(self):
        return time.time() - self.time_0
