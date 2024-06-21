
import time


class TimeObject:
    def __init__(self):
        self.last_time = time.time()
        self.time_list = []

    def new(self, message: str = None, do_print: bool = True):
        new_time = time.time()
        duration = new_time - self.last_time
        if do_print:
            print(f"{message} duration: {duration:.6f}")
        self.time_list.append(duration)
        self.last_time = new_time
        return duration

    def total(self):
        print(f"Total duration: {sum(self.time_list):.6f}")

