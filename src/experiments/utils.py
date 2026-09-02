class SaveLogger:
    def __init__(self):
        self.logs = {}
        self.steps = 0
        self.received_last_step = -1

    def log(self, losses, step):
        for key, value in losses.items():
            if key not in self.logs:
                self.logs[key] = []
            self.logs[key].append(value)

        self.steps += 1
        self.received_last_step = step
