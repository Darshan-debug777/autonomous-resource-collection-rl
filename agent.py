
class Agent:

    def __init__(self, capacity=50):

        self.capacity = capacity
        self.current_weight = 0
        self.total_value = 0
        self.collected = []

    def collect(self, resource):

        weight = resource["weight"]

        if self.current_weight + weight > self.capacity:
            return False

        self.current_weight += weight
        self.total_value += resource["value"]

        self.collected.append(resource)

        return True

    def get_results(self):

        return {
            "weight": self.current_weight,
            "value": self.total_value,
            "collected": self.collected
        }
