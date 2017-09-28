class UniqueCounter:
    def __init__(self):
        self.ids = {}

    def add_many(self, list_of_ids):
        for i in list_of_ids:
            self.ids[i] = True
    def add(self, id):
        self.ids[id] = True

    def remove_many(self, list_of_ids):
        for i in list_of_ids:
            if i in self.ids:
                del self.ids[i]
    def contains(self,item):
        return item in self.ids
    def get_count(self):
        return len(self.ids)