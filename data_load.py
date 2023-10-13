from collections import namedtuple

Sample = namedtuple('Sample', ['x', 'y'])

class DataLoader:
    def __init__(self):
        self.encoding = "latin-1"
        self.path = "data"
        self.files = ["test", "train", "val"]
        self.inx_file = 0
        fX_open = open(self.path + "/" + self.files[self.inx_file] + "_text.txt",
                       mode="r", encoding=self.encoding)
        fy_open = open(self.path + "/" + self.files[self.inx_file] + "_labels.txt",
                       mode="r", encoding=self.encoding)
        inx_file = 0
        results = []
        for i in range(len(self.files)):
            with (open(self.path + "/" + self.files[i] + "_text.txt",
                       mode="r", encoding=self.encoding) as fx,
                  open(self.path + "/" + self.files[i] + "_labels.txt",
                       mode="r", encoding=self.encoding) as fy):
                while True:
                    x = fx.readline()
                    y = fy.readline()
                    if x == "":
                        break
                    else:
                        results.append(Sample(x, y))

        self.loaded_samples = results
        self.i = 0

    def __iter__(self):
        return iter(self.loaded_samples)

    def __len__(self):
        return len(self.loaded_samples)

    def __getitem__(self, key):
        if type(key) == int:
            return self.loaded_samples[key]
        elif type(key) == str:
            if key == "X":
                return [sample.x for sample in self.loaded_samples]
            if key == "y":
                return [sample.y for sample in self.loaded_samples]

    def __next__(self):
        self.i += 1
        return self.loaded_samples[self.i - 1]