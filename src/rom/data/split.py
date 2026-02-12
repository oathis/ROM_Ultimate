def make_split(data, ratio=0.8):
    cutoff = int(len(data) * ratio)
    return data[:cutoff], data[cutoff:]
