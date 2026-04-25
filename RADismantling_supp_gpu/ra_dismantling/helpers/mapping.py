def load_mapping(
    file, key=None, value=None, separator=" ", key_dtype=str, value_dtype=str
):
    mapping = dict()

    with open(file) as f:
        header = f.readline().strip()
        header = header.split(separator)
        columns = dict(zip(header, range(len(header))))

        for line in f:
            line = line.strip().split(separator)
            mapping[key_dtype(line[columns[key]])] = value_dtype(line[columns[value]])

    return mapping
