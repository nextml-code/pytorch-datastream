import json


def read(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)


def write(obj, filepath):
    with open(filepath, 'w') as file:
        return json.dump(obj, file, indent=4)
