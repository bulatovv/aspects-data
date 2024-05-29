
def read_file(filename: str):
    with open(filename, "r") as file:
        result = file.read().strip()
    return result
