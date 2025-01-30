def get_dims(input, kernel, stride, padding=0):
    return (input + padding*2 - kernel) // stride + 1



if __name__ == "__main__":
    dim1 = 26
    dim2 = 19
    for i in range(4):
        # conv
        dim1 = get_dims(dim1, 12, 1, 0)
        dim2 = get_dims(dim2, 10, 1, 0)
        # max pool
        dim1 = get_dims(dim1, 15, 1)
        dim2 = get_dims(dim2, 10, 1)
        print(f"Dim1: {dim1}, iteration: {i}")
        print(f"Dim2: {dim2}, iteration: {i}")