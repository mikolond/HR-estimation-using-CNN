def get_dims(input, kernel, stride, padding=0):
    return (input + padding*2 - kernel) // stride + 1



if __name__ == "__main__":
    dim1 = 150
    dim2 = 0
    # dim1 = get_dims(dim1, 30, 1, 0)
    for i in range(10):
        # conv
        dim1 = get_dims(dim1, 16, 1, 0)
        dim2 = get_dims(dim2, 0, 1, 1)
        # print(f"Dim1: {dim1}, iteration: {i}")
        # print(f"Dim2: {dim2}, iteration: {i}")
        # max pool
        if i % 3 == 0:
            dim1 = get_dims(dim1, 5, 1, 0)
            dim2 = get_dims(dim2, 0, 10, 0)
        dim1 = get_dims(dim1, 0, 1, 0)
        # dim2 = get_dims(dim2, 0, 2, 0)
        print(f"Dim1: {dim1}, iteration: {i}")
        print(f"Dim2: {dim2}, iteration: {i}")

    print("final dimensions")
    print(f"Dim1: {dim1}, iteration: {i}")
    print(f"Dim2: {dim2}, iteration: {i}")