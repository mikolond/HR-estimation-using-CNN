def get_dims(input, kernel, stride, padding=0):
    return (input + padding*2 - kernel) // stride + 1



if __name__ == "__main__":
    dim1 = 150
    dim2 = 0
    for i in range(11):
        # conv
        dim1 = get_dims(dim1, 11, 1, 0)
        dim2 = get_dims(dim2, 0, 1, 0)
        print(f"Dim1: {dim1}, iteration: {i}")
    for i in range(4):
        # max pool
        dim1 = get_dims(dim1, 11, 1, 0)
        dim2 = get_dims(dim2, 0, 1, 0)
        print(f"Dim1: {dim1}, iteration: {i}")

    print("final dimensions")
    print(f"Dim1: {dim1}, iteration: {i}")
    print(f"Dim2: {dim2}, iteration: {i}")