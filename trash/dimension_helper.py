def get_dims(input, kernel, stride, padding=0):
    return (input + padding*2 - kernel) // stride + 1



if __name__ == "__main__":
    dim1 = 192
    dim2 = 128
    # 192 x 128
    # layer 0
    # conv
    dim1 = get_dims(dim1, 15, 1, 0)
    dim2 = get_dims(dim2, 10, 1, 0)
    # max pool
    dim1 = get_dims(dim1, 15, 2, 0)
    dim2 = get_dims(dim2, 10, 2, 0)
    print("layer 1: ", dim1, dim2)

    # layer 1
    # conv
    dim1 = get_dims(dim1, 15, 1, 0)
    dim2 = get_dims(dim2, 10, 1, 0)
    # max pool
    dim1 = get_dims(dim1, 15, 1, 0)
    dim2 = get_dims(dim2, 10, 1, 0)
    print("layer 2: ", dim1, dim2)

    # layer 2
    # conv
    dim1 = get_dims(dim1, 15, 1, 0)
    dim2 = get_dims(dim2, 10, 1, 0)
    # max pool
    dim1 = get_dims(dim1, 15, 1, 0)
    dim2 = get_dims(dim2, 10, 1, 0)
    print("layer 3: ", dim1, dim2)

    # layer 3
    # conv
    dim1 = get_dims(dim1, 12, 1, 0)
    dim2 = get_dims(dim2, 8, 1, 0)
    # max pool
    dim1 = get_dims(dim1, 12, 1, 0)
    dim2 = get_dims(dim2, 8, 1, 0)
    print("layer 4: ", dim1, dim2)

    dim1 = 300
    dim2 = 20


    dim1 = get_dims(dim1, 11, 1, 5)
    dim2 = get_dims(dim2, 12, 1, 0)
    print("layer 0: ", dim1, dim2)

    dim1 = get_dims(dim1, 9, 1, 4)
    dim2 = get_dims(dim2, 9, 1, 0)

    print("layer 1: ", dim1, dim2)  
