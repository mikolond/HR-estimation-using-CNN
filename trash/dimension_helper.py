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
    print("conv1 1: ", dim1, dim2)
    # max pool
    dim1 = get_dims(dim1, 15, 2, 0)
    dim2 = get_dims(dim2, 10, 2, 0)
    print("maxpool 1: ", dim1, dim2)

    # layer 1
    # conv
    dim1 = get_dims(dim1, 15, 1, 0)
    dim2 = get_dims(dim2, 10, 1, 0)
    print("conv2: ", dim1, dim2)
    # max pool
    dim1 = get_dims(dim1, 15, 1, 0)
    dim2 = get_dims(dim2, 10, 1, 0)
    print("maxpool2: ", dim1, dim2)

    # layer 2
    # conv
    dim1 = get_dims(dim1, 15, 1, 0)
    dim2 = get_dims(dim2, 10, 1, 0)
    print("conv 3: ", dim1, dim2)
    # max pool
    dim1 = get_dims(dim1, 15, 1, 0)
    dim2 = get_dims(dim2, 10, 1, 0)
    print("maxpool 3: ", dim1, dim2)

    # layer 3
    # conv
    dim1 = get_dims(dim1, 12, 1, 0)
    dim2 = get_dims(dim2, 10, 1, 0)
    print("conv4: ", dim1, dim2)
    # max pool
    dim1 = get_dims(dim1, 15, 1, 0)
    dim2 = get_dims(dim2, 10, 1, 0)
    print("maxpool4: ", dim1, dim2)

    dim1 = get_dims(dim1, 1, 1, 0)
    dim2 = get_dims(dim2, 1, 1, 0)
    print("conv5: ", dim1, dim2)

    dim1 = get_dims(dim1, 3, 2, 0)
    dim2 = get_dims(dim2, 3, 2, 0)
    print("maxpool5: ", dim1, dim2)

    print("layer 5: ", dim1, dim2)

    dim1 = 35
    dim2 = 55


    dim1 = get_dims(dim1, 14, 1, 0)
    dim2 = get_dims(dim2, 3, 3, 1)
    print("layer cat: ", dim1, dim2)

    dim1 = get_dims(dim1, 9, 1, 4)
    dim2 = get_dims(dim2, 9, 1, 0)

    print("layer 1: ", dim1, dim2)  
    # dim1 = 300
    # dim1 = get_dims(dim1, 6, 1, 0)
    # dim1 = get_dims(dim1, 6, 1, 0)
    # dim1 = get_dims(dim1, 12, 1, 0)
    # dim1 = get_dims(dim1, 12, 2, 0)
    # dim1 = get_dims(dim1, 24, 1, 0)
    # print("layer 5: ", dim1)
    # dim1 = get_dims(dim1, 50, 1, 0)
    # print("layer 6: ", dim1)
    # dim1 = get_dims(dim1, 50, 1, 0)
    # print("layer 7: ", dim1)
    # dim1 = get_dims(dim1, 50, 1, 0)
    # print("layer 8: ", dim1)
    # dim1 = get_dims(dim1, 6, 1, 0)
    # print("layer 9: ", dim1)


