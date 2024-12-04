def test(a,b=1,c = 2):

    return a+b+c

def test2(a ,b = None,c = 2):
    return test(a,b,c)


res = test2(1)
print(res)