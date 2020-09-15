from multiprocessing import Pool


def myFunc(x):
    return x*2

if __name__ == '__main__':
    list = [[1, 2], [3,4]]
    pool = Pool(processes=1)
    print(pool.map(myFunc, list))


