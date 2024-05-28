import numpy as np
# zad 1


def zad1 ():
    a = np.ones((3, 5, 7), dtype=int)
    print(np.sum(a, axis = 0))
    print(np.sum(a, axis = 1))
    print(np.sum(a, axis = 2))

zad1()

# zad 2
def zad2 ():
    a = np.linspace(0, 19, num=20).astype(int)
    print(a)
    print(a[5:13])

zad2()

#zad3

def zad3 ():
    a = np.random.normal(0.0, 1.0, (42, 13))
    print(a)
    print(a[:,-1])
    print(a[3,:])
zad3()