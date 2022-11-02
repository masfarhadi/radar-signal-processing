""""
The first session of Radar Class
"""
import numpy
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin as sin
import time
from matplotlib2tikz import save as tikz_save
from matplotlib import ticker



def my_cool_function(n):
    """
    the test function
    parameters
    --------
    :param n: (float) wait time in seconds.
    """
    print("Function called with parameter ", n)
    time.sleep(n)


def my_func(a, b, c=None):
    """

    :param a:
    :param b:
    :param c:
    :return:
    """
    if c is None:
        return a+b
    else:
        return c


def fibo_sequence(num):
    """
    fibonacci sequence
    :param num:
    """
    fib_seq = [1, 1]

    for cnt in range(num):
        fib_seq.append(fib_seq[-1]+fib_seq[-2])

    print('fibonacci sequence is equal to : ', fib_seq)


def add(a_val, b_val, squared=None):
    """
    Example for optional parameter
    """
    if squared:
        return (a_val+b_val)**2
    else:
        return a_val+b_val


def add_hidden(a_val):
    """
    Example for scope considerations (b is not defined but used)
    """
    return a_val+b

def add_sub(a, b):
    """

    :param a: first input number
    :param b: second input number
    :return: list of return values
    """
    return [a+b, a-b]


def my_test_fun(a, b, c=None):
    """

    :param a:
    :param b:
    :param c:
    :return:
    """
    if c is None:
        return a+b
    else:
        return c

if __name__ == "__main__":
    N = 1
    t = time.time()
    # call a function
    my_cool_function(N)
    print('Script took %d ms' % (1000*(time.time() - t)))

    for cnt in range(10):
        print(cnt)

    a = 4.0
    b = 3.0
    c = 1.0
    if a >= b and c >= 0:
        print('\n a is %f: ' % a)
    elif c == 0:
        pass
    else:
        print(c)

sum_value = my_func(2, 3)
print('\n the value of sum is equal to : %f' % sum_value)

print('\n min a and b is equal to : %f ' % min(a, b))
print('\n max a and b is equal to : %f ' % max(a, b))

aa = []
bb = [1,]

squares = [1, 4, 9, 16, 25]
squares[0]
squares.append(36)
last = squares.pop()
l = len(squares)
squares[0] = 0
squares[2:3] = [33, 44]
squares[-1] = 49

# integers from 3 to 9
for kk in range(3, 10):
    print(kk)

# loop over list entries
for val in squares:
    print(val)

# loop with index and list entries
for ii, val in enumerate(squares):
    print('square [%d] ' % ii + '= %f' % val)
    print('square [%d] = %f' % (ii, val))

# loop over multiple lists at once
aa = [1, 2, 3]
bb = [4, 5, 6]
cc = [7, 6, 5]
for a, b, c in zip(aa, bb, cc):
    print(a, b, c)

#fibonacci sequence
fibo_sequence(5)

print(add(2, 3))  # 5
print(add(2, 3, True))  # 25
print(add(2, 3, squared=True))  # 25
b = 33
print(add_hidden(2))  # 35 instead of an error


print(add_sub(4, 10))  # prints the list [14,-6]
res = add_sub(9.2, 34.1)
print(res[0])  # prints the float 43.3
plus, minus = add_sub(10, 5)
print(plus, minus)  # prints two integers 15 5


arr1 = np.empty((3,4))
arr2 = np.zeros((3, 4))

arr3 = np.array([1, 2, 3, 4], dtype=np.float)
arr4 = np.arange(4)

arr5 = arr4 + 1j * arr3
mag = np.abs(arr5)
phase = np.angle(arr5)

arr6 = np.linspace(1, 20, 12)
arr7 = np.power([1, 2, 3], 2)

arr8 = np.random.randn(2, 3)


def modify(arr):
    arr[0] = 100


def modify_keep(arr):
    res = arr.copy()
    res[0] = 100
    return res


test_arr = np.arange(3)
res = modify_keep(test_arr)
print('original array by copying feature:', test_arr)
print('modified array by copying feature:', res)

modify(test_arr)
print('original array without copying feature:', test_arr)

# %% plotting
np.random.seed(1)
ADC_data = np.random.rand(10, 4)
for cnt in range(ADC_data.shape[1]):
    plt.plot(ADC_data[:, cnt], label='ADC_channel[%d]' % (cnt,))

plt.xlabel('sample index (number)')
plt.ylabel('values (v)')
plt.legend()
plt.grid(True)
tikz_save('./figs/test_matplotlib.tikz')
#plt.savefig('test_matplotlib.png', dpi=150)

R = np.logspace(1, 3)
for plt_fct, name in zip((plt.plot, plt.loglog), ('lin', 'log')):
    plt.figure()
    plt_fct(R, 1/np.power(R, 4))
    plt.xlabel('Range (m)')
    plt.ylabel('1/R^4 (m^-4)')
    plt.title(name)
    plt.grid(True)

plt.figure()
plt.semilogy(R, 1/np.power(R, 4))

# %% MaxNLocator
plt.figure()
ax=plt.gca()
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot(np.arange(4), 3.0*np.sin(np.arange(4)*1.2))



