# -*- coding: utf-8 -*-
"""
Radar Class

This is a temporary script file.
"""
# import matplotlib2tikz
# from matplotlib2tikz import
import numpy

fib = [1, 1]

for cnt in range(10):
    fib.append(fib[-1] + fib[-2])

print(fib)


def fibo_func(n):
    """
    fibonacci function generate the sequence of n numbers
    :param n:
    """
    fibo = [1,1]
    if n > 1:
        for cnt in range(2,n):
            fibo.append(fibo[-2] + fibo[-1])

    print('fibonacci sequence is equal to : ', fibo)