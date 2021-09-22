'''
Created on Aug 4, 2017

@author: manu
'''

def protectedDiv(left, right):
    if abs(right) < 1e-4:
        return 1.0
    return left / right

    
def inverse(single):
    if abs(single) < 1e-4:
        return 1.0
    return 1.0/single
    
def mod(left,right):
    if abs(right) < 1e-4:
        return 1.0
    return left % right

def opposite(single):
    return -1.0*single

    
def squared(single):
    return single*single

def if_then_else(condition, out1, out2):
    return out1 if condition else out2


def equal(a,b):
    return abs(a-b) <= 1e-4

def neg(a):
    return not a
