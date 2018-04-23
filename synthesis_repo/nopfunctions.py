import math


##binary##
def addition(vector):
    a = 0
    for i in vector:
        if a > 1e+6: return 1e+6
        elif a < -1e+6: return -1e+6
        elif a < 1e-6 and a > 0: return 1e-6
        elif a > -1e-6 and a < 0: return -1e-6
        else: a += i
    return a

def multiplication(vector):
    a = 1
    for i in vector:
        if a > 1e+6: return 1e+6
        elif a < -1e+6: return -1e+6
        elif a < 1e-6 and a > 0: return 1e-6
        elif a > -1e-6 and a < 0: return -1e-6
        else: a *= i
    return a

def maximum(vector):
    return max(vector)

def minimum(vector):
    return min(vector)

##unary##
def relu(a):
    if a < 0: return 0
    else:
        if a > 1e+6: return 1e+6
        else: return a

def identity(a):
    return a

def pow_two(a):
    if a > 1e+3 or a < -1e+3: return 1e+6
    elif a < 1e-3 and a > 0: return 1e-6
    elif a > -1e-3 and a < 0: return 1e-6
    else: return a * a
    
def negative(a):
    return -a

def irer(a):
    if a < 1e-6 and a > 0: return 1e-6 #return math.sqrt(a)
    elif a > -1e-6 and a < 0: return -1e-6 #return -math.sqrt(math.fabs(a))
    else:
        if a < 0: return -math.sqrt(math.fabs(a))
        elif a > 0: return math.sqrt(a)
        else: return 0
    
def reverse(a):
    if a > 1e+6: return 1e-6
    elif a < -1e+6: return -1e-6
    elif a < 1e-6 and a > 0: return 1e+6
    elif a > -1e-6 and a < 0: return -1e+6
    elif a == 0.0: return 1e+6
    else: return 1.0/a

def exp(a):
    if a > 14: return 1e+6
    elif a < -14: return 1e-6
    elif a < 1e-6 and a > 0: return 1
    elif a > -1e-6 and a < 0: return 1
    else: return math.exp(a)

def natlog(a):
    if a > 1e+6: return 1e+6
    elif a <= 0: return 0
    elif a < 1e-6 and a > 0: return -1e+6
    else: return math.log(a)

def logic(a):
    if a >= 0: return 1
    else: return 0
    
def cosinus(a):
    if a > 1e+6: return math.cos(1e+6)
    elif a < 1e-6 and a > 0: return math.cos(1e-6)
    elif a < -1e+6: return math.cos(-1e+6)
    elif a > -1e-6 and a < 0: return math.cos(-1e-6)
    return math.cos(a)

def sinus(a):
    if a > 1e+6: return math.sin(1e+6)
    elif a < 1e-6 and a > 0: return math.sin(1e-6)
    elif a < -1e+6: return math.sin(-1e+6)
    elif a > -1e-6 and a < 0: return math.sin(-1e-6)
    return math.sin(a)

def cubicroot(a):
    if a < 1e-6 and a > 0: return 1e-6
    if a > -1e-6 and a < 0: return -1e-6
    else: return a ** (1/3)

def atan(a):
    return math.atan(a)

def cubic(a):
    if a > 1e+2: return 1e+6
    elif a < -1e+2: return -1e+6
    elif a > -1e-2 and a < 0: return -1e-6
    elif a < 1e-2 and a > 0: return 1e-6
    else: return a*a*a

