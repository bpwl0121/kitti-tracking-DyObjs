def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def test_addition():
    assert add(2, 2) == 4

def test_subtraction():
    assert subtract(5, 3) == 2

def test_multiplication():
    assert multiply(4, 3) == 12
