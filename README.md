# Calculation methods

This methods are being used to find function extrema:

*  Dichotomy;
*  Chord;
*  Coordinate descent;
*  Conjugate gradient;
*  Marquardt;
*  Swarm;

## Dichotomy method:
```python
	dichotomy(f, df, a, b, eps)
```
**where:**
*  **f** is a function
*  **df** is a differential of this function
*  **a** left boundary for x
*  **b** right boundary for x
*  **eps** accuracy
### Usage:
```python
def f(x):
    return 5*x**2 - 4*x + 1

def df(x):
    return 10*x - 4
    
dichotomy(f, df, 0, 10, 0.1)
```

## Chord method:
```python
	chord(f, df, a, b, eps)
```
**where:**
*  **f** is a function
*  **df** is a differential of this function
*  **a** left boundary for x
*  **b** right boundary for x
*  **eps** accuracy
### Usage:
```python
def f(x):
    return x**3 - 3*sin(x)

def df(x):
    return 3*(x**2 - cos(x))
    
chord(f, df, 0, 1, 10**-4)
```

## Coordinate descent method:
```python
	coordinate(f, x0, a, b, eps)
```
**where:**
*  **f** is a function
*  **x0** starting point
*  **a** left boundary for x
*  **b** right boundary for x
*  **eps** accuracy
### Usage:
```python
def f(x):
    return 129*x[0]**2 - 256*x[0]*x[1] + 129*x[1]**2 - 51*x[0] - 149*x[1] - 27
    
coordinate(f, [4, 4], -10, 10, 10**-8)
```

## Conjugate gradient method:
```python
	gradient(f, g, a, b, x0, eps)
```
**where:**
*  **f** is a function
*  **g** is a gradient of this function
*  **a** left boundary for x
*  **b** right boundary for x
*  **eps** accuracy
### Usage:
```python
def f(x):
    return 3*x[0]**2 - 3*x[0]*x[1] + 4*x[1]**2 - 2*x[0] + x[1]

def g(x):
    return array([-3*x[1] + 6*x[0] - 2, 8*x[1] - 3*x[0] + 1])    
    
gradient(f, g, -5, 6, array([0., 1.]), 10**-6)
```

## Marquardt method:
```python
	marquardt(f, f_a, j, x, eps)
```
**where:**
*  **f** is a function
*  **f_a** is a gradient of this function
*  **j** jacobi matrix for this function
*  **x** starting point
*  **eps** accuracy
### Usage:
```python
def f(x):
    return 0.5 * (1 - x[0])**2 + 0.5 * (x[1] - x[0]**2)**2

def jacobi(x):
    return array([ [-1, 0], [-2*x[0], 1]])

def f_a(x):
       return array([1 - x[0] , x[1] - x[0] ** 2]).reshape((2,1))    
       
marquardt(f, f_a, jacobi, array([-2, -2]), 10**-23)
```

## Swarm method:
```python
	swarm(f, xmin, xmax, d)
```
**where:**
*  **f** is a function
*  **xmin** left boundary for x
*  **xmax** right boundary for x
*  **d** dimension
### Usage:
```python
def f(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2    
       
swarm(f, -10, 10, 2)
```
