# Calculation methods

This methods are being used to find function extrema:

*  Dichotomy;
*  Chord;
*  Coordinate descent;
*  Conjugate gradient;
*  Marquardt;
*  Swarm;

# Dichotomy method:
```python
	dichotomy(f, df, a, b, eps)
```
	where:
	-f is a function
	-df is a differential of this function
	-a left boundary for x
	-b right boundary for x
	-eps accuracy

# Chord method:
```python
	chord(f, df, a, b, eps)
```
	where:
	-f is a function
	-df is a differential of this function
	-a left boundary for x
	-b right boundary for x
	-eps accuracy

# Coordinate descent method:
```python
	coordinate(f, x0, a, b, eps)
```
	where:
	-f is a function
	-x0 starting point
	-a left boundary for x
	-b right boundary for x
	-eps accuracy

# Conjugate gradient method:
```python
	gradient(f, g, a, b, x0, eps)
```
	where:
	-f is a function
	-g is a gradient of this function
	-a left boundary for x
	-b right boundary for x
	-eps accuracy

# Marquardt method:
```python
	marquardt(f, f_a, j, x, eps)
```
	where:
	-f is a function
	-f_a is a gradient of this function
	-j jacobi matrix for this function
	-x starting point
	-eps accuracy

# Swarm method:
```python
	swarm(f, xmin, xmax, d)
```
	where:
	-f is a function
	-xmin left boundary for x
	-xmax right boundary for x
	-d dimension
