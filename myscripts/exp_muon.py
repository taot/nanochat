def f(x):
    return 3.4445 * x - 4.7775 * x**3 + 2.0315 * x**5

iter = 5

# from 0 to 1 with step 0.1
xs = [i * 0.01 for i in range(10 + 1)]

for it in range(iter):
    print(f"iter {it}:")
    xs = [f(xi) for xi in xs]
    print(xs)
