def f(x, y):
    return x**2 + 2*y**2

def gradient(x, y):
    dfx = 2*x
    dfy = 4*y
    return [dfx, dfy]

m = 0.5  
alpha = 0.5
epsilon = 1e-6 
count = 0
x, y = 2, 1

while True:
    grad = gradient(x, y)
    t = 1.0
    
    while f(x - t * grad[0], y - t * grad[1]) > f(x, y) - m * t * (grad[0]**2 + grad[1]**2):
        t = t * alpha

    next_x = x - t * grad[0]
    next_y = y - t * grad[1]
    next_grad = gradient(next_x, next_y)

    if ( next_grad[0]**2 + next_grad[1]**2 ) < epsilon:
        x, y = next_x, next_y
        count += 1
        break
    else:
        x, y = next_x, next_y
        count += 1

print("Desired solution:" ,(x, y))
print("Value function with desired solution:", f(x, y))
print("Gradient with function:", gradient(x,y) )
print("Số lần lặp:", count)         # 2 lần lặp
