import numpy as np

def main():
    x = [1,2,3,4]
    w = np.random.rand(4) 
    b = np.random.uniform(-1, 1)
    output, y = perceptron(w,x,b)
    print('output: ', output)
    print('y: ', y)

def perceptron(w, x, b):
    output = np.dot(x,w) + b
    y = 1 if output > 0 else 0
    return output, y

if __name__ == "__main__":
    main()