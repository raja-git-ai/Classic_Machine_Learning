import numpy as np


def sigmoid(x,deriv=False):
    x= 1+np.exp(-x)
    return 1/x

def derivative(x):
    return x*(1-x)

# We are going to create a single hidden layer neural net of 3 neurons 
# Input data
x=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y=np.array([[0],[1],[1],[0]])

np.random.seed(1)

#synapses
# 3 input nodes 
#syn0 = 2*np.random.random([3,4]) -1
syn0 = np.random.random([3,4])
# 4 nodes in hidden layer
#syn1 = 2*np.random.random([4,1]) -1
syn1 = np.random.random([4,1])

for i in range(6000):
    # Doing orward propagation
    ln = x
    l1 = sigmoid(np.dot(ln,syn0))
    l2 = sigmoid(np.dot(l1,syn1))

    # Finding the error in the last layer
    l2_err = y-l2
    if((i % 1000) ==0):
        print("Err:- ",np.mean(l2_err))
    # Taking the derivative for backpropagation
    d_l2=derivative(l2)
    #print("l2",l2)
    #print("derivative l2",d_l2)
    # Doing backpropagation and adjusting the weights in the parameture matrix 
    l2_delta = l2_err * d_l2
    #print("l2_err",l2_err)
    #print("sigmoid_l2",d_l2)
    #print("l2_delta",l2_delta)
    l1_err = l2_delta.dot(syn1.T)
    d_l1 = derivative(l1)
    #print("l1",l1)
    #print("derivative l1",d_l1)
    l1_delta = l1_err * d_l1
    #print("l1_err",l1_err)
    #print("\n sigmoid_l1",d_l1)
    #print("\n l1_delta",l1_delta)
    # Update the parameture
    syn1 += l1.T.dot(l2_delta)
    syn0 += ln.T.dot(l1_delta)

print("Output after training",l2)
    


           
