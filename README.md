# Luparnet
  *Luparnet* is a bare bones machine learning module that creates a simple class "net" 
This new object is a neural network that can be trained on data. The purpose of this 
system is to avoid having to use more complicated high level and complicated libraries
and to serve as an education tool.

## Installation
To install luparnet one simply needs to possess the Python Package Installer
and enter the following into the shell:
```
pip install luparnet
```

## Training your first network.

```python
import luparnet as ln
import numpy as np

#input data
x = np.array([[0],[1]])

#output data
y = np.array([[1],[0]])

#build the network
myFirstNet=ln.net(2,[1,2,1])

#train the network and print the final loss
print("Final Loss: %.8f"%(np.mean(np.abs(myFirstNet.train(30000,x,y,error=5000)))))

#predict for the value one
print(myFirstNet.predict(np.array([[1]]),string=True))
```
This code is all you need to run a net on your computer, granted that you have 
both numpy and Luparnet installed.
## Methods
```python
ln.net(layer,ds,fuction="sigmoid")
```
Constructs a neural network.
*layer* is equal to the amount of layers in the network
*ds* is an array and initializes the dimensions of the network.
*function* is the activation function with the default function being the sigmoid
function see the [Activation Functions](#activation-functions) section for more
information and for a list of the functions.

```python
yourNet.train(run,indata,outdata,error=False)
```
Trains the network for *run* times.
*indata* and *outdata* are the input data and output data arrays of type np.array()
*error* by default is False but takes an integer argument that prints the average 
loss, time elapsed and epoch of the training data at steps that are multiples of
*error*

```python
yourNet.predict(data)
```
Makes a prediction for *data* of type np.array(). Data must be an np.array().

```python
ln.fit(csv)
```
Method takes a string, *csv* as input, must be a file name of a csv. The format
should be a single set of input data points seperated by commas on each line. 
Returns an np.array() that can be used as inputs for the other methods.




## Data Dimensions
It is **extremely important** to find the right dimensions of your array and 
understand exactly what that means to the network.

Take the example in [Training Your Network](#training-your-first-network)

```python
#input data
x = np.array([[0],[1]])

#output data
y = np.array([[1],[0]])

#build the network
myFirstNet=ln.net(2,[1,2,1])
```
In this example the input data has dimensions of 1,2 which then become the
first two numbers of the *ds* array. The last digit of the array comes from 
the number of columns of the output array. 

For a final example we can take the final example:
```python
#input data
x = np.array([[1,0,1,1],[1,0,0,0],[0,0,0,0],[0,1,0,1]])

#output data
y = np.array([[1,1],[1,0],[1,0],[1,1]])
```
The output dimensions are therefore 
```[4,4,2]```

## Activation Functions
The activation function defines the output of a given neuron or node within
the network. 

| Function      | String        | Range |
|:-------------:|:-------------:|:-----:|
| Sigmoid       | "sigmoid"     |  (0,1) |
| Arc Tangent   | "arctan"      |  (−π/2,π/2) |
| SoftPlus      | "softplus"    |  (0,∞) |




