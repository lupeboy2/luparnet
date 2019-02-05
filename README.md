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

### Training your first network.

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

## Methods
```
ln.net(layer,ds)
```
Constructs a neural network.
*layer* is equal to the amount of layers in the network
*ds* is an array and initializes the dimensions of the network.

```
yourNet.train(run,indata,outdata,error=False)
```
Trains the network for *run* times.
*indata* and *outdata* are the input data and output data arrays of type np.array()
*error* by default is False but takes an integer argument that prints the average 
loss, time elapsed and epoch of the training data at steps that are multiples of
*error*

```
yourNet.predict(data)
```
Makes a prediction for *data* of type np.array(). Data must be an np.array().

```
np.fit(csv)
```
Method takes a string, *csv* as input, must be a file name of a csv. The format
should be a single set of input data points seperated by commas on each line. 
Returns a np.array() that can be used as inputs for the other methods.

