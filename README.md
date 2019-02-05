# Luparnet
  *Luparnet* is a barebones machine learning module that creates a simple class "net" 
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

myFirstNet=ln.net(2,[1,2,1])
print("Final Loss: %.8f"%(np.mean(np.abs(myFirstNet.train(30000,x,y,error=5000)))))
print(myFirstNet.predict(np.array([[1]]),string=True))
```

## Methods
  The network takes only two arguments in the constructor, the first being
the desired number of layers and the second being an array of the data dimensions.
The train function takes three arguments with an option for overload. The function
takes first the number of epochs to train for, the second is the input data, the
third is the output data. The final argument is an integer which allows for debugging
and user training.






