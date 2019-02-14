#Neural Network module
import numpy as np
import time
np.set_printoptions(suppress=True)
np.warnings.filterwarnings('ignore')
#
class net():
    def __init__(self,layer,ds,function="sigmoid"):
        self.layers = layer
        self.d=ds
        self.func=function
        syn=[]
        c=0
        a=0
        b=1
        while c<self.layers-1:
            #synapses
            syn.append(2*np.random.random((self.d[a],self.d[b])) - 1)
            temp=a
            a=b
            b=temp
            c+=1    

        syn.append(2*np.random.random((self.d[1],self.d[2])) - 1)
        self.syn=syn

    #define activate
    def activate(self,toSig,deriv=False):
        if(deriv):
            if(self.func=="sigmoid"):
                return (toSig*(1-toSig))
            if(self.func=="arctan"):
                return 1/(1+x**2)
            if(self.func=="softplus"):
                return 1/(1+np.exp(-toSig))
        if(self.func=="sigmoid"):
            return 1/(1+np.exp(-toSig))
        if(self.func=="arctan"):
            return np.arctan(toSig)
        if(self.func == "softplus"):
            return  np.log(1+np.exp(toSig)) 

    #define predict
    def predict(self,data):
        c=0
        ph2=data
        while(c<self.layers):
            ph=ph2
            ph2=self.activate(np.dot(ph, self.syn[c]))
            c+=1
        return ph2

    #train
    def train(self,run,indata,outdata,error=False):
        err = error
        if type(error) is int:
            err=True
        for j in xrange(run+1):
            #setup
            t0 = time.clock()
            syn=self.syn

            # predictulate forward through the network.
            l=[indata]
            l_error=[]
            l_delta=[]
            c=0
            while c<self.layers:
                l.append(self.activate(np.dot(l[c], syn[c])))
                l_error.append(0)
                l_delta.append(0)
                c+=1
            l_error.append(0)
            l_delta.append(0)
            cmax=c
            
            # Back propagation of errors using the chain rule.

            l_error[c]=outdata-l[c]
            if err and (j % error==0):   # Only print the error every 5000 steps
                print "Epoch:%d, Loss: %.8f, Time: %.5fs" % (j,(np.mean(np.abs(l_error[c]))), time.clock()-t0)
            l_delta[c]=l_error[c]*self.activate(l[c], deriv=True)
            c-=1
            
            while c>0:
                l_error[c]=l_delta[c+1].dot(syn[c].T)
                l_delta[c]=l_error[c]*self.activate(l[c], deriv=True)
                c-=1
                
            #update weights
            c=cmax-1
            while c>=0:
                syn[c] += l[c].T.dot(l_delta[c+1])
                c-=1

        return l_error[cmax]

    '''
    timer = time.clock()
    err = train(40000,x,y)
    timenow =  time.clock() - timer
    print "Final Time: %.5fs Final Loss: %.8f" %(timenow, (np.mean(np.abs(err))))
    data = np.array([[1]])
    refinedData=','.join(map(str, data))
    print "Prediction for data set: %s"%(refinedData)

    print(predict(data))
    '''
    #    
