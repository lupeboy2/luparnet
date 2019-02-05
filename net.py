#Neural Network module
import numpy as np
import time
np.set_printoptions(suppress=True)
np.warnings.filterwarnings('ignore')

class net():
    def __init__(self,layer, ds):

        self.layers = layer
        self.d=ds
        
        #seed
        np.random.seed(1)
        syn=[]
        c=0
        a=0
        b=1
        temp=a
        tempa=a
        tempb=b
        temp=a
        while c<self.layers-1:
            #synapses
            syn.append(2*np.random.random((self.d[tempa],self.d[tempb])) - 1)
            temp=a
            a=b
            b=temp
            c+=1    

        syn.append(2*np.random.random((self.d[1],self.d[2])) - 1)
        self.syn=syn

    #define sigmoid
    def sigmoid(self,toSig, deriv=False):
        if(deriv==True):
            return (toSig*(1-toSig))
        
        return 1/(1+np.exp(-toSig))

    #define predict
    def predict(self,a,string=False):
        c=0
        ph2=a
        while(c<self.layers):
            ph=ph2
            ph2=self.sigmoid(np.dot(ph, self.syn[c]))
            c+=1
        if(string):
            data = a
            dataaaa='\n'.join('\t'.join('%0.3f' %x for x in y) for y in data)

            returnMe=''
            returnMe+="Prediction for data set: %s"%(dataaaa)
            returnMe+='\n'+str(float(ph2))
            ph2=returnMe
        return ph2

    #train
    def train(self,run,a,b,error=True):
        for j in xrange(run+1):
            #setup
            t0 = time.clock()
            syn=self.syn
            # predictulate forward through the network.
            l=[a]
            l_error=[]
            l_delta=[]
            c=0
            while c<self.layers:
                l.append(self.sigmoid(np.dot(l[c], syn[c])))
                l_error.append(0)
                l_delta.append(0)
                c+=1
            l_error.append(0)
            l_delta.append(0)
            cmax=c
            
            # Back propagation of errors using the chain rule.

            l_error[c]=b-l[c]
            if error and (j % 5000==0):   # Only print the error every 5000 steps
                print "Epoch:%d, Loss: %.8f, Time: %.5fs" % (j,(np.mean(np.abs(l_error[c]))), time.clock()-t0)
            l_delta[c]=l_error[c]*self.sigmoid(l[c], deriv=True)
            c-=1
            
            while c>0:
                l_error[c]=l_delta[c+1].dot(syn[c].T)
                l_delta[c]=l_error[c]*self.sigmoid(l[c], deriv=True)
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


