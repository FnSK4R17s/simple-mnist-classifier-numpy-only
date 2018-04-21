import numpy as np
import random
import pandas as pd

class Set():
    x = []
    y = []
    n = 0
    def ____init__(self):
        self.n=0

class Layer():
    weights = []
    adjustment = []

    def __init__(self):
        #self.input = input
        #self.output = output
        a=0

    def create(self,input,output):
        random.seed(1)
        self.weights = 2*np.random.random((input,output)) - 1
        self.adjustment = np.zeros((input,output), dtype=np.float32)

    def activation(self,x):
        return 1/(1 + np.exp(-x))

    def derivative(self,x):
        return x*( 1 - x )

    def forward(self,input_l):
        return self.activation(np.dot(input_l,self.weights))

    def back(self,input_l,delt):
        self.adjustment = np.dot(input_l.T,delta)
        return self.weights

    def update(self,lam):
        self.weights-=self.adjustment*lam

    def load_wt(self,wt):
        self.weights = wt


class Trainer():
    def __init__(self):
        a=0

    #___load_data
    def load_data(self,data_file,mem):
        initial = 0
        print("Loading Data")
        print("-"*60)

        data = pd.read_csv(data_file, header = None,)
        data.astype(np.float32)
        # assign the first and second columns to the variables X and y, respectively
        mem.x = np.c_[np.ones(data.shape[0]), data.iloc[:,1:785] ]
        labels = np.c_[data.iloc[:,0]]
        mem.n =  data.shape[0]
        mem.y = np.zeros((mem.n,10), dtype=np.float32)
        for v in range(0,mem.x.shape[0]-1):
            mem.y[v][labels[v]] = 1
        print("_"*60)


    #__show_data shows random data
    def show_data(self,mem):
        print("For example:")
        print("_"*60)
        location = random.randint(0,mem.n)
        j=0
        for j in range(1,mem.x.shape[1]):
            if (mem.x[location][j])==0:
                print(" ",end="")
            else:
                print('*',end="")
            if (j+1)%28==0:
                print()

        #print(mem.x.shape)
        #print()


    def softmax(self,z):
        t = np.exp(z - np.max(z))
        u = np.sum(t, axis=1)
        return t/u.reshape((t.shape[0],1))


    def cross_entropy_loss(self,pred,act):
        lg = -np.log((pred), dtype=np.float32)
        loss = np.zeros((pred.shape), dtype=np.float32)
        for w in range(0,pred.shape[0]-1):
            for j in range(0,9):
                loss[w][j] = lg[w][j]*act[w][j]
        #loss = np.inner(lg,act)
        return loss


    def cross_entropy_derivative(self,pred,act):
        return pred-act




if __name__ == "__main__":
    #input("Start??")
    #learning rate = 0.01
    L =0.0001
    trainer =  Trainer()
    train = Set()
    test = Set()
    trainer.load_data('mnist_train.csv',train)
    trainer.load_data('mnist_test.csv',test)
    print("-"*60)
    print("train.n",train.n)
    trainer.show_data(test)
    print("-"*60)
    print("test.n",test.n)
    trainer.show_data(train)
    #input("Continue???")
    l1 = Layer()
    l2 = Layer()
    l3 = Layer()
    l1.create(785,400)
    l2.create(400,200)
    l3.create(200,10)
    w1 = np.zeros((785,400), dtype=np.float32)
    w1 = np.zeros((400,200), dtype=np.float32)
    w1 = np.zeros((200,10), dtype=np.float32)
    print("training")
    learn = []

    #np.insert(m1, 0, 1, axis=1)
    for h in range(10000):
        j_train = 0
        m1 = train.x
        m2 = np.zeros((60000,400), dtype=np.float32)
        m3 = np.zeros((60000,200), dtype=np.float32)
        s = np.zeros((60000,100), dtype=np.float32)
        y1 = np.zeros((60000,10), dtype=np.float32)
        m2 = l1.forward(m1)
        m3 = l2.forward(m2)
        s = l3.forward(m3)
        y1 = trainer.softmax(s)
        j_train = np.sum(trainer.cross_entropy_loss(y1,train.y))
        j_train/=train.n
        delta = trainer.cross_entropy_derivative(y1,train.y)
        #backpropagation starts
        delta = delta*l3.derivative(s)
        w3 = l3.back(m3,delta)
        delta = np.dot(delta,w3.T)*l2.derivative(m3)
        w2 = l2.back(m2,delta)
        delta = np.dot(delta,w2.T)*l1.derivative(m2)
        w1 = l1.back(m1,delta)
        l1.update(L)
        l2.update(L)
        l3.update(L)
        print(j_train)
        print(h)
        learn.append(j_train)
    np.save("experiment_1.npz",learn)
    np.save("w1.npz",w1)
    np.save("w2.npz",w2)
    np.save("w3.npz",w3)


    m1 = test.x
    m2 = np.zeros((10000,400), dtype=np.float32)
    m3 = np.zeros((10000,200), dtype=np.float32)
    s = np.zeros((10000,100), dtype=np.float32)
    y1 = np.zeros((10000,10), dtype=np.float32)
    m2 = l1.forward(m1)
    m3 = l2.forward(m2)
    s = l3.forward(m3)
    y1 = trainer.softmax(s)
    pred = np.zeros((10000,1))
    pred = np.argmax(y1,axis=1)
    print(pred)
    print(test.y)
    correct = 0
    for i in range(10000-1):
        correct += test.y[i][pred[i]]
    print("accuracy=",correct/100,"%")
