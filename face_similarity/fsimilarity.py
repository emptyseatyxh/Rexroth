# -*- coding: utf-8 -*-

from scipy.optimize import fmin_cg
from matplotlib import pyplot as plt
import numpy as np
import os
import cv2
import face_recognition

# (255, 255, 255): BGR
# 0: white, 255: black

# debugging image show
def ds(name):
    plt.imshow(name, "gray")
    plt.show()
      
      
class NeuralNetwork(object):
    def __init__(self, X, y, reg_lambda=0.5, hidden_layer_size=[3,3]):
        self.X_m = X.shape[0]        
        self.X_f = X.shape[1]
        self.y   = y
        self.y_f = y.shape[1]
        self.reg = reg_lambda
        self.layer_l = len(hidden_layer_size) + 2
        self.layer_s = [self.X_f] + hidden_layer_size + [self.y_f]
        self.Theta   = []
        self.inputs  = []
        self.outputs = []
        for l in range(len(self.layer_s)):
            self.inputs.append( np.zeros( shape=(self.X_m, self.layer_s[l]) ) )
            self.outputs.append( np.zeros( shape=(self.X_m, self.layer_s[l]) ) )
        self.inputs[0]  = self.__normalization(X)
        self.outputs[0] = self.inputs[0]
        
        for lsize1, lsize2 in zip(self.layer_s, self.layer_s[1:]):
            self.Theta.append( np.random.normal( size=(lsize1+1, lsize2) ) )   
            
    def __normalization(self, X):
        self.mu  = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return ( X - self.mu ) / self.std
        
    def sigmoid(self, X):
        return 1.0 / ( 1.0 + np.exp(-X) )
    
    def sigmoid_prime(self, X):
        exp_X = np.exp(X)
        return exp_X / ( 1 + exp_X )**2
        
    def costfunction(self, theta_r):
        Theta_lst = []
        total_ele = 0
        for i, (lsize1, lsize2) in enumerate(zip(self.layer_s, self.layer_s[1:])):
            temp_the = theta_r[ total_ele : total_ele + (lsize1+1)*lsize2 ]
            Theta_lst.append( np.reshape( temp_the, self.Theta[i].shape ) )            
            total_ele += (lsize1+1)*lsize2
            
        for l in range(1, len(self.layer_s)):
            bias_output = np.hstack(( np.ones((self.outputs[l-1].shape[0], 1)), self.outputs[l-1] ))
            self.inputs[l]  = np.dot(bias_output, Theta_lst[l-1])
            self.outputs[l] = self.sigmoid(self.inputs[l])
        h = self.outputs[-1]
        J_reg = np.sum([(the[1:,:]**2).sum() for the in Theta_lst])
        J = 1./self.X_m * np.sum( -self.y*np.log(h) - (1-self.y)*np.log(1-h) ) + \
            self.reg/2/self.X_m * J_reg
        return J
    
    def backpropagation(self, theta_r):
        Theta_lst = []
        Theta_prime_lst = []
        total_ele = 0
        for i, (lsize1, lsize2) in enumerate(zip(self.layer_s, self.layer_s[1:])):
            temp_the = theta_r[ total_ele : total_ele + (lsize1+1)*lsize2 ]
            Theta_lst.append( np.reshape( temp_the, self.Theta[i].shape ) )
            Theta_prime_lst.append( 0 )
            total_ele += (lsize1+1)*lsize2
        
        delta_lst = []
        Delta_lst = []
        for l in range(len(self.layer_s)-1, 0, -1):
            bias_output = np.hstack(( np.ones((self.outputs[l-1].shape[0], 1)), self.outputs[l-1] ))
            if l == len(self.layer_s) - 1:
                delta_l = self.outputs[l] - self.y
                Delta_l = np.dot( bias_output.T, delta_l )
                delta_lst.append(delta_l)
                Delta_lst.append(Delta_l)
            else:
                delta_l = np.dot(delta_lst[-1], Theta_lst[l].T)[:,1:] * self.sigmoid_prime(self.inputs[l])
                Delta_l = np.dot( bias_output.T, delta_l )
                delta_lst.append(delta_l)
                Delta_lst.append(Delta_l)                
            Theta_prime = 1.0/self.X_m*Delta_l + self.reg/self.X_m*Theta_lst[l-1]
            Theta_prime[0,:] = 1.0/self.X_m*Delta_l[0,:]
            Theta_prime_lst[l-1] = Theta_prime
        theta_prime_ravel = np.concatenate(list(thep.ravel() for thep in Theta_prime_lst))
        return theta_prime_ravel
    
    
    def check_gradient(self):
        theta_ravel = np.concatenate(list(the.ravel() for the in self.Theta))
        numgrad = self.__cal_numerical_gradient(self.costfunction, theta_ravel)
        nnwgrad = self.backpropagation(theta_ravel)
        diff_ab = np.linalg.norm(numgrad-nnwgrad, 2) / np.linalg.norm(numgrad+nnwgrad, 2)
        print("difference in gradient computing: %.2e"%diff_ab)
        
    def __cal_numerical_gradient(self, costf, theta_r, eps=1e-5):
        perturb = np.zeros(theta_r.shape)
        numgrad = np.zeros(theta_r.shape)
        for i in range(theta_r.size):
            perturb[i] = eps
            J_d1 = costf(theta_r - perturb)
            J_d2 = costf(theta_r + perturb)
            numgrad[i] = (J_d2 - J_d1) / (2*eps)
            perturb[i] = 0
        return numgrad
        
    def train(self, niter, method="gradient decent", plotJ=True):
        theta_ravel = np.concatenate(list(the.ravel() for the in self.Theta))
        if method == "gradient decent":
            # update theta using gradient decent
            J_alst = np.empty((0))
            for i in range(niter):
                J = self.costfunction(theta_ravel)
                theta_grad = self.backpropagation(theta_ravel)
                theta_ravel -= theta_grad
                J_alst = np.append(J_alst, J)
            print("cost at the last training: %0.4f"%J)
            if plotJ == True:
                fig, ax = plt.subplots()
                ax.plot(np.arange(niter), J_alst)
                ax.set_xlabel("iteration")
                ax.set_ylabel("cost J")
        else:
            # update theta using conjugate gradient algorithm
            theta_ravel = fmin_cg(self.costfunction, theta_ravel, fprime=self.backpropagation)
        total_ele = 0
        for i, (lsize1, lsize2) in enumerate(zip(self.layer_s, self.layer_s[1:])):
            temp_the = theta_ravel[ total_ele : total_ele + (lsize1+1)*lsize2 ]
            self.Theta[i] = np.reshape( temp_the, self.Theta[i].shape )            
            total_ele += (lsize1+1)*lsize2
    
    def predict(self, X):
        self.inputs[0]  = ( X - self.mu ) / self.std
        self.outputs[0] = self.inputs[0]        
        for l in range(1, len(self.layer_s)):
            bias_output = np.hstack(( np.ones((self.outputs[l-1].shape[0], 1)), self.outputs[l-1] ))
            self.inputs[l]  = np.dot(bias_output, self.Theta[l-1])
            self.outputs[l] = self.sigmoid(self.inputs[l])
        h = self.outputs[-1]
        return h
        

# collect face features from webcam
def collect_faces_webcam(number, file_name="my_features.txt"):
    vc = cv2.VideoCapture(0)
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    fnumb = 0
    features = np.empty((0,128))
    while rval:
        rval, frame = vc.read()
        fnumb += rval
        facep128 = face_recognition.face_encodings(frame)
        if len(facep128) == 1:
            features = np.vstack(( features, facep128 ))
        if fnumb > number:
            break
        #time.sleep(0.5)
    vc.release()
    with open(file_name, "ab") as fappend:
        np.savetxt(fappend, features, fmt="%.9f", delimiter=",")

# collect face features from local files
def collect_faces_files(folder="oimages"):
    fpath = folder+r"/"
    files = [i for i in os.listdir(fpath) if "jpg" in i or "JPG" in i or "png" in i or "PNG" in i]
    features = np.empty((0,128))
    for image in files:
        readf    = face_recognition.load_image_file(fpath+image)
        facep128 = face_recognition.face_encodings(readf)
        for feature in facep128:
            features = np.vstack(( features, feature ))
    if features.shape[0] > 0:
        with open("other_faces.txt", "ab") as fappend:
            np.savetxt(fappend, features, fmt="%.9f", delimiter=",")


def static_face_recog(nnf, image_name="test1.jpg"):
    image = cv2.imread(image_name)
    face_locations = face_recognition.face_locations(image)
    face_features  = face_recognition.face_encodings(image)
    for facel, facef in zip(face_locations, face_features):
        y1, x1, y2, x2 = facel
        h = y2 - y1
        w = x2 - x1
        similar = nnf.predict(facef.reshape(1,-1))
        cv2.rectangle(image, (x1,y1),(x1+w,y1+h),(0,0,255),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image,"%0.2f"%similar[0],(x2,y2), font, 2,(0,0,255),3,cv2.LINE_AA)            
    cv2.imwrite("res_" + image_name, image)

def train_out():
    my_features = np.genfromtxt("my_features.txt", delimiter=",")
    my_comparef = np.genfromtxt("other_faces.txt", delimiter=",")
    X = np.vstack(( my_features, my_comparef))
    y = np.vstack(( np.ones((my_features.shape[0],1)), np.zeros((my_comparef.shape[0],1)) ))
    nnf = NeuralNetwork(X, y, 1, [50,50,50])
    nnf.train(1000, 2)
    return nnf

nnf = train_out()
static_face_recog(nnf, "ptest9.png")
