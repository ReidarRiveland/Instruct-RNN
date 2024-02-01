import numpy as np
import matplotlib.pyplot as plt    

def make_periodic_beta(max_beta):
    return max_beta/2*np.cos(np.linspace(-np.pi, np.pi, 11, endpoint=False))+max_beta/2

def softmax(x, beta=1):
    return np.exp(beta*x)/np.sum(np.exp(beta*x))

def circuit_recall_inhibit_act(x, threshold=0.7): 
    if x>=threshold: return 1
    else: return np.maximum(0,x)

def value_inhibit_act(x, shift=0.5, slope=8): 
    return -1*np.tanh((x-shift)*slope)

def surround_inhibit(post_activity, recover_act): 
    surround_inhibit = -1*np.full_like(post_activity, recover_act)
    surround_inhibit[post_activity.argmax()] = post_activity.max()
    return surround_inhibit

def cos_sim(x, y): 
    return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))
