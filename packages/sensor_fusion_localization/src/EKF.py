import numpy as np
class EKF(object):
    def __init__(self, mean, init_var=np.zeros((3,3))):
        self.mean = mean
        self.var = init_var

    def update(self, z, Q, R, dt, predicted, v, omega):
        lastSigma = self.var
        lastMean = self.mean
        G = self.jacobian(v, omega, dt)
        currentSigma = G * lastSigma * np.transpose(G) + R
        
        num_measurements = (int)(np.shape(z)[0]/3)
        H = np.zeros((3*num_measurements,3))
        Qnp = np.kron(np.eye(num_measurements), Q)
        for i in range(num_measurements):
            H[3*i:3*i+3,:] = np.eye(3)
        y = z - np.matmul(H, predicted)
        S = np.matmul(np.matmul(H,currentSigma),np.transpose(H)) + Qnp
        if np.linalg.det(S) == 0:
            print(S)
            return
        K = np.matmul(np.matmul(currentSigma,np.transpose(H)),np.linalg.inv(S))
        self.mean = predicted + np.matmul(K,y)
        self.var = np.matmul(np.eye(3) - np.matmul(K,H),currentSigma)
    
    def jacobian(self, v, omega, dt):
        theta = self.mean[2]
        G = np.eye(3)
        if np.abs(omega) < 0.000001:
            G[0,2] = -v * dt * np.sin(theta)
        else:
            k = v / omega
            G[0,2] = -k * np.cos(theta) + k * np.cos(theta + omega * dt)
            G[1,2] = -k * np.sin(theta) + k * np.sin(theta + omega * dt)
        return G

   
