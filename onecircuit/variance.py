import numpy as np
from scipy import integrate
from scipy.integrate import quad

class VarianceCalculator:
    def __init__(self, wavefunction):
        self.wavefunction = wavefunction

    def gauss_variance(self, fx, x_range, y, n_samples=10000):
        samples = np.random.uniform(x_range[0], x_range[1], n_samples)  
        values = np.array([fx(x, y) for x in samples]) 
        mean = np.mean(values)  
        variance = np.mean((values - mean) ** 2)  
        return variance, mean
        
    def single_variance(self, x_values, y):
        
        fx = lambda x: np.abs(self.wavefunction(x, y))
        norm_c = quad(fx, x_values[0], x_values[1])[0]
        pdf = lambda x: np.abs(fx(x)) / norm_c
        
        # Compute the mean
        xpdf = lambda x: x * pdf(x)
        mean = quad(xpdf, x_values[0], x_values[1])[0]
        
        # Compute the variance
        xmpdf = lambda x: ((x - mean)**2) * pdf(x)
        variance = quad(xmpdf, x_values[0], x_values[1])[0]
        
        return variance, mean

    def circuit_variance(self, circuit, ham, cost_func):
        results = cost_func(circuit, ham)

        return results
    
    def total_variance(circuit, gradient):

        gradients = []
        num_samples = 100  

        for i in range(num_samples):
            sampled_thetas = np.random.uniform(0, 2*np.pi, len(circuit[0][3]))
            circuit[0][3] = sampled_thetas

            gradient_data = gradient(circuit)
            gradients.append(gradient_data)

        mean_cost = np.mean(gradients)
   
        variance = np.var(gradients)

        return variance

    def first_variance(circuit, gradient):

        gradients = []
        num_samples = 100  

        for i in range(num_samples):
            sampled_thetas = np.random.uniform(0, 2*np.pi)
            circuit[0][3][0] = sampled_thetas

            gradient_data = gradient(circuit)
            gradients.append(gradient_data)

        mean_cost = np.mean(gradients)
   
        variance = np.var(gradients)

        return variance
