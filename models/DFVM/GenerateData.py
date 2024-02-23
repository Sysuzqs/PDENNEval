from scipy.stats import qmc
import numpy as np

# cube
def sampleCube(dim, l_bounds, u_bounds, N=100):
    '''Uniform Mesh

    Get the Uniform Mesh.

    Args:
        dim:      The dimension of space
        l_bounds: The lower boundary
        u_bounds: The upper boundary
        N:        The number of sample points

    Returns:
        numpy.array: An array of sample points
    '''
    sample = []
    for i in range(dim):
        sample.append( np.linspace(l_bounds[i], u_bounds[i], N) ) 
    if dim == 2:
        x, y = np.meshgrid(sample[0], sample[1])
        return np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
    if dim == 3:
        x, y, z = np.meshgrid(sample[0], sample[1], sample[2])
        return np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))
    return sample[0].reshape(-1, 1)

def sampleCubeMC(dim, l_bounds, u_bounds, N=100):
    '''Monte Carlo Sampling

    Get the sampling points by Monte Carlo Method.

    Args:
        dim:      The dimension of space
        l_bounds: The lower boundary
        u_bounds: The upper boundary
        N:        The number of sample points

    Returns:
        numpy.array: An array of sample points
    '''
    sample = []
    for i in range(dim):
        sample.append( np.random.uniform(l_bounds[i], u_bounds[i], [N, 1]) ) 
    data = np.concatenate(sample, axis=1)
    return data

def sampleCubeQMC(dim, l_bounds, u_bounds, expon=100):
    '''Quasi-Monte Carlo Sampling

    Get the sampling points by quasi-Monte Carlo Sobol sequences in dim-dimensional space. 

    Args:
        dim:      The dimension of space
        l_bounds: The lower boundary
        u_bounds: The upper boundary
        expon:    The number of sample points will be 2^expon

    Returns:
        numpy.array: An array of sample points
    '''
    sampler = qmc.Sobol(d=dim, scramble=False)
    sample = sampler.random_base2(expon)
    data = qmc.scale(sample, l_bounds, u_bounds)
    data = np.array(data)
    return data[1:]


# unit sphere
def sampleBall(d, n):
    '''Sampling inside the unit sphere

    Args:
        d: dimension
        n: sampling numbers

    Returns:
        numpy.array: An array of sample points
    '''
    # x = []
    # while(len(x) < n):
    #     t = np.random.normal(-1, 1, [1, d])
    #     if np.sum(t**2) < 1:
    #         x.append( t )
    # x = np.concatenate(x, axis=0)
    # return x
    sample = sampleBallBD(d, n)
    r      = np.random.uniform(0, 1, n)
    for i in range(n):
        sample[i] *= r[i]
    return sample

def sampleBallBD(d, n):
    '''Sampling on the surface of the unit sphere

    Args:
        d: dimension
        n: sampling numbers

    Returns:
        numpy.array: An array of sample points
    '''
    x = np.random.normal(0, 1, [n,d])
    x = x/np.sqrt(  np.sum( x**2 , 1 ) ).reshape(n, 1)
    return x


if __name__ == '__main__':
    l_bounds = [0, 0, 0]
    u_bounds = [1, 1, 1]
    x = sampleCubeQMC(3, l_bounds, u_bounds, 4)
    print(x)
