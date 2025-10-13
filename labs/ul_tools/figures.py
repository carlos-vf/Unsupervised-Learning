import numpy as np


def swiss_roll(n_samples, noise_x=[0,0], noise_y=[0,0], noise_z=[0,0], random_state=None):
    """Generate a 3D swiss roll dataset.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    noise_x : list, optional
        Mean and standard deviation of Gaussian noise added to x coordinates. Default is [0,0].
    noise_y : list, optional
        Mean and standard deviation of Gaussian noise added to y coordinates. Default is [0,0].
    noise_z : list, optional
        Mean and standard deviation of Gaussian noise added to z coordinates. Default is [0,0].
    random_state : int, optional
        Random seed for reproducibility. Default is None.
    """

    np.random.seed(random_state)

    # Sampling
    phi = np.random.uniform(1.5*np.pi, 4.5*np.pi, n_samples)
    psi = np.random.uniform(0, 10, n_samples)

    # Noise
    if noise_x == [0,0] and noise_y == [0,0] and noise_z == [0,0]:
        noise_x = np.zeros(n_samples)
        noise_y = np.zeros(n_samples)
        noise_z = np.zeros(n_samples)
    else:
        noise_x = np.random.normal(noise_x[0], noise_x[1], n_samples)
        noise_y = np.random.normal(noise_y[0], noise_y[1], n_samples)
        noise_z = np.random.normal(noise_z[0], noise_z[1], n_samples)

    # Coordinates
    x = phi*np.cos(phi) + noise_x
    y = phi*np.sin(phi) + noise_y
    z = psi + noise_z

    return np.vstack((x, y, z)).T, phi




def klein_bottle(n_samples):
    """Generate a 3D Klein bottle dataset.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    """

    # Sampling
    u =  np.random.uniform(0, 2*np.pi, n_samples)
    v =  np.random.uniform(0, 2*np.pi, n_samples)

    cos_u = np.cos(u)
    sin_u = np.sin(u)
    cos_v = np.cos(v)
    sin_v = np.sin(v)
    
    # Coordinates
    x = - (2/15) * cos_u * (3 * cos_v - 30 * sin_u + 90 * cos_u**4 * sin_u - 60 * cos_u**6 * sin_u + 5 * cos_u * cos_v * sin_u)
    y = - (1/15) * sin_u * (3 * cos_v - 3 * cos_u**2 * cos_v - 48 * cos_u**4 * cos_v + 48 * cos_u**6 * cos_v - 60 * sin_u + 5 * cos_u * cos_v * sin_u - 5 * cos_u**3 * cos_v * sin_u - 80 * cos_u**5 * cos_v * sin_u + 80 * cos_u**7 * cos_v * sin_u)
    z = (2/15) * (3 + 5 * cos_u * sin_u) * sin_v

    return np.vstack((x, y, z)).T
