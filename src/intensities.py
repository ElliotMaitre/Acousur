from pathlib import Path
import numpy as np

DATA_PATH = Path("/home/cloud-user/work/data/acousur")


def Jones2Stokes(Ex, Ey):
    """
    Convert Jones vectors to Stokes parameters.
    Args:
        Ex (numpy.ndarray): Electric field component in the x-direction over time and space.
                           Dimensions: (nt, nx, ny)
        Ey (numpy.ndarray): Electric field component in the y-direction over time and space.
                           Dimensions: (nt, nx, ny)
    Returns:
        tuple: A tuple containing four matrices representing the Stokes parameters:
            - S0 (numpy.ndarray): Total intensity. Dimensions: (nt, nx, ny)
            - S1 (numpy.ndarray): Stokes parameter S1. Dimensions: (nt, nx, ny)
            - S2 (numpy.ndarray): Stokes parameter S2. Dimensions: (nt, nx, ny)
            - S3 (numpy.ndarray): Stokes parameter S3. Dimensions: (nt, nx, ny)
    """
    # Calculate Stokes parameters
    S0 = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
    S1 = np.abs(Ex) ** 2 - np.abs(Ey) ** 2
    S2 = 2 * np.real(Ex * np.conj(Ey))
    S3 = 2 * np.imag(Ex * np.conj(Ey))

    return S0, S1, S2, S3


def Stokes2Intensities(S0, S1, S2, S3):
    """
    Convert Stokes parameters into positive intensities.
    Args:
        Si (numpy.ndarray): Stokes components   Dimensions: (nt, nx, ny)
    Returns:
        tuple: A tuple containing four matrices representing the Intensities parameters:
            - Ii (numpy.ndarray): Positive Intensity. Dimensions: (nt, nx, ny)

    """
    # Calculate Stokes parameters
    I0 = S0 + 1 / np.sqrt(3) * S1 + 1 / np.sqrt(3) * S2 + 1 / np.sqrt(3) * S3
    np.save(DATA_PATH / "2020_I0.npy", I0)
    I0 = 0
    I1 = S0 + 1 / np.sqrt(3) * S1 - 1 / np.sqrt(3) * S2 - 1 / np.sqrt(3) * S3
    np.save(DATA_PATH / "2020_I1.npy", I1)
    I1 = 0
    I2 = S0 - 1 / np.sqrt(3) * S1 - 1 / np.sqrt(3) * S2 + 1 / np.sqrt(3) * S3
    np.save(DATA_PATH / "2020_I2.npy", I2)
    I2 = 0
    I3 = S0 - 1 / np.sqrt(3) * S1 + 1 / np.sqrt(3) * S2 - 1 / np.sqrt(3) * S3
    np.save(DATA_PATH / "2020_I3.npy", I3)
    I3 = 0

    return I0, I1, I2, I3


Exx = np.load(DATA_PATH / "2020_VH.npy")
Eyy = np.load(DATA_PATH / "2020_VV.npy")

print("Jones")
S0, S1, S2, S3 = Jones2Stokes(Exx, Eyy)
print("Intensities")
I0, I1, I2, I3 = Stokes2Intensities(S0, S1, S2, S3)
