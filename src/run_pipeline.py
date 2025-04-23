# replace here by the import of reactiv.py
from reactiv import *
from reactiv_gpu import *

import os
import cv2
import sys
import glob
from glob import glob
import numpy as np
import rasterio
import rasterio.features
import rasterio.warp
import gc
import tracemalloc
import matplotlib.pyplot as plt

from pathlib import Path

DATA_PATH = Path("/home/cloud-user/work/data/acousur")  # <-- change this
YEARS = ["2020", "2021", "2022", "2023"]
POLARIZATIONS = ["ImagesTif_VH", "ImagesTif_VV"]


def renormalize_matrix(M, p1, p2):
    """
    Renormalize the values of a matrix between 0 and 1, where 0 corresponds to the 1st percentile
    and 1 corresponds to the 99th percentile.

    Parameters:
    - M: Input matrix.

    Returns:
    - M_normalized: Renormalized matrix.
    """
    # Calcul des percentiles 1 et 99
    percentile_1 = np.percentile(M, p1)
    percentile_2 = np.percentile(M, p2)

    # Renormalisation entre 0 et 1
    M_normalized = (M - percentile_1) / (percentile_2 - percentile_1)

    # Assurer que les valeurs sont limitées entre 0 et 1 (au cas où)
    M_normalized = np.clip(M_normalized, 0, 1)

    return M_normalized


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


def Stokes2Intensities(year, S0, S1, S2, S3):
    """
    Convert Stokes parameters into positive intensities and saves them to disk.
    """
    sqrt3_inv = 1 / np.sqrt(3)
    year_path = DATA_PATH / f"{year}"

    I0 = S0 + sqrt3_inv * S1 + sqrt3_inv * S2 + sqrt3_inv * S3
    np.save(year_path / f"{year}_I0.npy", I0)
    del I0
    gc.collect()

    I1 = S0 + sqrt3_inv * S1 - sqrt3_inv * S2 - sqrt3_inv * S3
    np.save(year_path / f"{year}_I1.npy", I1)
    del I1
    gc.collect()

    I2 = S0 - sqrt3_inv * S1 - sqrt3_inv * S2 + sqrt3_inv * S3
    np.save(year_path / f"{year}_I2.npy", I2)
    del I2
    gc.collect()

    I3 = S0 - sqrt3_inv * S1 + sqrt3_inv * S2 - sqrt3_inv * S3
    np.save(year_path / f"{year}_I3.npy", I3)
    del I3
    gc.collect()

    return  # On ne retourne rien, car tout est déjà sur le disque


def main():
    for year in YEARS:
        print("---------")
        print(f"{year=}")
        print("---------")

        year_path = DATA_PATH / f"{year}"
        year_path.mkdir(parents=True, exist_ok=True)

        Exx = np.load(DATA_PATH / f"{year}_VH.npy")
        Eyy = np.load(DATA_PATH / f"{year}_VV.npy")
        Nt, nx, ny = np.shape(Exx)

        # CVHH, CVVV
        file_CVHH = year_path / "CVHH.npy"
        file_CVVV = year_path / "CVVV.npy"
        if not file_CVHH.exists() or not file_CVVV.exists():
            p0 = np.abs(Exx) ** 2
            p2 = np.abs(Eyy) ** 2
            CVHH = Stack2SingleCV(p0, timeaxis=0)
            CVVV = Stack2SingleCV(p2, timeaxis=0)
            np.save(file_CVHH, CVHH)
            np.save(file_CVVV, CVVV)
        else:
            p0 = np.abs(Exx) ** 2
            p2 = np.abs(Eyy) ** 2
            print("CVHH & CVVV already exist")

        # Lmin3, Lmax3
        file_Lmin3 = year_path / "Lmin3.npy"
        file_Lmax3 = year_path / "Lmax3.npy"
        if not file_Lmin3.exists() or not file_Lmax3.exists():
            print("Lmin3 and Lmax3")
            Lmin3, Lmax3 = CV_Generalized_Limits_GPU_Chunked([p0, p2])
            np.save(file_Lmin3, Lmin3)
            np.save(file_Lmax3, Lmax3)
        else:
            print("Lmin3 & Lmax3 already exist")

        # CVRR, CVgv, CVVN, CVAZ
        files_cv = {
            "CVRR": year_path / "CVRR.npy",
            "CVgv": year_path / "CVgv.npy",
            "CVVN": year_path / "CVVN.npy",
            "CVAZ": year_path / "CVAZ.npy",
        }
        if not all(f.exists() for f in files_cv.values()):
            print("CVRR, ...")
            CVRR, CVgv, CVVN, CVAZ = CV_fromListofImages([p0, p2])
            np.save(files_cv["CVRR"], CVRR)
            np.save(files_cv["CVgv"], CVgv)
            np.save(files_cv["CVVN"], CVVN)
            np.save(files_cv["CVAZ"], CVAZ)
        else:
            print("CVRR, CVgv, CVVN, CVAZ already exist")

        # CV1, CVne1, CVnemoins1, CV_ne0
        files_gen = {
            "CV1": year_path / "CV1.npy",
            "CVne1": year_path / "CVne1.npy",
            "CVnemoins1": year_path / "CVnemoins1.npy",
            "CV_ne0": year_path / "CV_ne0.npy",
        }
        if not all(f.exists() for f in files_gen.values()):
            print("CV1, ...")
            CV1 = CV_Generalized_equally_GPU([p0, p2], 1)
            CVne1 = CV_Generalized_Non_equally_GPU([p0, p2], 1)
            CVnemoins1 = CV_Generalized_Non_equally_GPU([p0, p2], -1)
            CV_ne0 = CV_Generalized_Non_equally_Zero([p0, p2])
            np.save(files_gen["CV1"], CV1)
            np.save(files_gen["CVne1"], CVne1)
            np.save(files_gen["CVnemoins1"], CVnemoins1)
            np.save(files_gen["CV_ne0"], CV_ne0)
        else:
            print("CV1, CVne1, CVnemoins1, CV_ne0 already exist")

        # Libération mémoire avant étape critique
        del Exx, Eyy, p0, p2
        gc.collect()

        # Intensities
        I_files = {
            "I0": year_path / f"{year}_I0.npy",
            "I1": year_path / f"{year}_I1.npy",
            "I2": year_path / f"{year}_I2.npy",
            "I3": year_path / f"{year}_I3.npy",
        }
        if not all(f.exists() for f in I_files.values()):
            print("Intensities")
            Exx = np.load(DATA_PATH / f"{year}_VH.npy")
            Eyy = np.load(DATA_PATH / f"{year}_VV.npy")
            S0, S1, S2, S3 = Jones2Stokes(Exx, Eyy)
            Stokes2Intensities(year, S0, S1, S2, S3)
            del Exx, Eyy, S0, S1, S2, S3
            gc.collect()
        else:
            print("Intensity files already exist")
        # Lmin, Lmax
        file_Lmin = year_path / "Lmin.npy"
        file_Lmax = year_path / "Lmax.npy"
        if not file_Lmin.exists() or not file_Lmax.exists():
            # Chargement lazy avec mmap
            I0 = np.load(I_files["I0"], mmap_mode="r")
            I1 = np.load(I_files["I1"], mmap_mode="r")
            I2 = np.load(I_files["I2"], mmap_mode="r")
            I3 = np.load(I_files["I3"], mmap_mode="r")
            P = [I0, I1, I2, I3]
            print("Lmin, Lmax")

            # Profilage mémoire
            tracemalloc.start()
            Lmin, Lmax = CV_Generalized_Limits_GPU_Chunked(P)
            current, peak = tracemalloc.get_traced_memory()
            print(f"Memory: current={current / 1e6:.1f}MB, peak={peak / 1e6:.1f}MB")
            tracemalloc.stop()
            np.save(file_Lmin, Lmin)
            np.save(file_Lmax, Lmax)
            del I0, I1, I2, I3, P, Lmin, Lmax
            gc.collect()
        else:
            print("Lmin & Lmax already exist")

        Exx = np.load(DATA_PATH / f"{year}_VH.npy")
        Eyy = np.load(DATA_PATH / f"{year}_VV.npy")

        # Lmin2, Lmax2
        file_Lmin2 = year_path / "Lmin2.npy"
        file_Lmax2 = year_path / "Lmax2.npy"
        if not file_Lmin2.exists() or not file_Lmax2.exists():
            p0 = np.abs(Exx) ** 2
            p1 = np.abs(Eyy) ** 2
            print("Lmin2, Lmax2")
            Lmin2, Lmax2 = CV_Generalized_Limits_GPU_Chunked([p0, p1])
            np.save(file_Lmin2, Lmin2)
            np.save(file_Lmax2, Lmax2)
        else:
            print("Lmin2 & Lmax2 already exist")

        # Lmin4, Lmax4
        file_Lmin4 = year_path / "Lmin4.npy"
        file_Lmax4 = year_path / "Lmax4.npy"
        if not file_Lmin4.exists() or not file_Lmax4.exists():
            p0 = np.abs(Exx - Eyy) ** 2
            p2 = np.abs(Exx + Eyy) ** 2
            print("Lmin4, Lmax4")
            Lmin4, Lmax4 = CV_Generalized_Limits_GPU_Chunked([p0, p2])
            np.save(file_Lmin4, Lmin4)
            np.save(file_Lmax4, Lmax4)
        else:
            print("Lmin4 & Lmax4 already exist")


if __name__ == "__main__":
    main()
