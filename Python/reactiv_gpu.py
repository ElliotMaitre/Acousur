import torch
import numpy as np
from tqdm import tqdm


def CV_Generalized_Limits_GPU(series_np, device="cuda"):
    """
    Vectorized PyTorch version of CV_Generalized_Limits using GPU acceleration.
    series_np: list of p numpy arrays of shape (Nt, nx, ny)
    """
    # Convert input to a single PyTorch tensor on GPU
    series = torch.tensor(
        np.stack(series_np), dtype=torch.float32, device=device
    )  # shape: (p, Nt, nx, ny)
    p, Nt, nx, ny = series.shape

    # Reshape to (p, Nt, N), where N = nx * ny
    N = nx * ny
    series = series.view(p, Nt, N)  # shape: (p, Nt, N)

    # Compute mean per series and pixel: (p, N)
    mu = series.mean(dim=1)

    # Compute norm of mean vector for each pixel: shape (N,)
    normMU = torch.norm(mu, dim=0)

    # Centered series: shape (p, Nt, N)
    centered = series - series.mean(dim=1, keepdim=True)

    # Compute covariance: (p, p, N)
    cov = torch.einsum("pti,qti->pqi", centered, centered) / Nt

    # Compute eigenvalues of each covariance matrix: shape (p, N)
    eigvals = torch.linalg.eigvalsh(cov)  # sorted ascending

    # Extract min and max eigenvalues
    eig_min = eigvals[0]
    eig_max = eigvals[-1]

    # Compute limits
    limit_min = torch.sqrt(eig_min) / (normMU + 1e-8)
    limit_max = torch.sqrt(eig_max) / (normMU + 1e-8)

    # Reshape back to (nx, ny)
    return limit_min.view(nx, ny).cpu().numpy(), limit_max.view(nx, ny).cpu().numpy()


def CV_Generalized_equally_GPU(S, Q, device="cuda"):
    """
    GPU version of Contrast definition based on the Generalized Equally Weighted Mean at order Q.
    Inputs:
    - S: list of p numpy arrays of shape (Nt, nx, ny)
    - Q: order of the matrix power
    Output:
    - CV: numpy array of shape (nx, ny)
    """
    # Stack and move data to GPU
    series = torch.tensor(
        np.stack(S), dtype=torch.float32, device=device
    )  # (p, Nt, nx, ny)
    p, Nt, nx, ny = series.shape
    N = nx * ny

    # Reshape to (p, Nt, N)
    series = series.view(p, Nt, N)

    # Compute mean over time: shape (p, N)
    mu = series.mean(dim=1)
    norm_mu = torch.norm(mu, dim=0) + 1e-8  # avoid division by zero

    # Centered time series: (p, Nt, N)
    centered = series - series.mean(dim=1, keepdim=True)

    # Compute covariance matrix per pixel: (p, p, N)
    cov = torch.einsum("pti,qti->pqi", centered, centered) / Nt  # (p, p, N)
    cov = cov.permute(2, 0, 1)  # (N, p, p)

    # Raise each covariance matrix to the Q-th power
    CQ = torch.linalg.matrix_power(cov, Q)  # (N, p, p)

    # Compute trace of CQ for each pixel (N,)
    trace_CQ = CQ.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)

    # Compute CV per pixel (N,)
    CV_flat = torch.sqrt((trace_CQ / p) ** (1 / Q)) / norm_mu

    # Reshape to (nx, ny)
    CV = CV_flat.view(nx, ny).cpu().numpy()

    return CV


def CV_Generalized_Non_equally_GPU(S, Q, device="cuda"):
    """
    GPU version of Generalized Non Equally Weighted Mean at order Q ≠ 0.

    Inputs:
    - S: list of p numpy arrays of shape (Nt, nx, ny)
    - Q: power order (integer)

    Returns:
    - CV: numpy array of shape (nx, ny)
    """
    # Stack and move to GPU
    series = torch.tensor(
        np.stack(S), dtype=torch.float32, device=device
    )  # (p, Nt, nx, ny)
    p, Nt, nx, ny = series.shape
    N = nx * ny

    # Reshape to (p, Nt, N)
    series = series.view(p, Nt, N)

    # Compute mean over time (mu): shape (p, N)
    mu = series.mean(dim=1)
    norm_mu = torch.norm(mu, dim=0) + 1e-8  # shape (N,)

    # Center time series: (p, Nt, N)
    centered = series - series.mean(dim=1, keepdim=True)

    # Covariance matrices: (N, p, p)
    cov = torch.einsum("pti,qti->pqi", centered, centered) / Nt  # (p, p, N)
    cov = cov.permute(2, 0, 1)  # (N, p, p)

    # Covariance matrix to power Q: (N, p, p)
    C_q = torch.linalg.matrix_power(cov, Q)

    # Reshape mu to (N, p, 1) for batch matmul
    mu_vec = mu.permute(1, 0).unsqueeze(-1)  # (N, p, 1)

    # Compute mu^T @ C^Q @ mu for each pixel
    intermediate = torch.matmul(C_q, mu_vec)  # (N, p, 1)
    result = torch.matmul(mu_vec.transpose(1, 2), intermediate).squeeze()  # (N,)

    # Final contrast value per pixel
    CV_flat = torch.sqrt(result ** (1 / Q)) / (norm_mu ** (1 + 1 / Q))

    # Reshape to (nx, ny)
    CV = CV_flat.view(nx, ny).cpu().numpy()

    return CV


def CV_Generalized_Non_equally_Zero_GPU(S, device="cuda"):
    """
    Vectorized version of the Generalized Non Equally Weighted Mean contrast at order 0.
    Works entirely on GPU.
    """
    eps = 1e-8

    # Stack input list into one tensor: (p, Nt, nx, ny)
    S_torch = torch.stack(
        [torch.tensor(s, dtype=torch.float32, device=device) for s in S]
    )  # (p, Nt, nx, ny)
    p, Nt, nx, ny = S_torch.shape

    # Reshape for vectorized operations: (p, Nt, N), where N = nx * ny
    S_flat = S_torch.view(p, Nt, -1)  # (p, Nt, N)
    N = S_flat.shape[2]

    # Mean over time for each (p, N)
    mu = S_flat.mean(dim=1)  # (p, N)
    norm_mu = mu.norm(dim=0) + eps  # (N,)

    # Centered data for covariance computation
    centered = S_flat - mu.unsqueeze(1)  # (p, Nt, N)

    # Compute covariance matrices: (N, p, p)
    centered_T = centered.permute(2, 0, 1)  # (N, p, Nt)
    C = torch.matmul(centered_T, centered_T.transpose(1, 2)) / Nt  # (N, p, p)

    # Corriger les matrices non finies
    invalid_mask = ~torch.isfinite(C).all(dim=(1, 2))
    if invalid_mask.any():
        C[invalid_mask] = torch.eye(p, device=device).unsqueeze(0) * 1e-3

    # Eigen decomposition
    eigenvalues, U = torch.linalg.eigh(C)  # (N, p)
    eigenvalues = torch.clamp(eigenvalues, min=1e-8)  # éviter log(0)

    # Project mu into eigenbasis
    mu_T = mu.T  # (N, p)
    mu_prime = torch.matmul(U.transpose(1, 2), mu_T.unsqueeze(2)).squeeze(2)  # (N, p)

    # Weights and log-product
    weights = mu_prime.abs() ** 2  # (N, p)
    log_eigen_prod = torch.sum(weights * torch.log(eigenvalues + eps), dim=1)  # (N,)

    # Final result
    result = torch.exp(log_eigen_prod / (norm_mu**2 + eps))  # (N,)
    CV_flat = torch.sqrt(result) / norm_mu  # (N,)

    # Reshape to (nx, ny)
    CV = CV_flat.view(nx, ny)
    return CV.cpu().numpy()


def CV_Generalized_Limits_GPU_Chunked(series_np, device="cuda", chunk_size=10000):
    """
    Memory-efficient PyTorch version of CV_Generalized_Limits using GPU and chunking.
    Inputs:
        - series_np: list of p numpy arrays of shape (Nt, nx, ny)
        - chunk_size: number of pixels to process per batch (tune depending on your GPU)
    Returns:
        - limit_min, limit_max: (nx, ny) arrays
    """
    # Stack input and send to device
    series = torch.tensor(np.stack(series_np), dtype=torch.float32)  # (p, Nt, nx, ny)
    p, Nt, nx, ny = series.shape
    N = nx * ny

    series = series.view(p, Nt, N)  # shape: (p, Nt, N)

    # Pre-allocate output arrays
    limit_min = torch.empty(N, device="cpu")
    limit_max = torch.empty(N, device="cpu")

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        idx = slice(start, end)

        # Move chunk to device
        series_chunk = series[:, :, idx].to(device)  # (p, Nt, chunk_size)

        # Compute mean and norm per pixel
        mu = series_chunk.mean(dim=1)  # (p, chunk_size)
        normMU = torch.norm(mu, dim=0)  # (chunk_size,)

        # Centered data
        centered = series_chunk - series_chunk.mean(
            dim=1, keepdim=True
        )  # (p, Nt, chunk_size)

        # Covariance for each pixel: (p, p, chunk_size)
        cov = (
            torch.einsum("pti,qti->pqi", centered, centered) / Nt
        )  # (p, p, chunk_size)
        cov = cov.permute(2, 0, 1)  # (chunk_size, p, p)

        # Eigendecomposition
        eigvals = torch.linalg.eigvalsh(cov)  # (chunk_size, p)

        eig_min = eigvals[:, 0]
        eig_max = eigvals[:, -1]

        # Keep eps very small
        eps = 1e-8

        # Clamp only *negative or zero* eigenvalues — NOT everything
        eig_min = torch.where(eig_min <= 0, torch.full_like(eig_min, eps), eig_min)
        eig_max = torch.where(eig_max <= 0, torch.full_like(eig_max, eps), eig_max)

        # Avoid divide-by-zero in normMU, but only where needed
        normMU = torch.where(normMU < eps, torch.full_like(normMU, eps), normMU)

        # Now compute
        lmin = torch.sqrt(eig_min) / normMU
        lmax = torch.sqrt(eig_max) / normMU

        # Move results to CPU and store
        limit_min[idx] = lmin.cpu()
        limit_max[idx] = lmax.cpu()

    # Reshape to (nx, ny)
    return limit_min.view(nx, ny).numpy(), limit_max.view(nx, ny).numpy()


def CV_Generalized_Limits_GPU_Chunked(series_np, device="cuda", chunk_size=10000):
    """
    Memory-efficient PyTorch version of CV_Generalized_Limits using GPU and chunking.
    Inputs:
        - series_np: list of p numpy arrays of shape (Nt, nx, ny)
        - chunk_size: number of pixels to process per batch (tune depending on your GPU)
    Returns:
        - limit_min, limit_max: (nx, ny) arrays
    """
    # Stack input and send to device
    series = torch.tensor(np.stack(series_np), dtype=torch.float32)  # (p, Nt, nx, ny)
    p, Nt, nx, ny = series.shape
    N = nx * ny

    series = series.view(p, Nt, N)  # shape: (p, Nt, N)

    # Pre-allocate output arrays
    limit_min = torch.empty(N, device="cpu")
    limit_max = torch.empty(N, device="cpu")

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        idx = slice(start, end)

        # Move chunk to device
        series_chunk = series[:, :, idx].to(device)  # (p, Nt, chunk_size)

        # Compute mean and norm per pixel
        mu = series_chunk.mean(dim=1)  # (p, chunk_size)
        normMU = torch.norm(mu, dim=0)  # (chunk_size,)

        # Centered data
        centered = series_chunk - series_chunk.mean(
            dim=1, keepdim=True
        )  # (p, Nt, chunk_size)

        # Covariance for each pixel: (p, p, chunk_size)
        cov = (
            torch.einsum("pti,qti->pqi", centered, centered) / Nt
        )  # (p, p, chunk_size)
        cov = cov.permute(2, 0, 1)  # (chunk_size, p, p)

        # Eigendecomposition
        eigvals = torch.linalg.eigvalsh(cov)  # (chunk_size, p)

        eig_min = eigvals[:, 0]
        eig_max = eigvals[:, -1]

        # Keep eps very small
        eps = 1e-8

        # Clamp only *negative or zero* eigenvalues — NOT everything
        eig_min = torch.where(eig_min <= 0, torch.full_like(eig_min, eps), eig_min)
        eig_max = torch.where(eig_max <= 0, torch.full_like(eig_max, eps), eig_max)

        # Avoid divide-by-zero in normMU, but only where needed
        normMU = torch.where(normMU < eps, torch.full_like(normMU, eps), normMU)

        # Now compute
        lmin = torch.sqrt(eig_min) / normMU
        lmax = torch.sqrt(eig_max) / normMU

        # Move results to CPU and store
        limit_min[idx] = lmin.cpu()
        limit_max[idx] = lmax.cpu()

    # Reshape to (nx, ny)
    return limit_min.view(nx, ny).numpy(), limit_max.view(nx, ny).numpy()

def CV_Generalized_Limits_GPU_Chunked_low_memory(series_np, device="cuda", chunk_size=10000):
    """
    Memory-efficient PyTorch version of CV_Generalized_Limits using GPU and chunking.
    """
    with torch.no_grad():
        # Convert to tensor and reshape
        series = torch.tensor(np.stack(series_np), dtype=torch.float32)  # (p, Nt, nx, ny)
        p, Nt, nx, ny = series.shape
        N = nx * ny
        series = series.view(p, Nt, N)  # (p, Nt, N)

        # Pre-allocate output tensors on CPU
        limit_min = torch.empty(N, dtype=torch.float32)
        limit_max = torch.empty(N, dtype=torch.float32)

        eps = 1e-8

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            idx = slice(start, end)

            # Move chunk to GPU
            series_chunk = series[:, :, idx].to(device)  # (p, Nt, chunk_size)

            # Mean and norm
            mu = series_chunk.mean(dim=1)  # (p, chunk_size)
            normMU = torch.norm(mu, dim=0)  # (chunk_size,)

            # Center data
            centered = series_chunk - mu.unsqueeze(1)  # (p, Nt, chunk_size)

            # Covariance per pixel (chunk_size, p, p)
            cov = torch.einsum("pti,qti->pqi", centered, centered) / Nt
            cov = cov.permute(2, 0, 1)

            # Eigendecomposition
            eigvals = torch.linalg.eigvalsh(cov)  # (chunk_size, p)
            eig_min = torch.clamp(eigvals[:, 0], min=eps)
            eig_max = torch.clamp(eigvals[:, -1], min=eps)
            normMU = torch.clamp(normMU, min=eps)

            # Compute limits
            lmin = torch.sqrt(eig_min) / normMU
            lmax = torch.sqrt(eig_max) / normMU

            # Store in CPU tensors
            limit_min[idx] = lmin.cpu()
            limit_max[idx] = lmax.cpu()

            # Cleanup
            del series_chunk, mu, centered, cov, eigvals, eig_min, eig_max, normMU, lmin, lmax
            torch.cuda.empty_cache()
            gc.collect()

        # Reshape to (nx, ny)
        return limit_min.view(nx, ny).numpy(), limit_max.view(nx, ny).numpy()