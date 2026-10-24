"""Mean-variance weights under a hard stress-loss limit, solved without scipy.

    maximize    excess·x − (risk_aversion / 2)·xᵀ Σ x
    subject to  0 ≤ x ≤ max_weight (a scalar or one cap per asset),
                Σ x ≤ 1,   L x ≤ loss_limit  (every row of L)

``x`` holds the risky weights; whatever they leave over is cash. ``L`` is one row
of losses (each asset's worst drop) or several (one per crash scenario day), and
the limit holds for every row. The problem is a small convex QP, solved by ADMM
in the form OSQP uses: every constraint is a row of one matrix, and each step
solves one linear system whose matrix is only assets × assets, however many loss
rows there are. That matrix is inverted once, and again only when the step size
adapts, so thousands of scenario rows cost little more than one.
"""

from __future__ import annotations

import numpy as np


def solve_weights(excess: np.ndarray, cov: np.ndarray, worst_loss: np.ndarray,
                  loss_limit: float | None, max_weight, risk_aversion: float,
                  eps: float = 1e-7, max_iter: int = 20000, start: np.ndarray | None = None) -> np.ndarray:
    """Risky weights maximizing the objective above; the cash weight is ``1 - sum``.

    ``start`` is where the search begins (zero weights if not given); a nearby
    answer, such as the last one, makes it converge sooner.
    """
    n = len(excess)
    upper = np.broadcast_to(np.asarray(max_weight, dtype=float), (n,))
    losses = np.atleast_2d(np.asarray(worst_loss, dtype=float))
    blocks, lo, hi = [np.eye(n), np.ones((1, n))], [np.zeros(n), [-np.inf]], [upper, [1.0]]
    if loss_limit is not None and len(losses):
        blocks.append(losses)
        lo.append(np.full(len(losses), -np.inf))
        hi.append(np.full(len(losses), loss_limit))
    A, l, u = np.vstack(blocks), np.concatenate(lo), np.concatenate(hi)
    # Rows of unit length, so one step size suits the box, the budget and the loss rows alike.
    scale = 1 / np.maximum(np.linalg.norm(A, axis=1), 1e-12)
    A, l, u = A * scale[:, None], l * scale, u * scale
    P, q = risk_aversion * cov, -np.asarray(excess, dtype=float)

    sigma, alpha, rho = 1e-6, 1.6, 0.1
    x = np.zeros(n) if start is None else np.clip(np.asarray(start, dtype=float), 0.0, upper)
    z, y = np.clip(A @ x, l, u), np.zeros(len(l))

    def factor(rho):
        return np.linalg.inv(P + sigma * np.eye(n) + rho * A.T @ A)

    K = factor(rho)
    for k in range(max_iter):
        x_step = K @ (sigma * x - q + A.T @ (rho * z - y))
        x = alpha * x_step + (1 - alpha) * x
        relaxed = alpha * (A @ x_step) + (1 - alpha) * z
        z_next = np.clip(relaxed + y / rho, l, u)
        y = y + rho * (relaxed - z_next)
        z = z_next
        if k % 25:
            continue
        Ax = A @ x
        primal, dual = np.abs(Ax - z).max(), np.abs(P @ x + q + A.T @ y).max()
        if primal < eps and dual < eps:
            break
        if k and k % 100 == 0:
            # Rebalance the two residuals, as OSQP does; refactor only on a large change.
            rel_primal = primal / max(np.abs(Ax).max(), np.abs(z).max(), 1e-12)
            rel_dual = dual / max(np.abs(P @ x).max(), np.abs(A.T @ y).max(), np.abs(q).max(), 1e-12)
            new = min(max(rho * np.sqrt(rel_primal / max(rel_dual, 1e-12)), 1e-6), 1e6)
            if not rho / 5 <= new <= rho * 5:
                rho, K = new, factor(new)
    return _finish(x, losses, loss_limit, upper)


def _finish(x, losses, loss_limit, upper) -> np.ndarray:
    """Clear the residue so the loss limit and the budget hold exactly."""
    x = np.clip(x, 0.0, upper)
    worst = float((losses @ x).max()) if len(losses) else 0.0
    if loss_limit is not None and worst > loss_limit > 0:
        x *= loss_limit / worst
    if x.sum() > 1:
        x /= x.sum()
    return x
