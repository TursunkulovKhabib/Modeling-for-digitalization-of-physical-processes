from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def _safe_time_grid(t_max: float, dt: float) -> np.ndarray:
    """
    Сетка времени, гарантированно лежащая в [0, t_max] и включающая t_max,
    чтобы satisfy solve_ivp: "Values in t_eval are within t_span and sorted".
    """
    if t_max <= 0 or dt <= 0:
        return np.array([0.0, max(t_max, 0.0)], dtype=float)
    n = int(np.floor(t_max / dt))
    t = dt * np.arange(n + 1, dtype=float)
    if t[-1] < t_max:
        t = np.append(t, float(t_max))
    t = np.unique(np.round(t, 15))
    return t

def reflect_velocity(v: np.ndarray, n_unit: np.ndarray) -> np.ndarray:
    """
    Абсолютно упругое отражение о плоскость с единичной нормалью n_unit: v' = v - 2 (v·n) n.
    """
    n = n_unit / np.linalg.norm(n_unit)
    return v - 2.0 * np.dot(v, n) * n

def collide_elastic_two_spheres(
    m1: float, m2: float,
    x1: np.ndarray, v1: np.ndarray,
    x2: np.ndarray, v2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Угло‑свободная формула постударных скоростей двух гладких сфер при e=1.
    """
    dx = x1 - x2
    dist2 = float(np.dot(dx, dx))
    if dist2 == 0.0:
        return v1.copy(), v2.copy()
    factor1 = (2.0 * m2 / (m1 + m2)) * float(np.dot(v1 - v2, dx)) / dist2
    factor2 = (2.0 * m1 / (m1 + m2)) * float(np.dot(v2 - v1, -dx)) / dist2
    v1p = v1 - factor1 * dx
    v2p = v2 - factor2 * (-dx)
    return v1p, v2p

@dataclass
class Ball:
    m: float
    R: float
    x: np.ndarray
    v: np.ndarray

def contact_force(delta: float, n_unit: np.ndarray, k: float, p: float) -> np.ndarray:
    """
    Нормальная контактная сила: F = k * delta^p * n_unit (p=1 — Хук, p=1.5 — Герц).
    """
    if delta <= 0.0:
        return np.zeros_like(n_unit)
    return (k * (delta ** p)) * n_unit

def potential_energy(delta: float, k: float, p: float) -> float:
    """
    Потенциальная энергия контакта: U = k * delta^(p+1) / (p+1) для delta>0, иначе 0.
    """
    if delta <= 0.0:
        return 0.0
    return k * (delta ** (p + 1.0)) / (p + 1.0)

def potential_energy_vec(delta_vec: np.ndarray, k: float, p: float) -> np.ndarray:
    U = np.zeros_like(delta_vec)
    mask = delta_vec > 0.0
    U[mask] = k * delta_vec[mask] ** (p + 1.0) / (p + 1.0)
    return U

def two_ball_ode(
    t: float, y: np.ndarray, m1: float, m2: float, R1: float, R2: float, k: float, p: float
) -> np.ndarray:
    x1 = y[0:2]
    v1 = y[2:4]
    x2 = y[4:6]
    v2 = y[6:8]
    r = x1 - x2
    dist = float(np.linalg.norm(r))
    delta = (R1 + R2) - dist
    if dist > 0.0:
        n = r / dist
    else:
        n = np.array([1.0, 0.0])
    F = contact_force(delta, n, k, p)
    a1 = -F / m1
    a2 = F / m2
    return np.hstack([v1, a1, v2, a2])

def simulate_two_balls(
    b1: Ball, b2: Ball,
    k: float = 5e5, p: float = 1.5,
    t_max: float = 0.01, dt: float = 2e-6,
    rtol: float = 1e-8, atol: float = 1e-10,
    method: str = "RK45"
):
    """
    Численное моделирование упругого контакта двух сфер, возвращает траектории и инварианты.
    """
    y0 = np.hstack([b1.x, b1.v, b2.x, b2.v])
    t_eval = _safe_time_grid(t_max, dt)
    sol = solve_ivp(
        two_ball_ode, (0.0, t_max), y0,
        t_eval=t_eval, rtol=rtol, atol=atol, max_step=dt, method=method,
        args=(b1.m, b2.m, b1.R, b2.R, k, p)
    )
    Y = sol.y.T
    x1 = Y[:, 0:2]
    v1 = Y[:, 2:4]
    x2 = Y[:, 4:6]
    v2 = Y[:, 6:8]
    r = x1 - x2
    dist = np.linalg.norm(r, axis=1)
    delta = np.maximum((b1.R + b2.R) - dist, 0.0)
    KE = 0.5 * b1.m * np.sum(v1**2, axis=1) + 0.5 * b2.m * np.sum(v2**2, axis=1)
    U = potential_energy_vec(delta, k, p)
    E = KE + U
    P = b1.m * v1 + b2.m * v2
    return sol.t, x1, v1, x2, v2, delta, E, P

def wall_contact_force(
    x: np.ndarray, R: float, n_unit: np.ndarray, x0_on_plane: np.ndarray, k: float, p: float
) -> np.ndarray:
    n = n_unit / np.linalg.norm(n_unit)
    s = float(np.dot(x - x0_on_plane, n))
    delta = R - s
    if delta <= 0.0:
        return np.zeros_like(n)
    return (k * (delta ** p)) * n

def simulate_ball_wall(
    m: float = 0.17, R: float = 0.0285,
    x: np.ndarray = np.array([0.10, 0.0]),
    v: np.ndarray = np.array([-3.0, 0.4]),
    n_unit: np.ndarray = np.array([1.0, 0.0]),
    x0_on_plane: np.ndarray = np.array([0.0, 0.0]),
    k: float = 1e6, p: float = 1.5,
    t_max: float = 0.01, dt: float = 2e-6,
    rtol: float = 1e-8, atol: float = 1e-10,
    method: str = "RK45"
):
    def ode(t, y):
        x_ = y[:2]
        v_ = y[2:]
        F = wall_contact_force(x_, R, n_unit, x0_on_plane, k, p)
        a = F / m
        return np.hstack([v_, a])
    y0 = np.hstack([x, v])
    t_eval = _safe_time_grid(t_max, dt)
    sol = solve_ivp(
        ode, (0.0, t_max), y0,
        t_eval=t_eval, rtol=rtol, atol=atol, max_step=dt, method=method
    )
    X = sol.y[:2, :].T
    V = sol.y[2:, :].T
    KE = 0.5 * m * np.sum(V**2, axis=1)
    return sol.t, X, V, KE

def relative_energy_drift(E: np.ndarray) -> float:
    e0 = max(float(E[0]), 1e-16)
    return (float(np.max(E)) - float(np.min(E))) / e0

def momentum_drift(P: np.ndarray) -> float:
    return float(np.linalg.norm(P[-1] - P[0]))

def run_autotests(verbose: bool = True) -> None:
    """
    Набор автотестов:
    1) Сохранение полной энергии и импульса при контакте без внешних сил.
    2) Отражение о стенку: проверка качественного поведения.
    """
    tol_E = 2e-4
    tol_P = 2e-6
    tol_v = 2e-2
    m1 = m2 = 0.17
    R = 0.0285
    b1 = Ball(m1, R, np.array([-0.060, 0.0]), np.array([2.5, 0.0]))
    b2 = Ball(m2, R, np.array([+0.060, 0.0]), np.array([0.0, 0.0]))
    v1p, v2p = collide_elastic_two_spheres(m1, m2, b1.x, b1.v, b2.x, b2.v)
    t, x1, v1, x2, v2, delta, E, P = simulate_two_balls(b1, b2, k=5e5, p=1.5, t_max=0.006, dt=2e-6)
    assert relative_energy_drift(E) < tol_E, f"Тест1 энергия: {relative_energy_drift(E):.2e} >= {tol_E:.2e}"
    assert momentum_drift(P) < tol_P, f"Тест1 импульс: {momentum_drift(P):.2e} >= {tol_P:.2e}"
    if verbose:
        print(f"[OK] Тест1: энергия/импульс в норме (дрейф энергии: {relative_energy_drift(E):.2e})")
        print(f"     Аналитика: v1={v1p}, v2={v2p}")
        print(f"     Численно:  v1={v1[-1]}, v2={v2[-1]}")
    m1, m2 = 0.20, 0.10
    R = 0.0285
    b1 = Ball(m1, R, np.array([-0.060, -0.005]), np.array([2.2, 0.3]))
    b2 = Ball(m2, R, np.array([+0.060,  0.004]), np.array([0.0, 0.0]))
    v1p, v2p = collide_elastic_two_spheres(m1, m2, b1.x, b1.v, b2.x, b2.v)
    t, x1, v1, x2, v2, delta, E, P = simulate_two_balls(b1, b2, k=8e5, p=1.5, t_max=0.008, dt=2e-6)
    assert relative_energy_drift(E) < tol_E, f"Тест2 энергия: {relative_energy_drift(E):.2e} >= {tol_E:.2e}"
    assert momentum_drift(P) < 5e-6, f"Тест2 импульс: {momentum_drift(P):.2e} >= 5e-6"
    if verbose:
        print(f"[OK] Тест2: неравные массы, 2D, энергия/импульс в норме (дрейф энергии: {relative_energy_drift(E):.2e})")
    v_in = np.array([-1.0, 0.2])
    n = np.array([1.0, 0.0])
    v_ref = reflect_velocity(v_in, n)
    t, X, V, KE = simulate_ball_wall(
        m=0.17, R=0.0285,
        x=np.array([0.05, 0.0]),
        v=v_in,
        n_unit=n,
        k=1e6, p=1.0, t_max=0.05, dt=1e-6
    )
    energy_drift = (np.max(KE) - np.min(KE)) / np.max(KE)
    final_x = X[-1, 0]
    initial_x = X[0, 0]
    assert final_x > initial_x, f"Тест3 стена: шар не отлетел {final_x:.3f} <= {initial_x:.3f}"
    v_final_tangent = V[-1, 1]
    v_initial_tangent = v_in[1]
    tangent_diff = abs(v_final_tangent - v_initial_tangent)
    assert tangent_diff < 0.05, f"Тест3 стена: тангенциальная компонента изменилась {tangent_diff:.2f}"
    v_final_normal = V[-1, 0]
    v_initial_normal = v_in[0]
    assert np.sign(v_final_normal) != np.sign(v_initial_normal), \
        f"Тест3 стена: нормальная компонента не изменила знак {v_final_normal:.2f}"
    if verbose:
        print(f"[OK] Тест3: отражение о стенку (дрейф энергии: {energy_drift:.2e})")
        print(f"     Начальная: x={X[0, 0]:.3f}, v={V[0]}")
        print(f"     Конечная:  x={X[-1, 0]:.3f}, v={V[-1]}")
        print(f"     Ожидалось: {v_ref}")
    if verbose:
        print("Все автотесты пройдены успешно.")

def demo_plots():
    """
    Показ траекторий и энергетики с выделением фазы контакта и силы Герца.
    """
    m1 = m2 = 0.17
    R = 0.0285
    b1 = Ball(m1, R, np.array([-0.070, -0.004]), np.array([2.6, 0.25]))
    b2 = Ball(m2, R, np.array([+0.070,  0.003]), np.array([0.0, 0.0]))
    k_demo = 7e5
    p_demo = 1.5
    t, x1, v1, x2, v2, delta, E, P = simulate_two_balls(b1, b2, k=k_demo, p=p_demo, t_max=0.012, dt=1e-6)
    KE = 0.5 * m1 * np.sum(v1**2, axis=1) + 0.5 * m2 * np.sum(v2**2, axis=1)
    U = potential_energy_vec(delta, k_demo, p_demo)
    Fn = k_demo * (delta ** p_demo)
    coll_mask = delta > 0
    idx_coll = np.where(coll_mask)[0]
    idx0 = int(idx_coll[0]) if idx_coll.size > 0 else 0
    idx1 = int(idx_coll[-1]) if idx_coll.size > 0 else len(t) - 1
    n_vec = x1[idx0] - x2[idx0]
    n_norm = np.linalg.norm(n_vec)
    if n_norm > 0:
        n_hat = n_vec / n_norm
    else:
        n_hat = np.array([1.0, 0.0])
    vrel0 = float(np.dot(v1[idx0] - v2[idx0], n_hat))
    vrel1 = float(np.dot(v1[idx1] - v2[idx1], n_hat))
    e_n = (-vrel1 / vrel0) if abs(vrel0) > 1e-12 else np.nan
    energy_drift = (np.max(E) - np.min(E)) / max(E[0], 1e-16)
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
    ax = axes[0]
    ax.plot(x1[:, 0], x1[:, 1], 'b-', lw=2, label="Шар 1")
    ax.plot(x2[:, 0], x2[:, 1], 'r-', lw=2, label="Шар 2")
    ax.plot(x1[0, 0], x1[0, 1], 'bo', ms=7, label="Старт 1")
    ax.plot(x2[0, 0], x2[0, 1], 'ro', ms=7, label="Старт 2")
    ax.plot(x1[-1, 0], x1[-1, 1], 'bs', ms=6, label="Финиш 1")
    ax.plot(x2[-1, 0], x2[-1, 1], 'rs', ms=6, label="Финиш 2")
    if idx_coll.size > 0:
        cx = 0.5 * (x1[idx0:idx1+1, 0] + x2[idx0:idx1+1, 0])
        cy = 0.5 * (x1[idx0:idx1+1, 1] + x2[idx0:idx1+1, 1])
        ax.plot(cx[np.argmax(delta[idx0:idx1+1])], cy[np.argmax(delta[idx0:idx1+1])], 'g*', ms=14, label="Контакт")
    ax.set_xlabel('x, м')
    ax.set_ylabel('y, м')
    ax.set_title('Траектории двух шаров (p=1.5)')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(loc='best', fontsize=9)
    ax = axes[1]
    ax.plot(t, E, 'k-', lw=2, label='Полная энергия')
    ax.plot(t, KE, 'b--', lw=1.8, label='Кинетическая')
    ax.plot(t, U, 'r--', lw=1.8, label='Потенциальная')
    if idx_coll.size > 0:
        ax.axvspan(t[idx0], t[idx1], color='orange', alpha=0.15, label='Фаза контакта')
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Энергия, Дж')
    ax.set_title(f'Энергия и фаза контакта (дрейф={energy_drift:.2e}, e_n={e_n:.3f})')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax = axes[2]
    ax.plot(t, 1e3 * delta, color='purple', lw=2, label='Деформация, мм')
    ax2 = ax.twinx()
    ax2.plot(t, Fn, color='darkgreen', lw=2, label='Нормальная сила, Н', alpha=0.9)
    if idx_coll.size > 0:
        t_peak = t[idx0 + int(np.argmax(delta[idx0:idx1+1]))]
        ax.axvline(t_peak, color='crimson', ls='--', alpha=0.7)
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Деформация, мм', color='purple')
    ax2.set_ylabel('Сила, Н', color='darkgreen')
    ax.set_title('Деформация и сила контакта')
    ax.grid(True, alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='best', fontsize=9)
    plt.show()

def main():
    run_autotests(verbose=True)
    demo_plots()

if __name__ == "__main__":
    main()
