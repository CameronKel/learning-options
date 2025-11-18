import math
import random

import numpy as np
from flask import request, jsonify, render_template

from . import bp

# Import your Black/Greeks from black.py
from black import (
    d1, d2,
    call_price, put_price,
    call_delta, put_delta,
    gamma, theta, vega,
    vanna, charm, volga,
    call_dual_delta, put_dual_delta,
    dual_gamma, rho
)


def discount_factor(r, T):
    return math.exp(-r * T)


def compute_greeks(F, K, T, sigma, r, option_type):
    # Guardrails
    T = max(T, 0.01)
    sigma = max(sigma, 1e-4)

    df = discount_factor(r, T)

    d1_val = d1(F, K, T, sigma)
    d2_val = d2(d1_val, T, sigma)

    if option_type == "call":
        price = call_price(F, K, d1_val, d2_val, df)
        delta = call_delta(d1_val, df)
        dual_delta_val = call_dual_delta(d2_val, df)
    else:
        price = put_price(F, K, d1_val, d2_val, df)
        delta = put_delta(d1_val, df)
        dual_delta_val = put_dual_delta(d2_val, df)

    gamma_val = gamma(d1_val, F, T, sigma, df)
    theta_val = theta(d1_val, df, price, r, T, sigma, F)
    vega_val = vega(d1_val, F, T, df)
    vanna_val = vanna(df, sigma, d1_val, d2_val)
    charm_val = charm(r, delta, df, T, d2_val, d1_val)
    volga_val = volga(df, F, T, sigma, d1_val, d2_val)
    dual_gamma_val = dual_gamma(df, K, sigma, T, d2_val)
    rho_val = rho(price, T)

    F_grid = np.linspace(F * 0.5, F * 1.5, 200)

    price_grid = []
    delta_grid = []
    gamma_grid = []
    theta_grid = []
    vega_grid = []
    vanna_grid = []
    volga_grid = []
    rho_grid = []
    charm_grid = []

    for Fg in F_grid:
        d1g = d1(Fg, K, T, sigma)
        d2g = d2(d1g, T, sigma)

        if option_type == "call":
            price_g = call_price(Fg, K, d1g, d2g, df)
            delta_g = call_delta(d1g, df)
        else:
            price_g = put_price(Fg, K, d1g, d2g, df)
            delta_g = put_delta(d1g, df)

        gamma_g = gamma(d1g, Fg, T, sigma, df)
        theta_g = theta(d1g, df, price_g, r, T, sigma, Fg)
        vega_g = vega(d1g, Fg, T, df)
        vanna_g = vanna(df, sigma, d1g, d2g)
        volga_g = volga(df, Fg, T, sigma, d1g, d2g)
        rho_g = rho(price_g, T)
        charm_g = charm(r, delta_g, df, T, d2g, d1g)

        price_grid.append(float(price_g))
        delta_grid.append(float(delta_g))
        gamma_grid.append(float(gamma_g))
        theta_grid.append(float(theta_g))
        vega_grid.append(float(vega_g))
        vanna_grid.append(float(vanna_g))
        volga_grid.append(float(volga_g))
        rho_grid.append(float(rho_g))
        charm_grid.append(float(charm_g))

    return {
        "F": float(F),
        "K": float(K),
        "T": float(T),
        "sigma": float(sigma),
        "r": float(r),
        "option_type": option_type,
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma_val),
        "theta": float(theta_val),
        "vega": float(vega_val),
        "vanna": float(vanna_val),
        "charm": float(charm_val),
        "volga": float(volga_val),
        "dual_delta": float(dual_delta_val),
        "dual_gamma": float(dual_gamma_val),
        "rho": float(rho_val),
        "F_grid": F_grid.tolist(),
        "price_grid": price_grid,
        "delta_grid": delta_grid,
        "gamma_grid": gamma_grid,
        "theta_grid": theta_grid,
        "vega_grid": vega_grid,
        "vanna_grid": vanna_grid,
        "volga_grid": volga_grid,
        "rho_grid": rho_grid,
        "charm_grid": charm_grid,
    }


# ===== Greeks UI & API =====

@bp.route("/greeks", methods=["GET"])
def greeks():
    return render_template("greeks.html", active_page="greeks")


@bp.route("/api/greeks", methods=["POST"])
def api_greeks():
    data = request.get_json(force=True) or {}

    primary = data.get("primary", {})
    comparison = data.get("comparison", {})
    comparison_enabled = bool(data.get("comparison_enabled", False))

    # Primary
    F1 = float(primary.get("F", 100.0))
    K1 = float(primary.get("K", 100.0))
    T1 = float(primary.get("T", 1.0))
    sigma1 = float(primary.get("sigma", 0.20))
    r1 = float(primary.get("r", 0.01))
    opt_type1 = primary.get("option_type", "call")
    if opt_type1 not in ("call", "put"):
        opt_type1 = "call"

    res1 = compute_greeks(F1, K1, T1, sigma1, r1, opt_type1)

    resp = {
        "primary": res1,
        "comparison_enabled": comparison_enabled,
        "comparison": None
    }

    if comparison_enabled:
        F2 = float(comparison.get("F", F1))
        K2 = float(comparison.get("K", K1))
        T2 = float(comparison.get("T", T1))
        sigma2 = float(comparison.get("sigma", sigma1))
        r2 = float(comparison.get("r", r1))
        opt_type2 = comparison.get("option_type", opt_type1)
        if opt_type2 not in ("call", "put"):
            opt_type2 = "call"

        res2 = compute_greeks(F2, K2, T2, sigma2, r2, opt_type2)
        resp["comparison"] = res2

    return jsonify(resp)


# ===== Put-Call Parity Game =====

def _cents_to_price(cents: int) -> float:
    # Ensure two-decimal float
    return round(cents / 100.0, 2)


def generate_parity_question():
    """
    Generate a random question obeying:
        C - P = F - K
    with:
      - all values >= 0
      - prices in 0.01 increments (internally: cents)
      - F and K around 100
      - K a multiple of 0.50
    One of {C, P, F, K} is missing.
    """
    while True:
        missing = random.choice(["C", "P", "F", "K"])

        # We'll always work in cents for exact 0.01 increments.
        C_c = P_c = F_c = K_c = None

        if missing == "P":
            # Pick F and K around 100, K multiple of 0.50
            F_c = random.randint(9000, 11000)  # 90.00 - 110.00
            K_c = random.randrange(9000, 11001, 50)  # 90.00 - 110.00 step 0.50
            S_c = F_c - K_c  # can be negative

            lower_C = max(S_c, 0)  # ensure P >= 0
            upper_C = lower_C + 2000  # up to +20.00 over that
            C_c = random.randint(lower_C, upper_C)
            P_c = C_c - S_c
            if P_c < 0:
                continue  # safety

        elif missing == "C":
            F_c = random.randint(9000, 11000)
            K_c = random.randrange(9000, 11001, 50)
            S_c = F_c - K_c
            P_c = random.randint(0, 2000)  # 0.00 - 20.00
            C_c = P_c + S_c
            if C_c < 0:
                continue

        elif missing == "F":
            K_c = random.randrange(9000, 11001, 50)
            C_c = random.randint(0, 2000)
            P_c = random.randint(0, 2000)
            F_c = C_c - P_c + K_c
            if F_c <= 0 or F_c < 9000 or F_c > 11000:
                continue

        else:  # missing == "K"
            F_c = random.randint(9000, 11000)
            C_c = random.randint(0, 2000)
            P_c = random.randint(0, 2000)
            K_c = F_c + P_c - C_c
            if K_c <= 0 or K_c < 9000 or K_c > 11000:
                continue
            if K_c % 50 != 0:
                continue  # enforce multiple of 0.50

        # If we got here, all four are defined and valid
        break

    C = _cents_to_price(C_c)
    P = _cents_to_price(P_c)
    F = _cents_to_price(F_c)
    K = _cents_to_price(K_c)

    answer_map = {"C": C, "P": P, "F": F, "K": K}

    question = {
        "missing": missing,
        "C": None if missing == "C" else C,
        "P": None if missing == "P" else P,
        "F": None if missing == "F" else F,
        "K": None if missing == "K" else K,
        "answer": answer_map[missing],
    }
    return question


@bp.route("/parity-game", methods=["GET"])
def parity_game():
    return render_template("parity_game.html", active_page="parity_game")


@bp.route("/api/parity-question", methods=["GET"])
def parity_question():
    q = generate_parity_question()
    return jsonify(q)
