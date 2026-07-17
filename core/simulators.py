# simulators.py
# Ground-truth agents for pipeline validation (no LLM required):
#   1. ProjectiveAgent : 2D projective sequential measurement -> q_QQ must be 0
#   2. POVMAgent       : unsharp (noisy) measurements -> q_QQ may deviate from 0
#   3. AnchoringAgent  : symmetric classical repetition -> satisfies QQ
#      identically (interpretive counterexample for H-A; see validation
#      report 2026-07-12_1445 Sec 4.1)
# All agents expose exact joints for both orders, plus optional `other` mass
# injection and finite sampling for envelope / Wilson-check validation.
# ASCII only. English comments.

import numpy as np

RNG = np.random.default_rng


def _projector(theta):
    """Rank-1 projector onto (cos t, sin t) and its complement in R^2."""
    v = np.array([np.cos(theta), np.sin(theta)])
    p = np.outer(v, v)
    return p, np.eye(2) - p


class ProjectiveAgent:
    """Standard quantum question-order model (Wang-Busemeyer).

    State psi in R^2; question A -> projectors at angle a, question B at
    angle b. Sequential joint: p(x then y) = || P_y P_x psi ||^2.
    The QQ equality holds exactly for this agent.
    """

    def __init__(self, psi_angle, a_angle, b_angle):
        self.psi = np.array([np.cos(psi_angle), np.sin(psi_angle)])
        self.pa_y, self.pa_n = _projector(a_angle)
        self.pb_y, self.pb_n = _projector(b_angle)

    def _joint(self, first, second):
        out = {}
        for x, px in first.items():
            for y, py in second.items():
                v = py @ (px @ self.psi)
                out[(x, y)] = float(v @ v)
        return out

    def joints(self):
        a = {"y": self.pa_y, "n": self.pa_n}
        b = {"y": self.pb_y, "n": self.pb_n}
        joint_ab = self._joint(a, b)  # keyed (a_answer, b_answer)
        joint_ba = self._joint(b, a)  # keyed (b_answer, a_answer)
        return joint_ab, joint_ba


class POVMAgent(ProjectiveAgent):
    """Unsharp two-outcome measurements: E_y = (1-eta) P_y + eta/2 I.

    Implemented with Kraus operators K_y = sqrt(1-eta) P_y + sqrt(eta/2) ...
    For simplicity we use the standard noisy-projective instrument:
    K_y = sqrt((1-eta)) P_y + sqrt(eta/2) I is NOT a valid Kraus pair in
    general; instead we mix: with prob (1-eta) perform the projective
    measurement, with prob eta output a coin flip WITHOUT state update
    mismatch -- i.e., a Lueders instrument for the POVM
    E_y = (1-eta) P_y + (eta/2) I, with Kraus K_y = sqrt(E_y).
    """

    def __init__(self, psi_angle, a_angle, b_angle, eta):
        super().__init__(psi_angle, a_angle, b_angle)
        self.eta = eta
        self.ka = {k: self._sqrt_povm(p) for k, p in
                   {"y": self.pa_y, "n": self.pa_n}.items()}
        self.kb = {k: self._sqrt_povm(p) for k, p in
                   {"y": self.pb_y, "n": self.pb_n}.items()}

    def _sqrt_povm(self, proj):
        e = (1.0 - self.eta) * proj + (self.eta / 2.0) * np.eye(2)
        w, v = np.linalg.eigh(e)
        return v @ np.diag(np.sqrt(np.clip(w, 0, None))) @ v.T

    def _joint_k(self, first, second):
        out = {}
        for x, kx in first.items():
            for y, ky in second.items():
                v = ky @ (kx @ self.psi)
                out[(x, y)] = float(v @ v)
        return out

    def joints(self):
        joint_ab = self._joint_k(self.ka, self.kb)
        joint_ba = self._joint_k(self.kb, self.ka)
        return joint_ab, joint_ba


class AnchoringAgent:
    """Classical repetition/anchoring model.

    Base rates pA, pB. First answer ~ base rate of the first question.
    Second answer: with prob r repeat the first answer's polarity, else
    sample from the second question's base rate.
    """

    def __init__(self, p_a, p_b, r):
        self.p_a, self.p_b, self.r = p_a, p_b, r

    def _joint(self, p_first, p_second):
        out = {}
        for x, px in (("y", p_first), ("n", 1 - p_first)):
            for y in "yn":
                base = p_second if y == "y" else 1 - p_second
                rep = 1.0 if y == x else 0.0
                out[(x, y)] = px * (self.r * rep + (1 - self.r) * base)
        return out

    def joints(self):
        joint_ab = self._joint(self.p_a, self.p_b)
        joint_ba = self._joint(self.p_b, self.p_a)
        return joint_ab, joint_ba


# ---------------------------------------------------------------------------
# Helpers for validation runs
# ---------------------------------------------------------------------------

def inject_other_mass(joint, m_first, m_y, m_n):
    """Convert an exact binary joint into 'raw' components with other mass.

    Removes mass proportionally (simulating probability leaked to non-label
    tokens) and returns the raw dict format expected by q_qq_envelope.
    """
    p_first_y = joint[("y", "y")] + joint[("y", "n")]
    p_first = (p_first_y * (1 - m_first), (1 - p_first_y) * (1 - m_first), m_first)

    def _cond(first):
        tot = joint[(first, "y")] + joint[(first, "n")]
        py = joint[(first, "y")] / tot
        m = m_y if first == "y" else m_n
        return (py * (1 - m), (1 - py) * (1 - m), m)

    return {"p_first": p_first,
            "p_second_given_y": _cond("y"),
            "p_second_given_n": _cond("n")}


def sample_joint(joint, n, rng):
    """Draw n joint samples (rejection-sampling analogue) from an exact joint."""
    cells = [("y", "y"), ("y", "n"), ("n", "y"), ("n", "n")]
    probs = np.array([joint[c] for c in cells])
    probs = probs / probs.sum()
    draws = rng.multinomial(n, probs)
    return {c: int(k) for c, k in zip(cells, draws)}


class AsymmetricAnchoringAgent:
    """Polarity-dependent repetition (carryover) classical model.

    Note: distinct from acquiescence/yes-bias, which is a base-rate
    phenomenon; acquiescence only motivates the plausibility of
    polarity-dependent repetition rates (r_y != r_n).

    Repetition probability depends on the polarity of the first answer:
    r_y if first answer was Y, r_n if it was N. Analytically:
        q_QQ = (r_n - r_y) * (pA_y * pB_n - pA_n * pB_y)
    so QQ is violated whenever r_y != r_n and pA != pB appropriately.
    The symmetric case r_y == r_n satisfies QQ identically (see V2 note).
    """

    def __init__(self, p_a, p_b, r_y, r_n):
        self.p_a, self.p_b, self.r_y, self.r_n = p_a, p_b, r_y, r_n

    def _joint(self, p_first, p_second):
        out = {}
        for x, px in (("y", p_first), ("n", 1 - p_first)):
            r = self.r_y if x == "y" else self.r_n
            for y in "yn":
                base = p_second if y == "y" else 1 - p_second
                rep = 1.0 if y == x else 0.0
                out[(x, y)] = px * (r * rep + (1 - r) * base)
        return out

    def joints(self):
        return self._joint(self.p_a, self.p_b), self._joint(self.p_b, self.p_a)


class AsymmetricPOVMAgent(ProjectiveAgent):
    """Unsharp measurement with outcome-asymmetric noise.

    E_y = alpha * P_y + beta * P_n, E_n = (1-alpha) * P_y + (1-beta) * P_n,
    with Kraus K = sqrt(E) (Lueders instrument). Symmetric noise
    (alpha + beta = 1 case in POVMAgent) preserves QQ numerically;
    asymmetric noise is the candidate violator per Lebedev-Khrennikov.
    """

    def __init__(self, psi_angle, a_angle, b_angle, alpha, beta):
        super().__init__(psi_angle, a_angle, b_angle)
        self.ka = self._kraus_pair(self.pa_y, self.pa_n, alpha, beta)
        self.kb = self._kraus_pair(self.pb_y, self.pb_n, alpha, beta)

    @staticmethod
    def _kraus_pair(py, pn, alpha, beta):
        def _sqrt(mat):
            w, v = np.linalg.eigh(mat)
            return v @ np.diag(np.sqrt(np.clip(w, 0, None))) @ v.T
        e_y = alpha * py + beta * pn
        e_n = (1 - alpha) * py + (1 - beta) * pn
        return {"y": _sqrt(e_y), "n": _sqrt(e_n)}

    def _joint_k(self, first, second):
        out = {}
        for x, kx in first.items():
            for y, ky in second.items():
                v = ky @ (kx @ self.psi)
                out[(x, y)] = float(v @ v)
        return out

    def joints(self):
        return self._joint_k(self.ka, self.kb), self._joint_k(self.kb, self.ka)
