"""FROZEN known-answer oracle. DO NOT EDIT. tests/test_ev_reference.py fails if this file and
params/params.yaml disagree; fix params.yaml, never this file.

Implements the truth-blind optimizer EV (docs/00_context.md §3) with every P-hat replaced by its
population parameter. Uplifts are ASSUMED (params.yaml uplift_true). PHI is the assumed share of
disputes against correct fulfillment (params.yaml reference.phi).

This module may hold numeric literals (exempt from the no-magic-numbers rule) because it IS the
parameter cross-check. Nothing under vulcan_proof/opt or models may import it.
"""
import itertools, math, json

HOURLY = 300.0
FEE = 500.0  # enters prevention only; NOT in defence EV (§10)

# ---------- §8.5 category profiles ----------
CAT = {
 "Electronics": dict(pA=0.0040, mix={"NR":0.55,"NAD":0.30,"EB":0.15}, vmin=8000,  vmax=90000),
 "Jewellery":   dict(pA=0.0055, mix={"NR":0.45,"NAD":0.20,"EB":0.35}, vmin=15000, vmax=200000),
 "Apparel":     dict(pA=0.0035, mix={"NR":0.15,"NAD":0.80,"EB":0.05}, vmin=500,   vmax=6000),
 "Home":        dict(pA=0.0030, mix={"NR":0.20,"NAD":0.75,"EB":0.05}, vmin=2000,  vmax=40000),
 "FMCG":        dict(pA=0.0005, mix={"NR":0.30,"NAD":0.65,"EB":0.05}, vmin=200,   vmax=2000),
}

# ---------- §8.6 archetype-weighted population contest rate ----------
ARCH = [(0.15,0.85),(0.15,0.75),(0.15,0.65),(0.15,0.60),(0.30,0.50),(0.10,0.35)]
PC_POP = sum(s*c for s,c in ARCH)          # 0.6125
COMPLIANCE_POP = 0.15*0.90+0.15*0.85+0.15*0.85+0.15*0.80+0.30*0.95+0.10*0.30  # 0.825

# ---------- share of disputes against correct fulfillment (§8.2 output; ASSUMED mid-range) ----------
PHI = 0.65

# ---------- §6 catalogue: cash, seconds, materialisation factor beyond compliance ----------
# mat = extra factor: customer presence for OTP/sig (0.90), ack response rates (§8.8: 0.60 basic, 0.45 verified)
EV_TYPES = {
 "weight":   dict(cash=0,    sec=5,  mat=1.00, adm={"NAD","EB"}),
 "serial":   dict(cash=0,    sec=15, mat=1.00, adm={"NAD","EB"}),
 "sealed":   dict(cash=0,    sec=20, mat=1.00, adm={"NAD","EB"}),
 "packing":  dict(cash=0,    sec=30, mat=1.00, adm={"NAD","EB"}),
 "geotag":   dict(cash=8,    sec=0,  mat=1.00, adm={"NR"}),
 "otp":      dict(cash=25,   sec=0,  mat=0.90, adm={"NR"}),
 "signature":dict(cash=40,   sec=0,  mat=0.90, adm={"NR"}),
 "ack":      dict(cash=0.30, sec=0,  mat=0.60, adm={"NR"}),
 "vack":     dict(cash=0.30, sec=0,  mat=0.45, adm={"NR","NAD","EB"}),
}
# acks are system-sent: compliance = 1. Others: merchant compliance.
SYSTEM_SENT = {"ack","vack"}

# ---------- ASSUMED true uplifts (pp of win prob, on correct-fulfillment disputes) ----------
BASE_WIN = {"NR":0.25,"NAD":0.20,"EB":0.15}
UPLIFT = {
 "NR":  {"otp":0.40,"signature":0.35,"geotag":0.20,"ack":0.16,"vack":0.20},   # ack = 0.4x OTP per §6
 "NAD": {"packing":0.25,"serial":0.15,"sealed":0.10,"weight":0.05,"vack":0.10},
 "EB":  {"weight":0.30,"packing":0.20,"sealed":0.15,"serial":0.10,"vack":0.08},
}
OVERLAP = {"NR":0.55,"NAD":0.35,"EB":0.30}   # §8.7

def set_uplift(d, present):
    """uplift of a materialised set: max + (1-rho)*sum(rest), capped at 1-base."""
    us = sorted([UPLIFT[d][e] for e in present if e in UPLIFT[d]], reverse=True)
    if not us: return 0.0
    u = us[0] + (1-OVERLAP[d])*sum(us[1:])
    return min(u, 1-BASE_WIN[d])

def mat_prob(e, compliance):
    t = EV_TYPES[e]
    return (1.0 if e in SYSTEM_SENT else compliance) * t["mat"]

def expected_uplift(d, S, compliance):
    """E over independent materialisation of each item in S."""
    S = [e for e in S if d in EV_TYPES[e]["adm"]]
    tot = 0.0
    for bits in itertools.product([0,1], repeat=len(S)):
        p = 1.0; present=[]
        for e,b in zip(S,bits):
            m = mat_prob(e, compliance)
            p *= m if b else (1-m)
            if b: present.append(e)
        tot += p*set_uplift(d, present)
    return tot

def ev_set(S, cat, V, risk=1.0, pc=PC_POP, compliance=COMPLIANCE_POP, phi=PHI):
    c = CAT[cat]; pA = c["pA"]*risk
    benefit = 0.0
    for d,pB in c["mix"].items():
        benefit += pA*pB*pc*phi*expected_uplift(d, S, compliance)*V
    cost = 0.0
    for e in S:
        t = EV_TYPES[e]
        cost += t["cash"]*mat_prob(e,compliance) if e not in SYSTEM_SENT else t["cash"]  # cash on compliance (§8.3); acks always sent
        cost += t["sec"]*HOURLY/3600.0                                                  # time on request
    return benefit - cost, benefit, cost

def best_subset(cat, V, **kw):
    names = list(EV_TYPES)
    best=(-1e9,()); 
    for r in range(len(names)+1):
        for S in itertools.combinations(names, r):
            ev,_,_ = ev_set(S,cat,V,**kw)
            if ev > best[0]+1e-9: best=(ev,S)
    return best

def standalone(e, cat, V, **kw):
    return ev_set((e,),cat,V,**kw)

def threshold(e, cat, risk=1.0, pc=PC_POP, compliance=COMPLIANCE_POP, phi=PHI):
    """closed form: EV linear in V."""
    ev0,b0,c0 = ev_set((e,),cat,1.0,risk,pc,compliance,phi)   # benefit per rupee = b0
    return c0/b0 if b0>0 else math.inf

if __name__=="__main__":
    print("PC_POP",PC_POP,"COMPLIANCE_POP",COMPLIANCE_POP)
