The arithmetic is run, not reasoned. Every number below comes from `vulcan_proof_ev.py`, which implements §5.2 literally against the §6/§8/§10 parameters; you can lift it into Phase 1 as the optimizer's reference implementation and as the test oracle. Two demo beats die at average risk, one is rescued by risk elevation, one is rescued by merchant contest rate, and the stacking refusal exists — but only above ₹45k electronics at 4× risk or ₹200k jewellery at average. The decision-region question has a plain answer at the end.

---

# Vulcan Proof — Pre-Build Arithmetic

## §5.2 EV evaluated against the v9 parameters

**Companion to Final Build Specification v9 · Design locked · Numbers only**

---

## 0. What was computed

For each of 20 representative orders across all five categories, at Stage A risk multiples of 1×, 2×, 4×:

- Standalone EV of every admissible evidence type (§5.2, truth-blind form)
- The optimizer's chosen subset (brute force over all 2⁹ subsets, materialisation-weighted, §8.7 overlap applied)
- Closed-form break-even order value per paid type per category
- Incremental EV of each handoff proof over the rest of the chosen plan (the stacking test)
- Oracle value of per-order optimisation vs a tuned 5-band value rule, as risk dispersion grows

---

## 1. Parameters used, and the one the spec does not supply

Everything below is from v9 except the per-evidence uplifts. §8.7 says true uplifts exist and are hidden; it never states them. The optimizer's arithmetic cannot run without a value, so these are **assumed, stated, and first in the sweep list (§7)**.

### 1.1 From the spec

| Quantity | Value | Source |
|---|---|---|
| Merchant time | ₹300/hr | §10 |
| Dispute fee | ₹500 — **not in defence EV** (sunk on win and loss) | §10 |
| Category dispute rates, type mixes, value ranges | §8.5 table | §8.5 |
| Archetype contest rates, shares | §8.6 table → population contest **0.6125** | §8.6 |
| Archetype compliance → population **0.825** | §8.6 | §8.6 |
| Base win rates | NR 0.25 · NAD 0.20 · EB 0.15 | §8.7 |
| Overlap coefficients | NR 0.55 · NAD 0.35 · EB 0.30 | §8.7 |
| Ack/OTP uplift ratio | 0.4× | §6 |
| Customer response: basic ack 0.60, verified ack 0.45 | §8.8 (verified −15pp) | §8.8 |
| Cash cost paid on compliance; time cost on request | §8.3 | §8.3 |
| Evidence cash, seconds, admissibility | §6 tables | §6 |

### 1.2 Assumed (not in the spec)

| Quantity | Value | Basis |
|---|---|---|
| **φ** — share of disputes arising against correct fulfillment | **0.65** | §8.2 makes this an output; 0.65 is mid-range of what the funnel plausibly produces. Truth-blind Δ = φ × Δ_correct, treating evidence effects on genuine-failure disputes as net zero (negative on merchant fault, positive on transit damage). |
| Customer presence for OTP / signature | 0.90 | §6 "must be present" |
| **Uplifts (pp of win probability, correct-fulfillment disputes)** | | |
| Non-receipt | OTP +40 · signature +35 · geotag +20 · verified ack +20 · basic ack +16 | ack = 0.4 × OTP per §6; signature slightly below OTP (both prove receipt; signature is weaker on identity) |
| Not-as-described | packing +25 · serial +15 · sealed +10 · weight +5 · verified ack +10 | packing photo is the decisive item for contents |
| Empty box | weight +30 · packing +20 · sealed +15 · serial +10 · verified ack +8 | weight is the decisive item for absence |

Set-uplift rule: `max(u) + (1 − ρ_d) × Σ(rest)`, capped at `1 − base_win`. Materialisation is independent per item; the expected uplift is averaged over all 2^k presence patterns.

### 1.3 The equation as evaluated

```
EV̂(E | x) = Σ_d  pA·risk · pB(d) · pC · φ · E[uplift_d(materialised ⊆ E)] · V
            − Σ_{e∈E} cash_e · m_e        (m_e = compliance × presence; acks: 1 × response)
            − Σ_{e∈E} sec_e · 300/3600
```

Because EV is linear in V, the standalone break-even is `V* = cost / (benefit per rupee)`. For **cash** items the compliance term appears in both numerator and denominator and **cancels**: `V*_cash = cash / (pA·pB·pC·φ·Δ)`. For **time** items it does not: `V*_time = sec·(300/3600) / (pA·pB·pC·m·φ·Δ)`. Compliance therefore never moves an OTP threshold and always moves a packing-photo threshold.

---

## 2. Worked arithmetic for the two demo anchors

### 2.1 ₹45,000 phone, OTP, average risk, population contest

```
benefit = 0.0040 × 0.55 × 0.6125 × (0.825×0.90) × 0.65 × 0.40 × 45,000
        = 0.0040 × 0.55            = 0.00220
        × 0.6125                   = 0.0013475
        × 0.7425                   = 0.0010005
        × 0.65                     = 0.00065032
        × 0.40                     = 0.00026013
        × 45,000                   = ₹11.71
cost    = 25 × 0.7425              = ₹18.56
EV      = 11.71 − 18.56            = ₹−6.86        NOT recommended
```

Break-even: `25 / (0.0040 × 0.55 × 0.6125 × 0.65 × 0.40) = 25 / 0.00035033 = ₹71,357`.

### 2.2 ₹3,500 kurta, packing photo, average risk

```
benefit = 0.0035 × [0.80 × 0.25 + 0.05 × 0.20] × 0.6125 × 0.825 × 0.65 × 3,500
        = 0.0035 × 0.21             = 0.000735
        × 0.6125                    = 0.00045019
        × 0.825                     = 0.00037141
        × 0.65                      = 0.00024141
        × 3,500                     = ₹0.845
cost    = 30 s × 300/3600           = ₹2.50
EV      = 0.845 − 2.50              = ₹−1.66        NOT recommended
```

Break-even: `2.50 / 0.00024141 = ₹10,356` — **above the apparel value range (₹500–₹6,000)**. At average risk, no apparel order in the catalogue gets a packing photo.

---

## 3. The 20-order grid — standalone EV (₹) and the optimizer's chosen plan

`−` = inadmissible for the category's dispute types. Plan = brute-force best subset.

### 3.1 Average risk (1×)

| Category | V | weight | serial | sealed | packing | geotag | OTP | signature | ack | v-ack | Plan (EV) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Electronics | 8,000 | 0.21 | −0.62 | −1.11 | −1.40 | −5.44 | −16.48 | −27.88 | 0.37 | 0.57 | weight, ack, v-ack (0.9) |
| Electronics | 15,000 | 0.77 | −0.07 | −0.63 | −0.43 | −4.43 | −14.66 | −26.29 | 0.96 | 1.33 | weight, ack, v-ack (2.7) |
| Electronics | 25,000 | 1.55 | 0.72 | 0.06 | 0.95 | −2.99 | −12.06 | −24.01 | 1.80 | 2.42 | weight, serial, packing, ack, v-ack (5.7) |
| Electronics | 45,000 | 3.13 | 2.30 | 1.44 | 3.71 | −0.10 | −6.86 | −19.46 | 3.48 | 4.60 | all 4 pre-dispatch, ack, v-ack (14.5) |
| Electronics | 60,000 | 4.31 | 3.48 | 2.47 | 5.78 | 2.07 | −2.95 | −16.04 | 4.75 | 6.24 | all 4 pre-dispatch, ack, v-ack (21.5) — geotag standalone +2.07 but refused (overlaps ack) |
| Electronics | 90,000 | 6.68 | 5.84 | 4.54 | 9.92 | 6.41 | 4.85 | −9.21 | 7.27 | 9.50 | + geotag (36.7) — OTP standalone +4.85 but refused (overlaps geotag+ack) |
| Jewellery | 15,000 | 2.70 | 0.51 | 0.30 | 0.75 | −4.16 | −14.17 | −25.86 | 1.12 | 1.74 | weight, packing, ack, v-ack (5.2) |
| Jewellery | 40,000 | 7.89 | 3.45 | 3.57 | 6.17 | −0.10 | −6.86 | −19.46 | 3.48 | 5.14 | all 4 pre-dispatch, ack, v-ack (23.7) |
| Jewellery | 100,000 | 20.36 | 10.49 | 11.43 | 19.18 | 9.66 | 10.70 | −4.09 | 9.16 | 13.30 | + geotag, OTP (75.2) |
| Jewellery | 200,000 | 41.13 | 22.23 | 24.53 | 40.86 | 25.92 | 39.97 | 21.51 | 18.62 | 26.90 | + geotag, OTP (182.0) — **signature standalone +21.51, refused** |
| Apparel | 500 | −0.39 | −1.18 | −1.62 | −2.38 | −6.58 | −18.53 | −29.67 | −0.29 | −0.26 | none |
| Apparel | 1,500 | −0.32 | −1.03 | −1.52 | −2.14 | −6.55 | −18.47 | −29.62 | −0.27 | −0.19 | none |
| Apparel | 3,500 | −0.20 | −0.75 | −1.31 | −1.66 | −6.48 | −18.35 | −29.51 | −0.23 | −0.05 | **none** |
| Apparel | 6,000 | −0.04 | −0.39 | −1.06 | −1.05 | −6.39 | −18.19 | −29.37 | −0.18 | 0.13 | v-ack (0.1) |
| Home | 2,000 | −0.31 | −1.02 | −1.50 | −2.11 | −6.52 | −18.42 | −29.58 | −0.25 | −0.17 | none |
| Home | 10,000 | 0.10 | −0.09 | −0.85 | −0.55 | −6.21 | −17.85 | −29.08 | −0.07 | 0.34 | weight, v-ack (0.4) |
| Home | 40,000 | 1.65 | 3.38 | 1.59 | 5.28 | −5.02 | −15.72 | −27.22 | 0.62 | 2.26 | all 4 pre-dispatch, ack, v-ack (10.9) |
| FMCG | 200 | −0.42 | −1.25 | −1.66 | −2.49 | −6.60 | −18.56 | −29.70 | −0.30 | −0.30 | none |
| FMCG | 800 | −0.41 | −1.24 | −1.66 | −2.48 | −6.59 | −18.55 | −29.69 | −0.30 | −0.29 | none |
| FMCG | 2,000 | −0.40 | −1.22 | −1.64 | −2.44 | −6.58 | −18.53 | −29.67 | −0.29 | −0.28 | none |

### 3.2 Elevated risk (2×)

| Category | V | weight | serial | sealed | packing | geotag | OTP | signature | ack | v-ack | Plan (EV) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Electronics | 8,000 | 0.84 | 0.01 | −0.56 | −0.29 | −4.29 | −14.40 | −26.06 | 1.05 | 1.44 | weight, ack, v-ack (2.9) |
| Electronics | 15,000 | 1.95 | 1.11 | 0.40 | 1.64 | −2.26 | −10.76 | −22.87 | 2.22 | 2.97 | weight, serial, packing, ack, v-ack (7.8) |
| Electronics | 25,000 | 3.52 | 2.69 | 1.78 | 4.40 | 0.63 | −5.56 | −18.32 | 3.90 | 5.15 | all 4 pre-dispatch, ack, v-ack (16.8) |
| Electronics | 45,000 | 6.68 | 5.84 | 4.54 | 9.92 | 6.41 | **4.85** | −9.21 | 7.27 | 9.50 | + geotag (36.7) — **OTP clears standalone but is refused: incremental −1.27 over geotag+ack** |
| Electronics | 60,000 | 9.04 | 8.21 | 6.61 | 14.05 | 10.74 | 12.65 | −2.39 | 9.79 | 12.77 | + geotag, OTP (57.8) |
| Electronics | 90,000 | 13.77 | 12.94 | 10.75 | 22.33 | 19.41 | 28.26 | 11.27 | 14.84 | 19.31 | + geotag, OTP (102.5) — **signature +11.27 standalone, refused** |
| Jewellery | 15,000 | 5.82 | 2.27 | 2.26 | 4.00 | −1.72 | −9.78 | −22.02 | 2.54 | 3.78 | all 4 pre-dispatch, ack, v-ack (16.2) |
| Jewellery | 40,000 | 16.20 | 8.14 | 8.81 | 14.84 | 6.41 | 4.85 | −9.21 | 7.27 | 10.58 | + geotag (55.1) |
| Jewellery | 100,000 | 41.13 | 22.23 | 24.53 | 40.86 | 25.92 | 39.97 | 21.51 | 18.62 | 26.90 | + geotag, OTP (182.0) — signature refused |
| Jewellery | 200,000 | 82.68 | 45.72 | 50.72 | 84.21 | 58.43 | 98.50 | 72.73 | 37.54 | 54.09 | **everything, including signature** (415.7) |
| Apparel | 3,500 | 0.03 | −0.24 | −0.96 | −0.81 | −6.36 | −18.13 | −29.32 | −0.16 | 0.20 | v-ack (0.2) |
| Apparel | 6,000 | 0.34 | 0.47 | −0.46 | 0.40 | −6.19 | −17.82 | −29.05 | −0.06 | 0.56 | weight, serial, v-ack (1.0) |
| Home | 10,000 | 0.62 | 1.07 | −0.04 | 1.39 | −5.81 | −17.14 | −28.46 | 0.16 | 0.98 | weight, serial, packing, ack, v-ack (2.8) |
| Home | 40,000 | 3.72 | 8.01 | 4.84 | 13.07 | −3.45 | −12.89 | −24.73 | 1.53 | 4.82 | all 4 pre-dispatch, ack, v-ack (28.3) |
| FMCG | all | <0 | <0 | <0 | <0 | <0 | <0 | <0 | <0 | <0 | none |

### 3.3 Elevated risk (4×)

| Category | V | weight | serial | sealed | packing | geotag | OTP | signature | ack | v-ack | Plan (EV) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Electronics | 8,000 | 2.11 | 1.27 | 0.54 | 1.91 | −1.98 | −10.24 | −22.42 | 2.39 | 3.19 | weight, serial, packing, ack, v-ack (8.6) |
| Electronics | 15,000 | 4.31 | 3.48 | 2.47 | 5.78 | 2.07 | −2.95 | −16.04 | 4.75 | 6.24 | all 4 pre-dispatch, ack, v-ack (21.5) |
| Electronics | 25,000 | 7.47 | 6.63 | 5.23 | 11.30 | 7.85 | 7.45 | −6.94 | 8.11 | 10.59 | + geotag, OTP (42.9) |
| Electronics | 45,000 | 13.77 | 12.94 | 10.75 | 22.33 | 19.41 | 28.26 | **11.27** | 14.84 | 19.31 | + geotag, OTP (102.5) — **signature clears standalone (+11.27), refused: incremental −9.76** |
| Electronics | 90,000 | 27.96 | 27.13 | 23.16 | 47.16 | 45.43 | 75.09 | 52.24 | 29.97 | 38.91 | everything (246.8) |
| Jewellery | 15,000 | 12.05 | 5.80 | 6.19 | 10.51 | 3.16 | −1.00 | −14.34 | 5.38 | 7.86 | all 4 pre-dispatch, ack, v-ack (38.7) |
| Jewellery | 40,000 | 32.82 | 17.54 | 19.29 | 32.18 | 19.41 | 28.26 | 11.27 | 14.84 | 21.46 | + geotag, OTP (139.3) — signature refused |
| Apparel | 3,500 | 0.47 | 0.76 | −0.26 | 0.88 | −6.12 | −17.69 | −28.94 | −0.02 | 0.70 | **weight, serial, packing, v-ack (1.7)** |
| Apparel | 6,000 | 1.10 | 2.20 | 0.75 | 3.29 | −5.77 | −17.07 | −28.40 | 0.18 | 1.42 | weight, serial, packing, ack, v-ack (6.2) |
| Home | 40,000 | 7.86 | 17.27 | 11.34 | 28.64 | −0.29 | −7.21 | −19.77 | 3.37 | 9.93 | all 4 pre-dispatch, ack, v-ack (63.0) |
| FMCG | all | | | | | | | | | | none |

---

## 4. Recommendation boundaries — standalone break-even order value (₹)

Closed-form `V* = cost / benefit-per-rupee`, population contest 0.6125, φ 0.65.

| Category (range) | Type | 1× | 2× | 4× |
|---|---|---:|---:|---:|
| **Electronics** (8k–90k) | geotag | 45,669 | 22,834 | 11,417 |
| | OTP | 71,357 | 35,679 | 17,839 |
| | signature | 130,482 | 65,241 | 32,620 |
| | packing photo | 18,122 | 9,061 | 4,531 |
| | serial | 15,857 | 7,929 | 3,964 |
| | weight | 5,286 | 2,643 | 1,321 |
| | sealed photo | 24,163 | 12,082 | 6,041 |
| **Jewellery** (15k–200k) | geotag | 40,594 | 20,297 | 10,149 |
| | OTP | 63,429 | 31,714 | 15,857 |
| | signature | 115,984 | 57,992 | 28,996 |
| | packing photo | 11,532 | 5,766 | 2,883 |
| | weight | 2,006 | 1,003 | 501 |
| **Apparel** (500–6k) | OTP | 299,021 | 149,510 | 74,755 |
| | geotag | 191,373 | 95,687 | 47,843 |
| | packing photo | 10,356 | 5,178 | 2,589 |
| | serial | 8,699 | 4,349 | 2,175 |
| | weight | 6,590 | 3,295 | 1,647 |
| **Home** (2k–40k) | OTP | 261,643 | 130,822 | 65,411 |
| | packing photo | 12,846 | 6,423 | 3,212 |
| | serial | 10,796 | 5,398 | 2,699 |
| | weight | 8,054 | 4,027 | 2,014 |
| **FMCG** (200–2k) | packing photo | 88,249 | 44,124 | 22,062 |
| | weight | 53,414 | 26,707 | 13,353 |

**Reading this table:**

- **Paid handoff evidence is an electronics-and-jewellery product.** Every apparel, home, and FMCG handoff threshold sits 8–500× above the category's maximum value, at 4× risk. No amount of Stage A elevation the spec contemplates buys OTP for a kurta.
- **Signature never clears standalone inside any category range at average risk** (₹130k electronics vs ₹90k max; ₹116k jewellery reaches only at the top of the range). It clears at 2× above ₹65k electronics / ₹58k jewellery — and is still refused by the optimizer there because of overlap (§5).
- **Free pre-dispatch evidence has real thresholds.** At ₹300/hr a 30-second packing photo needs ₹18k electronics or ₹10k apparel at average risk. "Cheap evidence is always worth it on expensive orders" is true; "cheap evidence is always worth it" is false, and apparel's whole range is below its own packing threshold.
- **FMCG never gets anything.** Coverage there is prediction-only regardless of tier.

**OTP break-even by merchant contest rate (electronics):**

| Contest rate | 1× | 2× |
|---|---:|---:|
| 0.35 (Erratic) | 124,875 | 62,438 |
| 0.50 (Minimal) | 87,413 | 43,706 |
| 0.6125 (population) | 71,357 | 35,679 |
| 0.75 (Handoff-heavy) | 58,275 | 29,138 |
| 0.85 (Diligent) | 51,419 | 25,710 |

Contest rate moves the OTP boundary by **2.4×** across the archetype range — more than a 2× Stage A elevation.

---

## 5. Demo scenarios: which survive

### 5.1 The ₹3,500 kurta — dead at average risk, alive at ~3×

Standalone packing photo EV is **₹−1.66** at average risk. The optimizer recommends **nothing** for a ₹3,500 apparel order at 1× and only a verified ack at 2×. The packing photo first clears at **2.96× Stage A**; at 4× the plan is weight + serial + packing + verified ack (₹1.7 total EV).

Apparel value at which pre-dispatch evidence first clears at average risk: **weight ₹6,590 · serial ₹8,699 · packing ₹10,356** — all above the ₹6,000 category ceiling. The old "kurta gets a packing photo, OTP refused" beat was two truths and one falsehood: OTP is refused (₹−18), the inversion in *admissibility* is real, but the photo is not bought unless the order is flagged.

**What survives:** the inversion beat works as a *risk-flagged* kurta (repeat address mismatch, new customer, history of not-as-described claims → Stage A ≈ 3–4×). The story is then: "same price, different customer, the photo is worth 88 paise net; OTP would lose ₹18." That is a stronger beat than the original because it shows Stage A doing work on a cheap order.

### 5.2 The ₹45,000 phone — no OTP at average risk; needs 1.6× at population contest, 1.14× for a Diligent merchant

| Contest rate | Risk multiple at which OTP standalone clears |
|---|---:|
| 0.35 | 2.78× |
| 0.6125 | 1.59× |
| 0.85 | 1.14× |

At 1×, the plan is all four pre-dispatch items + basic ack + verified ack (₹14.5 EV); geotag is ₹−0.10 and OTP ₹−6.86.

**But at 2× the optimizer still does not buy OTP.** OTP standalone is +₹4.85, yet its incremental value over geotag + basic ack is **₹−1.27** — geotag (₹8, +20pp) plus a 60%-responding ack (₹0.30, +16pp) already cover most of the non-receipt uplift under ρ = 0.55, and the third overlapping proof adds 45% of its standalone uplift for 100% of its cost. OTP enters the plan at ₹45k only at ≈ **2.5×**, or at 2× for a Diligent merchant (contest 0.85 → OTP standalone +₹13.93, incremental positive).

**What survives:** the phone beat works at **2× risk and a Diligent merchant**, or at **4× risk and population contest** (plan: all pre-dispatch + geotag + OTP + ack, ₹102.5 EV). Average risk shows the *pre-dispatch and acknowledgement* plan, which is itself the day-three point: ₹14.5 of expected value from free-and-30-paise items that Dispute Responder currently never receives.

### 5.3 The stacking refusal — exists, at three grid points

"Optimizer buys one overlapping handoff proof and refuses another that would clear on its own":

| Order | Refused item | Standalone EV | Incremental over rest of plan |
|---|---|---:|---:|
| Electronics ₹45k, **2×** | OTP | +4.85 | **−1.27** (geotag + ack chosen) |
| Electronics ₹45k, **4×** | signature | +11.27 | **−9.76** (geotag + OTP + ack chosen) |
| Electronics ₹90k, 2× | signature | +11.27 | −9.76 |
| Jewellery ₹200k, **1×** | signature | +21.51 | **−4.78** (geotag + OTP + ack chosen) |

The beat exists and the cleanest instance is **Jewellery ₹200k at average risk**: signature would return ₹21.51 on its own, the optimizer holds OTP and geotag and declines it, and the reason is a learned overlap, not a rule. The ₹45k electronics at 4× is the same story on the demo's own order. At 2× jewellery ₹200k the optimizer buys all four handoff proofs — the ceiling `1 − base` kicks in and overlap stops binding — so the beat is bounded above as well as below.

**Caveat that must be on the slide:** this refusal is only learnable if the (OTP, signature) bitmask has ≥50 contested-dispute support, which v9's Handoff-heavy archetype (signature above ₹50k) and the 5% random stratum exist to provide. Check the support diagnostic before scripting the beat.

### 5.4 The contest-rate case — survives cleanly

₹45k electronics, 2× risk:

| Merchant | Plan | Plan EV | OTP standalone |
|---|---|---:|---:|
| Contest 0.85 | all 4 pre-dispatch + geotag + **OTP** + ack + v-ack | ₹61.5 | +13.93 |
| Contest 0.35 | all 4 pre-dispatch + ack + v-ack (no handoff evidence at all) | ₹17.5 | −5.18 |

Identical order, identical risk, two plans. Geotag flips too (+11.45 vs +0.83, refused at 0.35 in combination). This is the strongest inversion in the grid and it is driven by a merchant-level feature a category × value rule cannot see.

---

## 6. Is the paid-evidence decision region fat enough for an optimizer?

Oracle experiment: 300 orders per category, values log-uniform over the category range, Stage A risk drawn lognormal with mean 1 and dispersion σ (σ = 0.7 puts ~10% of orders at ≥ 2.4× and ~10% at ≤ 0.4×). **Arm 5 ceiling** = per-order best subset with *oracle* knowledge of true risk. **Arm 4** = best fixed subset per 5-band value rule, tuned with oracle knowledge of realised EV. Both are upper bounds; the gap is the most the ML could ever add.

| Category | σ | Arm 5 ceiling ₹/1,000 | Arm 4 ₹/1,000 | Gap | Gap as % of ceiling |
|---|---:|---:|---:|---:|---:|
| Electronics | 0.00 | 10,183 | 10,131 | 52 | 0.5% |
| | 0.35 | 9,547 | 9,150 | 397 | 4.2% |
| | 0.70 | 11,656 | 10,246 | 1,411 | 12.1% |
| | 1.00 | 14,029 | 10,593 | 3,436 | 24.5% |
| Jewellery | 0.00 | 52,392 | 52,277 | 116 | 0.2% |
| | 0.35 | 54,740 | 53,758 | 981 | 1.8% |
| | 0.70 | 54,756 | 50,932 | 3,824 | 7.0% |
| | 1.00 | 66,024 | 58,769 | 7,255 | 11.0% |
| Apparel | 0.70 | 65 | 14 | 51 | 79% (of ₹65) |
| Home | 0.70 | 1,949 | 1,518 | 431 | 22.1% |

Adding merchant contest-rate variation (rule blind to it, oracle sees it) at σ = 0.7 moves the electronics gap from ₹1,411 to ₹1,402 and jewellery from ₹3,824 to ₹3,636 — contest rate changes *which* handoff proof, not much *whether*, at the population level.

**Plain answer.**

1. **At σ = 0 the kill condition fires, as it should.** A tuned 5-band value rule captures 99.5–99.8% of the achievable ₹ in the two categories where the ₹ is. This is World A conceding its tie, for free, before any code.

2. **The orchestration layer is the product; the optimizer is a 5–12% improvement on it at plausible dispersion.** At σ = 0.7 the oracle gap is 7–12% of the ceiling in electronics and jewellery. A real Stage A recovers a fraction of an oracle — a third would be respectable — so the realistic ML contribution is **₹400–1,300 per 1,000 electronics/jewellery orders on top of an orchestration layer worth ₹10,000–50,000 per 1,000.** In apparel the ML is most of the value but the value is ₹65 per 1,000 orders.

3. **Reframe the headline now, not after Phase 4.** "Vulcan Proof beats tuned rules" is not the claim the arithmetic supports. The supportable claim is: *the orchestration layer is worth ₹X per 1,000 orders at zero ML; per-order optimisation adds Y% on top when individual-risk signal exists, and κ\* is where Y crosses zero.* That is a smaller ML claim and a much harder one to puncture. A judge who reads "tuned rules capture 99.5% at κ = 0" on your own slide will not need to ask the question.

4. **The optimizer's value is concentrated, and you know where.** ~29% of electronics orders (log-uniform) sit in the ₹35k–₹71k band where a 2× elevation flips OTP; ~27% of jewellery sits in ₹32k–₹63k. Outside those bands, and outside those categories, the optimizer is deciding whether to spend 5–30 seconds of merchant time and 30 paise — decisions worth rupees, not hundreds. Report the band share and the ₹ inside it as a diagnostic; that is the fat part of the decision region and it is not very fat.

---

## 7. Sensitivity — what to sweep first

Every parameter in `V* = cost / (pA · pB · pC · φ · Δ · [m])` has **elasticity exactly 1** on the break-even. So the sweep priority is purely a question of **how wide each parameter's plausible range is**:

| Parameter | Plausible range | Threshold multiplier | Sweep priority | Note |
|---|---|---:|---|---|
| **Uplift Δ (per type)** | ±50% around the assumed values (OTP 0.20–0.60) | 0.67–2.0× | **1** | Not in the spec at all; OTP threshold ₹48k–₹143k across the range. Anchor before anything else. |
| **Merchant time rate** | ₹100–₹600/hr | 0.33–2.0× on *free* evidence only | **2** | At ₹100/hr the kurta clears (+₹0.01); at ₹600/hr packing needs ₹20.7k apparel. Every free-evidence boundary is a function of this one number. |
| **Contest rate pC** | 0.35–0.85 | 0.72–1.75× | **3** | Merchant-level; observed, not assumed. Drives the strongest demo inversion. |
| **Dispute-type mix (Stage B)** | NR share 0.30–0.90 in electronics | 0.61–1.83× | **4** | OTP threshold ₹131k → ₹44k. This is how Stage B earns its place. |
| **φ (share vs correct fulfillment)** | 0.50–0.90 | 0.72–1.30× | 5 | Derived by §8.2; report the realised value. |
| **Category dispute rate pA** | ±50% | 0.67–2.0× | 6 | Same lever as risk elevation; the κ dial already sweeps it. |
| **Overlap ρ_NR** | 0.2–0.7 | — | 7 | Does not move standalone thresholds; **moves every stacking decision.** The refusal beats in §5.3 exist at ρ = 0.55 and vanish below ~0.35. |
| Compliance | 0.30–0.95 | none on cash items; 1.0–3.2× on time items | 8 | Cancels for OTP/signature/geotag. |
| Dispute fee | ₹200–₹1,500 | none | — | Not in defence EV. Sweep in prevention only. |

Sweep Δ, hourly rate, and ρ first. Contest rate and type mix are learned, not swept — their range is a fact about the population.

---

## 8. Demo — rewritten with the numbers in place (§16 replacement)

All figures: population contest 0.6125 and φ 0.65 unless stated; risk multiples are Stage A relative to category average. Slides show the number.

**0:00–0:45** The ₹45,000 phone. Clean payment, correct approval, delivered. Day 62 non-receipt claim. Dispute Responder finds `DELIVERED` and nothing else. Merchant loses ₹45,000 and the goods. *The evidence that would have won this had to be created on day three.*

**0:45–1:40 — The same order, average risk.** Plan: parcel weight (+₹3.13), serial (+₹2.30), sealed photo (+₹1.44), packing photo (+₹3.71), basic acknowledgement (+₹3.48), verified acknowledgement (+₹4.60). Total ₹14.5 expected value, zero cash, 70 seconds of merchant time and one customer tap. OTP shown **refused at ₹−6.86**: *at this merchant's contest rate and this category's base risk, ₹25 of carrier OTP buys ₹11.71 of expected recovery.* Break-even for OTP on this merchant is ₹71k. *Most of the day-three value is free. The question is who is telling the merchant to collect it.*

**1:40–2:20 — The same order, flagged.** Stage A at 4× — new customer, address mismatch, prior not-as-described history. Plan adds geotagged delivery (+₹19.41) and OTP (+₹28.26); total ₹102.5. Signature shown **refused: standalone +₹11.27, incremental −₹9.76** — *OTP and signature both prove receipt; the second overlapping proof returns 45% of its uplift for 100% of its cost. The optimizer priced that overlap from data; nobody typed it.* (Support check on the OTP×signature bitmask shown in the corner.)

**2:20–2:50 — The inversion: ₹3,500 kurta, flagged.** Same Stage A elevation (3–4×). Plan: weight, serial, packing photo (+₹0.88), verified ack — ₹1.7 total. OTP refused at ₹−17.69; geotag refused at ₹−6.12. *At average risk this order gets nothing — apparel's packing-photo break-even is ₹10.4k at ₹300/hr, above the whole category. Stage A is what makes a 30-second photo worth taking on a ₹3,500 order.*

**2:50–3:20 — The contest-rate case.** Two ₹45k electronics orders, both 2× risk. Merchant A contests 85%: all pre-dispatch + geotag + **OTP** + acks, ₹61.5. Merchant B contests 35%: pre-dispatch + acks, **no handoff evidence**, ₹17.5. *Same phone, same risk. Evidence only pays if you fight. The system tells Merchant B one thing only: on the disputes it does contest, it now holds admissible evidence.*

**3:20–3:45 — Defense-only slide.** Three bars: claims against correct fulfillment ↑; against merchant fault flat (packing photo shows the wrong item — negative uplift); against carrier fault reported, with the wrong-recipient-OTP sweep shown at 0 and +partial.

**3:45–4:15 — Day 62 again.** Dispute lands. Package pre-assembled and mapped to Razorpay's contest-API slots: OTP → `proof_of_service`, geotag → `shipping_proof`, serial → `billing_proof`, acknowledgement → `customer_communication`. Merchant wins.

**4:15–4:50 — The honest chart.** Olist held-out PR curve and reliability diagram (real orders). Then the κ-sweep: *at κ = 0 a tuned value rule captures 99.5% of achievable ₹ — that is the tie we pre-registered. At realistic dispersion the optimizer adds 5–12% on top of an orchestration layer worth ₹10k–50k per 1,000 orders in electronics and jewellery. κ\* is marked. Whether Razorpay's data sits above it is the question we cannot answer from outside.*

**4:50–5:00** *Razorpay fights disputes after they happen. Vulcan Proof makes sure the evidence exists before they happen — and is honest about how much of that a rule could do.*

---

## 9. Build notes that fall out of the arithmetic

- **The optimizer must evaluate subsets, not items.** Three of the four demo refusals are incremental, not standalone. A per-item threshold optimizer would buy OTP at ₹45k/2× and be wrong by ₹1.27 per order; brute force over 512 subsets per order is ~microseconds with cached uplifts — do that.
- **Compliance cancels for cash items.** Do not "correct" OTP thresholds for merchant compliance; it drops out. It does not drop out for time items.
- **Free evidence is not free.** At ₹300/hr, four pre-dispatch items cost ₹5.83 per order. Below ~₹5k electronics only weight clears. Coverage in the low-value half of every category will be acks only, and FMCG will be zero. Report this before the judge computes it.
- **Verified ack is the most-recommended item in the grid** (admissible for all three types, 30 paise, 45% response). It is on 15 of 20 plans at average risk. If its true uplift is lower than assumed, every plan in the grid shrinks. Anchor its uplift alongside OTP's.
- **Set ρ_NR = 0.55 as the canonical value and show the refusal beat disappearing below 0.35 in the sweep.** The beat is a function of one hidden parameter; say so on the slide rather than letting a judge ask.

`vulcan_proof_ev.py` reproduces every table above; `best_subset`, `threshold`, and `ev_set` are the three functions Phase 3 needs.

---
*Appendix note for implementers: this is the pre-build arithmetic report, kept verbatim as the
provenance of `params.yaml: reference.known_answers` and of the findings in `00_context.md` §4.
It refers to "§16" and "v9" of a design spec that is not in this repo; the repo docs supersede it
wherever they differ. The script it references is `vulcan_proof/ev_reference.py`.*
