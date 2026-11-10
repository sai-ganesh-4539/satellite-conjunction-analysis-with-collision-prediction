# Dataset Retrieval & Schema Specification

This document details the source data, retrieval procedures, verification checksums, and schema definitions for the Satellite Conjunction Assessment and Collision Prediction research pipeline.

---

## 1. Data Sources & Provenance

The benchmark dataset combines two complementary Space Situational Awareness (SSA) feeds:

1. **Conjunction Data Messages (CDMs):**
   - **Source:** USSPACECOM / Space-Track.org public CDM catalog & ESA Kelvins Conjunction Data Repository.
   - **Standard:** CCSDS 508.0-B-1 recommended standard for Conjunction Data Messages.
   - **Scope:** 836 real satellite-satellite and satellite-debris conjunction tracks (3,364 sequential CDM updates).
   - **Local Path:** `data/cdms.csv`
   - **MD5 Checksum:** `18079794cc1403b46d014ee5913a3fd1`
   - **Uncompressed Size:** 382.2 KB

2. **Satellite Catalog (SATCAT):**
   - **Source:** Space-Track.org Satellite Catalog (`SATCAT.csv`).
   - **Scope:** Comprehensive metadata catalog of tracked orbital objects, including NORAD catalog ID, object name, international designator, object type (payload, rocket body, debris), launch date, operational status, apogee, perigee, inclination, period, and radar cross-section (RCS).
   - **Local Path:** `data/satcat.csv`
   - **MD5 Checksum:** `77b99affa6ad3aff73990913653d2020`
   - **Uncompressed Size:** 9.98 MB (compressed in bundle: ~1.9 MB)

---

## 2. Automated Retrieval Procedure

For users retrieving fresh data directly from upstream sources:

### Space-Track REST API
1. Create a free account at [Space-Track.org](https://www.space-track.org).
2. Set your environment variables:
   ```bash
   export SPACETRACK_USER="your_email@domain.com"
   export SPACETRACK_PASSWORD="your_password"
   ```
3. Execute the automated retrieval script:
   ```bash
   python download_cdms.py
   ```
   This script authenticates with Space-Track, queries the `/basicspacedata/query/class/cdm_public` endpoint, and downloads the active CDM tracks and latest SATCAT metadata.

---

## 3. Data Schema & Feature Engineering

Each conjunction encounter is indexed by primary satellite (`SAT_1`) and secondary object (`SAT_2`). Raw CDMs are ordered chronologically by `CDM_ID` leading up to the Time of Closest Approach (TCA).

The processing engine (`src/data_processing.py`) constructs a 14-dimensional feature vector $\mathbf{x}_t \in \mathbb{R}^{14}$ for each received CDM:

| Index | Feature Name | Description | Source |
| :---: | :--- | :--- | :--- |
| **0** | `MISS_DISTANCE` | Total relative scalar miss distance at TCA ($m$) | CDM |
| **1** | `RELATIVE_SPEED` | Relative velocity between primary and secondary ($m/s$) | CDM |
| **2** | `RELATIVE_POSITION_R` | Radial relative displacement in RTN frame ($m$) | CDM |
| **3** | `RELATIVE_POSITION_T` | Transverse (along-track) relative displacement ($m$) | CDM |
| **4** | `RELATIVE_POSITION_N` | Normal (cross-track) relative displacement ($m$) | CDM |
| **5** | `RELATIVE_VELOCITY_R` | Relative velocity in radial axis ($m/s$) | CDM |
| **6** | `RELATIVE_VELOCITY_T` | Relative velocity in transverse axis ($m/s$) | CDM |
| **7** | `RELATIVE_VELOCITY_N` | Relative velocity in normal axis ($m/s$) | CDM |
| **8** | `SAT1_OBJECT_TYPE` | Primary object classification (payload / debris / rocket body) | SATCAT |
| **9** | `SAT2_OBJECT_TYPE` | Secondary object classification (payload / debris / rocket body) | SATCAT |
| **10** | `SAT1_RCS_VALUE` | Primary radar cross section size category ($m^2$) | SATCAT |
| **11** | `SAT2_RCS_VALUE` | Secondary radar cross section size category ($m^2$) | SATCAT |
| **12** | `SAT1_EXCL_VOL` | Primary operational screening exclusion radius ($m$) | SATCAT |
| **13** | `SAT2_EXCL_VOL` | Secondary operational screening exclusion radius ($m$) | SATCAT |

### Missing Value Handling & Imputation
- Approximately 10.5% of secondary objects in this archive are small debris fragments lacking explicit SATCAT dimension records.
- Missing physical dimensions and exclusion volumes are imputed using median values across the corresponding object type.
- A binary indicator feature (`satcat_missing`) is tracked to ensure full auditability.

---

## 4. Ground-Truth Target Labels

- **Triage Safety Class:** Binary target $y \in \{0, 1\}$ defined by whether the peak collision probability reaches the critical mitigation threshold:
  $$y = \mathbb{I}(\max_{t} P_c(t) \ge 5.0 \times 10^{-4})$$
  - Negative class (Safe / Nominal): 626 events (74.9%)
  - Positive class (Critical Risk): 210 events (25.1%)
- **Multi-Horizon Targets:**
  - `final_miss_m`: Observed scalar miss distance at final CDM ($m$).
  - `final_pc`: Catalog collision probability recorded at final CDM.

---

## 5. Offline Reproducibility

Both `data/cdms.csv` and `data/satcat.csv` are packaged directly within `results/SUBMISSION_BUNDLE.zip`. Extracting the bundle allows running the entire test suite, benchmarks, and figures immediately with zero external network downloads.
