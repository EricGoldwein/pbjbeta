# 🧾 PBJ Quarterly Reporting Checklist (LTCCC)

**Version: Q2 2024 — includes CMI-based expectations**

---

## 📥 1. Load & Prep Raw Data
- [ ] Load `PBJ_dailynursestaffing_CY2024Q3.csv`
- [ ] Load `PBJ_dailyNonnurseStaffing_CY2024Q3.csv`
- [ ] Load `NH_ProviderInfo_Mar2025.csv`
- [ ] Load `Regional Coding.csv` and merge on `STATE`
- [ ] Filter for valid facilities/days (CMS rules)
- [ ] Check for consistent `PROVNUM`, remove nulls

---

## 🧮 2. Core Staffing Calculations
- [ ] Compute total and care-level HPRD and hours:
  - **Total Nurse Staff** = RN, RN Admin, RN DON, LPN, LPN Admin, CNA, NA TR, MedAide Tech
  - **Total Nurse Care Staff** = RN, LPN, CNA, NA TR, MedAide Tech
  - **Total RN** = RN, RN Admin, RN DON
  - **RN Care** = RN only
  - **Total Nurse Aide** = CNA, NA TR, MedAide

- [ ] Include employee + contract hours
- [ ] Weekend staffing metrics (Sat/Sun HPRD, % of total)
- [ ] Apply CMS exclusions:
  - `nurse_HPRD == 0`  
  - `nurse_HPRD > 12`  
  - `aide_HPRD > 5.25`  
  - `weekend_nurse_HPRD == 0`  
  - `weekend_nurse_HPRD > 12`  
  - `weekend_aide_HPRD > 5.25`

---

## 📈 3. Case-Mix Index (CMI) Expectations

Use `Nursing_CMI` from Provider Info:

- RN Expected HPRD  
  `0.55 + ((CMI - 0.62) / (3.84 - 0.62))^0.973947642 * (2.39 - 0.55)`
- CNA Expected HPRD  
  `2.45 + ((CMI - 0.62) / (3.84 - 0.62))^0.236050268 * (3.6 - 2.45)`
- Total Expected HPRD  
  `3.48 + ((CMI - 0.62) / (3.84 - 0.62))^0.715361977 * (7.68 - 3.48)`

Add expected vs actual gap columns.

---

## 📊 4. File Outputs
- [ ] **Nurse Staffing** spreadsheet
- [ ] **Non-Nurse Staffing** spreadsheet
- [ ] **Summary** (CMI, HPRD, contract %, flags)
- [ ] **Provider Info** repackaged
- [ ] **State Ranking CSV**
  - CMS Region as string
  - Sort as 1–10 in Excel slicer

---

## 🗺️ 5. Maps & Alerts
- [ ] Aggregate regional/state means
- [ ] Alert copy + data for [staffing alert](https://nursinghome411.org/alert-staffing-q2-2024/)
- [ ] Tableau map-ready CSV

---

## 🧪 6. QA & Final Checks
- [ ] Decimal formatting: HPRD (2 decimals), % (1 decimal)
- [ ] PROVNUM spot-check
- [ ] Slicers: CMS Region ordered 1–10
- [ ] Freeze panes, filters, sort
