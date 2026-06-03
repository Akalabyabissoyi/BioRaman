# BioRaman validation

This folder validates BioRaman's **quantitative** outputs — chiefly the
component-analysis concentration estimates — so the numbers can be trusted (and
cited) rather than just *looking* right.

## 1. Synthetic phantom (automated, no data required)

Builds a map with **known** component concentrations from synthetic reference
spectra, runs the exact `component_fit()` used by the GUI, and checks the
recovered concentrations against the truth. It exits non-zero if the error
exceeds the tolerance, so it can gate a release / run in CI.

```bash
python validation/validate_concentration.py            # default 1% tolerance
python validation/validate_concentration.py --tol 0.5  # stricter
```

Expected: DCLS recovers the known concentrations to < 0.1 %; NNLS to within the
noise level; per-pixel dominant-component classification > 98 %.

## 2. Real-data cross-check against Renishaw WiRE

Use this to confirm BioRaman reproduces WiRE's **Concentration Estimate**
numbers on the same files. The Renishaw "Advanced Data Analysis 1" training set
(Tablet.wdf with aspirin/caffeine/paracetamol references) is ideal.

1. In **WiRE**, run Component analysis (NNLS, Spectrum, background polynomial
   order 4, **No normalisation**) with the three references, then
   `Procedures → Concentration estimates`, and record the % per component.
2. Run BioRaman on the same files:

   ```bash
   python validation/validate_concentration.py \
       --map Tablet.wdf \
       --refs aspirin.wdf caffeine.wdf paracetamol.wdf \
       --method NNLS --background 4 --normalise "None (quantitative)"
   ```

3. Compare the printed concentration estimates to WiRE's. They should agree to
   within a few percent (small differences are expected from baseline-handling
   details). Record the comparison in the table below.

### Validation record

| Sample | Component | WiRE % | BioRaman % | Δ |
|--------|-----------|--------|------------|---|
| Tablet | aspirin |  |  |  |
| Tablet | caffeine |  |  |  |
| Tablet | paracetamol |  |  |  |

> Tip: load the map in BioRaman with **Preprocessing → Normalisation = none**
> and choose **None (quantitative)** in Component Analysis, matching WiRE's
> "No normalisation" so the intensity scales are comparable.

## Notes / limitations

- Concentration estimates are **relative** (each component's summed abundance ÷
  total), as defined by both WiRE and BioRaman.
- The phantom validates the math; the WiRE cross-check validates the end-to-end
  workflow on real instrument data. Do both before reporting quantitative
  results in a publication.
