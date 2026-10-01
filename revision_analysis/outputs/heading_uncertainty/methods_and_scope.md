# Heading uncertainty analysis (R1.5)

- Position uncertainty uses independent CNC validation data (`cnc_fingerprint_02.csv`, n=528), not radial RMSE.
- Per-axis sample standard deviations are $\sigma_x=0.01506$ m and $\sigma_y=0.01418$ m; the pooled isotropic value is 0.01463 m.
- At the held-out run's median measured step, $d=0.11190$ m, first-order propagation gives $\sqrt{\sigma_x^2+\sigma_y^2}/d=10.59$ degrees. This equals $\sqrt{2}\sigma_{xy}/d$ for the pooled isotropic definition.
- On held-out dataset 10 the empirical circular standard deviation is 13.41 degrees; median/P90 absolute errors are 7.25/20.47 degrees.
- The first-order expression assumes independent position errors. Consecutive fingerprint errors can be spatially correlated, so the propagated value is an approximation rather than a calibrated confidence interval.
- A two-fix heading is undefined at zero displacement and becomes poorly conditioned as displacement approaches zero. It is a motion-derived quantity, not an independent photodiode heading measurement.
