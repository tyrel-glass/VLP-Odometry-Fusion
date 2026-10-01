# WP8 primary-source audit: robotic visible-light positioning

Audit date: 1 October 2026. Values below were checked against the primary
publication, not against review-paper summaries. Page numbers refer to the
pagination printed in each PDF. Accuracy values retain the source definition and
must not be treated as a common benchmark.

| Study | Pages checked | Receiver / fusion | State and operation | Reference and scale | Accuracy entered in manuscript | Qualification |
|---|---:|---|---|---|---|---|
| Konings et al. (2017), *HVLP: Hybrid visible light positioning of a mobile robot*, DOI 10.1109/WPNC.2017.8250077 | 1–6; setup/results pp. 4–5 | Photodiode VLP + Zigbee | Position; static/offline test points | Surveyed test locations; rooms 4.8×5.7×2.5 m and 4.8×3.3×2.5 m | Room-1 median 0.058 m (mean 0.059 m, RMSE 0.065 m) | Not a closed-loop trajectory test; hybrid radio supplies coarse room/location information. |
| Hsueh et al. (2018), *Robot Indoor Navigation Using Visible Light Communication*, DOI 10.1109/APSIPA.2018.8659707 | 1–6; system pp. 2–4, results pp. 4–5 | Si PIN photodiode receives lamp commands; wheel encoders/PWM | Lamp waypoint and odometry pose; live route execution | No independent trajectory ground truth; reported lamp–receiver range up to about 300 cm | Qualitative route completion; no numerical localization error | This is a VLC-command navigation demonstration, not an independently referenced localization benchmark. |
| Amsters et al. (2019), *In-Depth Analysis of Unmodulated Visible Light Positioning Using the Iterated Extended Kalman Filter*, DOI 10.3390/s19235198 | 1–24; setup pp. 7–9, Table 6 p. 18 | One or five photodiodes + wheel odometry, IEKF | Planar pose; post-processed recorded motion | Marvelmind ultrasound, nominal 0.02 m accuracy; laboratory paths at about 0.2 m/s | Fused mean error 0.298–0.406 m; P95 0.363–0.502 m | Explicitly offline; values depend on path and number of receivers. |
| Guan et al. (2021), *Robot Localization and Navigation Using Visible Light Positioning and SLAM Fusion*, DOI 10.1109/TIM.2021.3081939 | 1–12; setup/results pp. 7–10 | Rolling-shutter camera + 2-D LiDAR + wheel odometry, EKF | Planar pose; ROS navigation demonstrated | Surveyed static points; about 8.4×4.9 m, four LEDs at 2.2 m | Static fusion mean 0.0248 m, maximum 0.0573 m | The approximately 17.6 m dynamic path had no motion-capture reference, so dynamic localization accuracy was not reported. |
| Guan et al. (2022), *Robust Robotic Localization Using Visible Light Positioning and Inertial Fusion*, DOI 10.1109/JSEN.2022.3143451 | 1–11; setup/results pp. 6–9 | Rolling-shutter camera + MPU9250 IMU + wheel odometry, EKF | Planar pose; dynamic localization | Static points in about 7.0×3.8×2.7 m setup; no independent dynamic ground truth | Static fusion mean 0.0214 m, maximum 0.0501 m | The moving experiment used the camera VLP result as a baseline; it is not an independently referenced closed-loop navigation accuracy result. |
| Chew et al. (2024), *Visible Light Positioning-Based Robot Localization and Navigation*, DOI 10.3390/electronics13020368 | 1–16; setup pp. 5–8, results pp. 10–14 | Photodiode-only VLP position; CNC orientation used by controller | Position from VLP; live CNC-platform navigation | CNC position reference, 0.1 mm resolution; LEDs over 0.76×0.78 m footprint at 1.05 m height | Median path deviation 0.00586–0.01681 m | Orientation was supplied by the CNC platform, not estimated optically. Metric is deviation from paths, not the same endpoint/pose error used here. |
| Sun et al. (2025), *Tightly Coupled VLP/INS Integrated Navigation by Inclination Estimation and Blockage Handling*, DOI 10.1186/s43020-025-00158-9 | 1–18; setup pp. 9–11, Table 4 p. 15, large-area result pp. 15–16 | Single OPT101 photodiode + IMU, tightly coupled graph optimization | 3-D position and attitude; dynamic localization | Robot API reference in 3.8×6.3×2.8 m site; LiDAR–IMU FAST-LIO2 reference in 6.0×8.4×2.8 m site | Mean 0.096 m across tilt/blockage tests; 0.115 m large-area test | Initial heading came from a phone compass because one-PD RSS did not observe heading. |
| Present work | Manuscript Sections II–VII | Photodiode fingerprint VLP + wheel odometry, LSTM; offline EKF comparator | Planar pose; motion-derived VLP heading; live LSTM/VLP/odometry navigation | HTC Vive; 1.0×0.9 m CNC testbed | Live LSTM median position error 0.023–0.034 m over final 100 steps | EKF evidence is offline only and no direct ranking against the heterogeneous studies above is claimed. |

## Recent-work search

Searches covered robotic VLP, VLP/INS, photodiode navigation, and VLP/LiDAR work
published through 1 October 2026. Sun et al. (2025) was added as a recent physical
robot study. A 2025 IEEE conference paper on visible-light mobile-robot
multi-sensor fusion (DOI 10.1109/ICIEA65512.2025.11148429) was identified, but its
abstract did not expose the reference method, area, or numerical accuracy needed
for a traceable table row. A 2026 hybrid VLP/camera/UKF article (DOI
10.1016/j.trpro.2026.03.032) was excluded because it reports a modeling and
simulation study rather than a physical robot experiment. These exclusions prevent
abstract-only or simulation results being presented as comparable experimental
evidence.

## Table interpretation rule

The manuscript may compare system scope—receiver type, auxiliary sensing, estimated
state, live operation, reference method, and scale—but must not rank the numerical
errors. The listed values mix medians, means, path deviations, static points, and
dynamic trajectories.
