# Dataset splits and naming

The source files use zero-based run identifiers, while the thesis describes
the same runs as Datasets 1–10.

| Source run | Thesis dataset | Original RNN role |
|---:|---:|---|
| `exp1_0` | Dataset 1 | training |
| `exp1_1` | Dataset 2 | training |
| `exp1_2` | Dataset 3 | training |
| `exp1_3` | Dataset 4 | training |
| `exp1_4` | Dataset 5 | training |
| `exp1_5` | Dataset 6 | training |
| `exp1_6` | Dataset 7 | training |
| `exp1_7` | Dataset 8 | training |
| `exp1_8` | Dataset 9 | validation/model selection |
| `exp1_9` | Dataset 10 | held-out evaluation |

These labels describe the split used for the saved Paper 2 RNN. They should
not be silently changed when conducting later experiments with different data
volumes or resampling strategies.

The live datasets are independent recorded navigation experiments organized
by navigation method (RNN, VLP-only, or odometry-only) and path shape (square,
hourglass, or random).
