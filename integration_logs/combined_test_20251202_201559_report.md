# Vision-Motor Integration Analysis Report
Generated: 2025-12-02T20:15:59.874495
Data file: integration_logs\combined_test_20251202_201559.csv

## Metrics

```json
{
  "timing": {
    "frame_time_mean": 5.363991327374895,
    "detection_time_mean": 76.8037984304339,
    "control_time_mean": 3.9288666203757314,
    "total_latency_mean": 86.09665637818452,
    "total_latency_max": 1196.7103481292725,
    "total_latency_std": 39.41754593195933,
    "fps_mean": 11.632748538011695
  },
  "position": {
    "gain": 3.3027059049505842,
    "offset": -0.061378846684241484,
    "r_squared": 0.8558341323661467,
    "left_response_mean": -0.5621914280227652,
    "right_response_mean": 0.5429591217504564,
    "center_response_mean": -0.0033811046295086555
  },
  "distance": {
    "gain": 3.2291450810806253,
    "offset": 0.15292673433308257,
    "r_squared": 0.7569605966201494,
    "far_response_mean": 0.9051370033074484,
    "close_response_mean": 0.4358543028828355,
    "good_response_mean": 0.30000000000000004
  },
  "motor_tracking": {
    "left_rmse": 0.7017931060929862,
    "right_rmse": 2.3566280781450137,
    "left_mae": 0.5552961143195005,
    "right_mae": 1.186135888816271
  },
  "zones": {
    "LEFT_FAR": {
      "left": 0,
      "right": 0
    },
    "LEFT_GOOD": {
      "left": 0,
      "right": 0
    },
    "LEFT_CLOSE": {
      "left": 0,
      "right": 0
    },
    "CENTER_FAR": {
      "left": 0,
      "right": 0
    },
    "CENTER_GOOD": {
      "left": 0,
      "right": 0
    },
    "CENTER_CLOSE": {
      "left": 0,
      "right": 0
    },
    "RIGHT_FAR": {
      "left": 0,
      "right": 0
    },
    "RIGHT_GOOD": {
      "left": 0,
      "right": 0
    },
    "RIGHT_CLOSE": {
      "left": 0,
      "right": 0
    }
  },
  "sync": {
    "sync_lag_ms": 181.73337791398257,
    "correlation_peak": 66.3632040537363
  }
}
```
