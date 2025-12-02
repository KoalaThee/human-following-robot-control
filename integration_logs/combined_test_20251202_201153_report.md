# Vision-Motor Integration Analysis Report
Generated: 2025-12-02T20:11:54.492072
Data file: integration_logs\combined_test_20251202_201153.csv

## Metrics

```json
{
  "timing": {
    "frame_time_mean": 6.410062133005379,
    "detection_time_mean": 80.77444842934968,
    "control_time_mean": 18.83572970272191,
    "total_latency_mean": 106.02024026507698,
    "total_latency_max": 6199.69630241394,
    "total_latency_std": 335.81423331195407,
    "fps_mean": 11.196969696969697
  },
  "position": {
    "gain": 1.0102132848770577,
    "offset": -0.0018880685625878475,
    "r_squared": 0.2951043114662115,
    "left_response_mean": -0.13000339145393375,
    "right_response_mean": 0.1788197611188205,
    "center_response_mean": -0.0024305388684439117
  },
  "distance": {
    "gain": 1.9067887527420153,
    "offset": 0.500878677363368,
    "r_squared": 0.701122338068992,
    "far_response_mean": 0.9610498230717534,
    "close_response_mean": 0.23424657534246576,
    "good_response_mean": 0.1333333333333333
  },
  "motor_tracking": {
    "left_rmse": 0.5349821890333637,
    "right_rmse": 0.8826346674377465,
    "left_mae": 0.3845869163397303,
    "right_mae": 0.7392388127334556
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
    "sync_lag_ms": -1220.9500233332315,
    "correlation_peak": 1.4420048528190414
  }
}
```
