# Vision-Motor Integration Analysis Report
Generated: 2025-12-02T20:08:30.804222
Data file: integration_logs\combined_test_20251202_200830.csv

## Metrics

```json
{
  "timing": {
    "frame_time_mean": 6.003949165344238,
    "detection_time_mean": 81.66805362701416,
    "control_time_mean": 4.148940086364746,
    "total_latency_mean": 91.82094287872314,
    "total_latency_max": 1130.5372714996338,
    "total_latency_std": 51.34168258917064,
    "fps_mean": 11.150300601202405
  },
  "position": {
    "gain": 1.9993033338849209,
    "offset": 0.009712161345277914,
    "r_squared": 0.6296090719751389,
    "left_response_mean": -0.2581570634810529,
    "right_response_mean": 0.46396685947602806,
    "center_response_mean": 0.0015734387143146896
  },
  "distance": {
    "gain": 2.66367681974267,
    "offset": 0.14384426679517057,
    "r_squared": 0.6914538304344056,
    "far_response_mean": 0.5011983552373767,
    "close_response_mean": 0.0,
    "good_response_mean": 0.07706422018348624
  },
  "motor_tracking": {
    "left_rmse": 0.6319811024956296,
    "right_rmse": 2.7300162416537135,
    "left_mae": 0.5515807288331671,
    "right_mae": 1.0561044053242743
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
    "sync_lag_ms": 0.0,
    "correlation_peak": 22.946541454590577
  }
}
```
