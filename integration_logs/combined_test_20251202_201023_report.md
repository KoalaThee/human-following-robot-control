# Vision-Motor Integration Analysis Report
Generated: 2025-12-02T20:10:24.558911
Data file: integration_logs\combined_test_20251202_201023.csv

## Metrics

```json
{
  "timing": {
    "frame_time_mean": 6.158904800284175,
    "detection_time_mean": 87.970816659682,
    "control_time_mean": 4.080511489004459,
    "total_latency_mean": 98.21023294897063,
    "total_latency_max": 1114.6204471588135,
    "total_latency_std": 51.08693203750486,
    "fps_mean": 10.685567010309278
  },
  "position": {
    "gain": 3.504439774523327,
    "offset": 0.010798293284771844,
    "r_squared": 0.8799345937212077,
    "left_response_mean": -0.49889227322252067,
    "right_response_mean": 0.5234968618498623,
    "center_response_mean": 0.002489195640746975
  },
  "distance": {
    "gain": 1.6965753006654094,
    "offset": 0.2517485059268969,
    "r_squared": 0.8049798889226817,
    "far_response_mean": 0.3932079669439622,
    "close_response_mean": 0.0,
    "good_response_mean": 0.2767605633802817
  },
  "motor_tracking": {
    "left_rmse": 0.5691289695884155,
    "right_rmse": 1.9511094624994494,
    "left_mae": 0.4913052510350255,
    "right_mae": 0.6929397745201955
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
    "sync_lag_ms": 103.0166194201335,
    "correlation_peak": 14.929428974948246
  }
}
```
