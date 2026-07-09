# Audio Extraction Status

- Status: `ok`
- Reason: Decoded known VIGIL rosbag CDR layout into local WAV windows without ROS 2 message imports.
- Manifest created: `True`
- Manifest path: `finetune/experiments/false_trigger_regression/runs/20260709_022138/rosbag_cases.jsonl`

## ROS 2 Decode Availability

- `rosbag2_py`: ModuleNotFoundError: No module named 'rosbag2_py'
- `rosidl_runtime_py`: ModuleNotFoundError: No module named 'rosidl_runtime_py'
- `rclpy`: ModuleNotFoundError: No module named 'rclpy'
- `vigil_msgs`: ModuleNotFoundError: No module named 'vigil_msgs'

## Exact Resume Command

```bash
cd /home/hj/Data_Collect_Web && PYTHONPATH=finetune/src:finetune/experiments/false_trigger_regression/src:. python finetune/experiments/false_trigger_regression/scripts/run_false_trigger_score_audit.py --manifest finetune/experiments/false_trigger_regression/runs/20260709_022138/rosbag_cases.jsonl
```
