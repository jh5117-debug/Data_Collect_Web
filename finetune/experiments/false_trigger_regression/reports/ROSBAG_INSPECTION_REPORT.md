# ROS Bag Inspection Report

- Zip path: `finetune/experiments/false_trigger_regression/private/rosbag-trigger-word.zip`
- Zip size bytes: `331051`
- Bag count: `2`

## Bags

### false_positive

- Expected label: `0`
- Duration seconds: `13.341038`
- Metadata message count: `57`
- Relative DB files: `false_positive_0.db3`

| Topic | Type | Metadata messages | SQLite messages |
|---|---|---:|---:|
| `/microphone/audio/transcription` | `vigil_msgs/msg/ProcessedAudio` | 3 | 3 |
| `/microphone/audio` | `vigil_msgs/msg/AudioMessage` | 54 | 54 |

SQLite DB3 files:

- `false_positive_0.db3`: 495616 bytes, messages rows `57`

### true_positive

- Expected label: `1`
- Duration seconds: `5.274978`
- Metadata message count: `23`
- Relative DB files: `true_positive_0.db3`

| Topic | Type | Metadata messages | SQLite messages |
|---|---|---:|---:|
| `/microphone/audio/transcription` | `vigil_msgs/msg/ProcessedAudio` | 1 | 1 |
| `/microphone/audio` | `vigil_msgs/msg/AudioMessage` | 22 | 22 |

SQLite DB3 files:

- `true_positive_0.db3`: 217088 bytes, messages rows `23`

## Binary Safety

This report inspects metadata and SQLite table counts only. It does not dump raw CDR message blobs.
