

Between error detections:
Minimum interval: 0.875 s
Average interval: 15.803 s

Errors per step:
Maximum errors in a single step: 21
Average errors per step: 0.5542





First trial:
model: google/gemma-4-26b-a4b-it
video time: 176 seconds
  ========================================================
  Simulation complete
  ========================================================
  Frames delivered:  353
  Audio delivered:   36 chunks
  ========================================================
  Frames delivered:  353
  Audio delivered:   36 chunks
  Events detected:   16
  Wall time:         648.3s
  Mean detect delay: 219.81s
  Max detect delay:  472.73s
  Results saved to: output/events.json

  Output: output/events_1.json
  Events: 16


Evaluation
  VLM ORCHESTRATOR — EVALUATION REPORT  prompt v6
============================================================
  Tolerance: ±5s

  STEP COMPLETION
  --------------------------------------------------------
    Precision:  54.5%
    Recall:     54.5%
    F1:         0.545
    6/11 matched, 5 FP, 5 FN

  ERROR DETECTION
  --------------------------------------------------------
    Precision:  33.3%
    Recall:     16.7%
    F1:         0.222
    1/6 matched, 2 FP, 5 FN


