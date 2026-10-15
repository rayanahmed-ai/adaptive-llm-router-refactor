# Router Evaluation Report

## V1–V8.4 Version History
- **V1**: Initial classifier model with 15,011 examples, 12,008 train and 3,003 test. Class counts: simple 12,940, medium 1,997, complex 74. Recorded accuracy: 88.88%.
- **V2**: Improved classifier with new features. Validation accuracy: 50.78% on 639 labeled examples. Not directly comparable to V1 due to different evaluation setup.
- **V3**: Added quality metrics. Quality: 0.9251, latency: 638.88 ms, cost signal: 0.48.
- **V4**: Optimized for latency.
- **V5**: Introduced exploration vs exploitation.
- **V6**: Persistent state management.
- **V7.1–V7.7**: Integration milestones.
- **V8.4**: Latest benchmark results.

## V1 Classifier Result
- **Dataset Class Imbalance**: Simple class dominates with 12,940 examples, medium class has 1,997, and complex class has only 74 examples.
- **Majority-Class Baseline**: Always-predict-simple baseline accuracy: approximately 86.20%.

## V2 Validation Result
- **Validation Result**: 50.78% accuracy on 639 labeled examples.
- **Comparison to V1**: Not directly comparable due to different evaluation setup and dataset.

## V3/V4 Quality, Latency, and Cost-Signal Results
- **Quality**: 0.9251.
- **Latency**: 638.88 ms.
- **Cost-Signal**: 0.48. The cost-signal unit and methodology were not documented.

## V5 Exploration vs Exploitation
- **Exploration**: Not detailed.
- **Exploitation**: Not detailed.
- **V7.5 Epsilon Parameters**: epsilon 0.20, minimum epsilon 0.05, decay 0.90.

## V6 Persistent-State Work
- **Integration Milestones**: Not detailed.

## V8 Benchmark Results
- **Success Rate**: Not available in inspected evidence.
- **Quality**: Not available in inspected evidence.
- **Average Latency**: Not available in inspected evidence.
- **P95 Latency**: Not available in inspected evidence.
- **Output Tokens**: Not available in inspected evidence.

## Definitions
- **Accuracy**: Proportion of true results in the total number of cases examined.
- **Precision**: Proportion of positive identifications that were actually correct.
- **Recall**: Proportion of actual positives that were identified correctly.
- **Macro-F1**: Harmonic mean of precision and recall.
- **Success Rate**: Proportion of successful requests.
- **Average Latency**: Mean time taken for a request to complete.
- **P95 Latency**: Time below which 95% of requests complete.
- **Exploration**: Strategy to try new models.
- **Exploitation**: Strategy to use the best-known model.
- **Quality Retention**: Ability to maintain quality over time.

## Discussion
- **Accuracy Insufficiency**: Accuracy is misleading on the imbalanced V1 dataset due to the large number of simple class examples.

## Limitations and Future Work
- **Limitations**: Incomplete validation results for V2, missing quality, latency, and output tokens for V8.
- **Missing Evidence**: Detailed exploration/exploitation results, V6 integration milestones, and V8 benchmark results.
- **Recommended Experiments**: Collect missing V8 benchmark results and document the cost-signal methodology.