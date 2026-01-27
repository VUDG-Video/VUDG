# VUDG

### VUDG: A Dataset for Video Understanding Domain Generalization ###

VUDG is a benchmark dataset for evaluating domain generalization (DG) in video understanding. It contains 7,899 video clips and 36,388 high-quality QA pairs, covering 11 diverse visual domains, such as cartoon, egocentric, surveillance, rainy, snowy, etc. Each video is annotated with both multiple-choice and open-ended question-answer pairs, designed via a multi-expert progressive annotation pipeline using large multimodal models and human verification.

The dataset maintains semantic consistency across domains to ensure that model performance reflects domain generalization ability rather than semantic variability.

### Zero-Shot Evaluation ###
For zero-shot evaluation, models are directly tested on the VUDG testing set without any training. Please use:
- Videos in the test folder
- Annotation in test_mul.json (for multiple-choice QA) or test_open.json (for open-ended QA)

Models are expected to generalize to unseen domains under real-world distribution shifts.

### Fine-tuning on VUDG Train Set ###
Note: our testing set is strictly not allowed for training.

For fine-tuning in both multi-source domain generalization and single-source domain generalization scenarios:
- Use videos from the trainset folder
- Use annotation from train_multi.json for training
- Evaluate on test videos and test_mul.json

For Multiple Domain Generalization (MultiDG): train on multiple source domains and test on one held-out domain.

For Single Domain Generalization (SingleDG): train on one domain and test on all remaining domains.
