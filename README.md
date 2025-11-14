# SNN Privacy Benchmark

## Overview

This project investigates **membership inference attacks** on **Spiking Neural Networks (SNNs)**, a type of neuromorphic AI that processes information through discrete spike events rather than continuous values. The research focuses on analyzing privacy vulnerabilities in these brain-inspired neural networks.

## What are Spiking Neural Networks?

Spiking Neural Networks are the third generation of neural networks that more closely mimic biological neural computation. Unlike traditional ANNs that use continuous activation values, SNNs communicate through discrete spike events over time, making them:
- More energy-efficient
- Better suited for temporal data processing
- Capable of real-time processing
- Promising for neuromorphic hardware

## What are Membership Inference Attacks?

Membership inference attacks are privacy attacks where an adversary tries to determine whether a specific data sample was used to train a machine learning model. This is a serious privacy concern because:
- It can reveal sensitive information about training data
- It violates user privacy in machine learning systems
- It's particularly concerning for medical, financial, or personal datasets

## Research Focus

This project specifically examines:

1. **Side-Channel Analysis**: How spike patterns in SNNs might leak information about training data membership
2. **Privacy Vulnerabilities**: Unique privacy risks in neuromorphic computing systems
3. **Attack Methodologies**: Different approaches to exploit SNN characteristics for membership inference
4. **Defense Mechanisms**: Potential countermeasures to protect against such attacks

## Key Components

- **SNN Training Pipeline**: Scripts to train spiking neural networks on various datasets
- **Spike Collection**: Tools to capture and analyze spike patterns during inference
- **Attack Implementation**: Membership inference attack algorithms tailored for SNNs
- **Analysis Tools**: Utilities for evaluating attack success and side-channel leakage

## Quick Start

1. Clone this repository
2. Install dependencies: `pip install torch torchvision numpy pandas matplotlib seaborn jupyter`
3. Run training: `python main_train.py --model vgg11 --epochs 5`
4. Analyze results with the TVLA notebook: `compute_tvla.ipynb`

---

*Research at the intersection of neuromorphic computing and privacy-preserving AI.*
