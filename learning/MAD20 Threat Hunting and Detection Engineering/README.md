![MAD20 Threat Hunting and Detection Engineering](images/Threat_Hunting_and_Detection_Engineering.jpg "MAD20 Threat Hunting and Detection Engineering")

> :bulb: Notes on "Building Deep Learning Models with TensorFlow"






# Threat Hunting Fundamentals

## Introduction

Threat hunting
- proactive detection and investigation of malicious activity

Dection engineering
- designing and implementing analytics to discover malicious activity


Threat Hunting Overview

![Threat Hunting Overview](images/threat_hunting_overview.png)



## Detection Approaches

### Key Terms: Precision and Recall
- Precision
    - Ration of true positives to total results
    - Good precision --> very few false positives

- Recall
    - Ration of true positives to total relevant (malicious) events
    - Good recall --> very few false negatives

- Improving one of these often makes the other one worse

### Traditional Detection Approaches

- Signature-based
    - explicitly define malicious behavior
    - good precision
    - detects only what you signature for
    - large signature libraries, hard to manage
    - modern attacks are typically dynamic, signatures are quickly out-of-date
- Allow-List
    - inverse of signature-base detection
    - only approved actions are allowed
    - can be very effective
    - in practice so many exceptions must be made, that adversaries can operate within the allowed
    - developing and maintaining is costly and difficult
- Anomaly-based
    - statistical baseline of "normal"
    - define "normal" as goog, "abnormal" as bad
    - can detect previously unknown attacks
    - usefull where benign behavior follow patterns
    - baseline typically isn't stable: people and network patterns often change
    - requires additional work to investigate anomalies
    - begnin activity is often "abnormal" due to unexpected spikes in network traffic or unusual but benign ativities


### TTP-Based Detection

#### IOCs vs TTPs

IOCs
- known to be malicious
- no advantage for initial victims before IOSs are discovered and shared
- can easily be changed

TTPs
- limited by functionality of the underlying technology targeted
- expensive to develop and maintain an interface


#### Pyramid of Pain
![Threat Hunting Overview](images/pyramid_of_pain.png)


## Prioritization

https://mad.mad20.io/ModuleDetail/19/24

## Methodology Overview



