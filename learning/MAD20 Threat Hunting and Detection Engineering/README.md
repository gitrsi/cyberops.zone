![MAD20 Threat Hunting and Detection Engineering](images/Threat_Hunting_and_Detection_Engineering.jpg "MAD20 Threat Hunting and Detection Engineering")

> :bulb: Notes on "Building Deep Learning Models with TensorFlow"




https://mad.mad20.io/CourseDetail/19



# Threat Hunting Fundamentals

## Introduction

Threat hunting
- proactive detection and investigation of malicious activity

Dection engineering
- designing and implementing analytics to discover malicious activity


Threat Hunting Overview

![Threat Hunting Overview](images/1/threat_hunting_overview.png)



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
![Threat Hunting Overview](images/1/pyramid_of_pain.png)

## Prioritization
![Purpose Driven](images/1/purpose_driven.png)

![Defining Purpose](images/1/defining_purpose.png)

![Prioritizing Technology](images/1/prioritization_technology.png)

![Prioritizing Business Impact](images/1/prioritize_business.png)

![Prioritizing Behavior](images/1/prioritize_behavior.png)

![Purpose Example](images/1/purpose_example.png)

![APT3 Example](images/1/apt3_example.png)

![Summary](images/1/summary.png)


## Methodology Overview

![TTP Hunting Methotology Overview](images/1/ttp_hunting_meth_overview.png)

![TTP Hunting Methotology Overview](images/1/ttp_hunting_meth_overview2.png)

![TTP Hunting Methotology Overview](images/1/ttp_hunting_meth_overview3.png)

![TTP Hunting Methotology Overview](images/1/ttp_hunting_meth_overview4.png)

![TTP Hunting Methotology Overview](images/1/ttp_hunting_meth_overview5.png)

![TTP Hunting Methotology Overview](images/1/ttp_hunting_meth_overview6.png)

![TTP Hunting Methotology Overview](images/1/taught_in_sequence.png)

![TTP Hunting Methotology Overview](images/1/implemented_in_loops.png)


# Develop Hypotheses & Abstract Analytics

## Developing Hypotheses

![Hypotheses](images/2/hypotheses1.png)

![Hypotheses](images/2/hypotheses2.png)

![Hypotheses](images/2/hypotheses3.png)

![Hypotheses](images/2/hypotheses4.png)

![Hypotheses](images/2/hypotheses5.png)

![Hypotheses](images/2/hypotheses6.png)

## Hypotheses Considerations

![Hypotheses](images/2/hypotheses7.png)

![Hypotheses](images/2/hypotheses8.png)

![Hypotheses](images/2/hypotheses9.png)

![Hypotheses](images/2/hypotheses10.png)

![Hypotheses](images/2/hypotheses11.png)

![Hypotheses](images/2/hypotheses12.png)

![Hypotheses](images/2/hypotheses13.png)


## Finding Low-Variance Behaviors

![Hypotheses](images/2/hypotheses14.png)

![Hypotheses](images/2/hypotheses15.png)

![Hypotheses](images/2/hypotheses16.png)

![Hypotheses](images/2/hypotheses17.png)

![Hypotheses](images/2/hypotheses18.png)

![Hypotheses](images/2/hypotheses19.png)

![Hypotheses](images/2/hypotheses20.png)


## Researching Low-Variance Behaviors

![Hypotheses](images/2/hypotheses21.png)

![Hypotheses](images/2/hypotheses22.png)

![Hypotheses](images/2/hypotheses23.png)

![Hypotheses](images/2/hypotheses24.png)

![Hypotheses](images/2/hypotheses25.png)


## Investigating Low-Variance Bahaviors

![Hypotheses](images/2/hypotheses26.png)

![Hypotheses](images/2/hypotheses27.png)

![Hypotheses](images/2/hypotheses28.png)

![Hypotheses](images/2/hypotheses29.png)

![Hypotheses](images/2/hypotheses30.png)

![Hypotheses](images/2/hypotheses31.png)

![Hypotheses](images/2/hypotheses32.png)


## Refining Hypotheses

![Hypotheses](images/2/hypotheses33.png)

![Hypotheses](images/2/hypotheses34.png)

![Hypotheses](images/2/hypotheses35.png)

![Hypotheses](images/2/hypotheses36.png)

![Hypotheses](images/2/hypotheses37.png)

![Hypotheses](images/2/hypotheses38.png)

![Hypotheses](images/2/hypotheses39.png)

![Hypotheses](images/2/hypotheses40.png)

![Hypotheses](images/2/hypotheses41.png)

![Hypotheses](images/2/hypotheses42.png)

![Hypotheses](images/2/hypotheses43.png)

![Hypotheses](images/2/hypotheses44.png)


## Creating Abstract Analytics

![Hypotheses](images/2/hypotheses45.png)

![Hypotheses](images/2/hypotheses46.png)

![Hypotheses](images/2/hypotheses47.png)

![Hypotheses](images/2/hypotheses48.png)

![Hypotheses](images/2/hypotheses49.png)

## Leveraging External Resources for Analytics

![Hypotheses](images/2/hypotheses50.png)

![Hypotheses](images/2/hypotheses51.png)

![Hypotheses](images/2/hypotheses52.png)

![Hypotheses](images/2/hypotheses53.png)

![Hypotheses](images/2/hypotheses54.png)

![Hypotheses](images/2/hypotheses55.png)








# Determine Data Requirements

## Balancing Data Requirements

## Diving into Data Sources

## Leveraging External Resources for Data Requirements




# Identifying and Mitigating Data Collection Gaps

## Identifying Gaps

## Time, Terrain and Behavior Considerations

## Developing a Sensor Strategy

## Using Alternative Data Sources and Analytics

## Communicating with Network Managers

## Validating Configuration






# Implementing and Testing Analytics

## Implementing Analytics

## Validating Analytics

## Improving Performance, Precision, Recall

## Expanding Time, Terrain, Behavior

## Exploring the Three Dimensions

## Updating Analytics
















