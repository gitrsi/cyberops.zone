![MAD20 Threat Hunting and Detection Engineering](images/Threat_Hunting_and_Detection_Engineering.jpg "MAD20 Threat Hunting and Detection Engineering")

> :bulb: Notes on "MAD20 Threat Hunting and Detection Engineering"




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

![Hypotheses](images/2/hypotheses56.png)

![Hypotheses](images/2/hypotheses57.png)



# Determine Data Requirements

## Balancing Data Requirements

![Data Requirements](images/3/datarequirements1.png)

![Data Requirements](images/3/datarequirements2.png)

![Data Requirements](images/3/datarequirements3.png)

![Data Requirements](images/3/datarequirements4.png)

![Data Requirements](images/3/datarequirements5.png)

![Data Requirements](images/3/datarequirements6.png)

![Data Requirements](images/3/datarequirements7.png)

![Data Requirements](images/3/datarequirements8.png)

![Data Requirements](images/3/datarequirements9.png)


## Diving into Data Sources

![Data Requirements](images/3/datarequirements10.png)

![Data Requirements](images/3/datarequirements11.png)

![Data Requirements](images/3/datarequirements12.png)

![Data Requirements](images/3/datarequirements13.png)

![Data Requirements](images/3/datarequirements14.png)

![Data Requirements](images/3/datarequirements15.png)

![Data Requirements](images/3/datarequirements16.png)

![Data Requirements](images/3/datarequirements17.png)

![Data Requirements](images/3/datarequirements18.png)

![Data Requirements](images/3/datarequirements19.png)

![Data Requirements](images/3/datarequirements20.png)

![Data Requirements](images/3/datarequirements21.png)

![Data Requirements](images/3/datarequirements22.png)

![Data Requirements](images/3/datarequirements23.png)

![Data Requirements](images/3/datarequirements24.png)

## Leveraging External Resources for Data Requirements

![Data Requirements](images/3/datarequirements25.png)

![Data Requirements](images/3/datarequirements26.png)

![Data Requirements](images/3/datarequirements27.png)

![Data Requirements](images/3/datarequirements28.png)

![Data Requirements](images/3/datarequirements29.png)

![Data Requirements](images/3/datarequirements30.png)


# Identifying and Mitigating Data Collection Gaps

## Identifying Gaps

![Data Gaps](images/4/datagaps1.png)

![Data Gaps](images/4/datagaps2.png)

![Data Gaps](images/4/datagaps3.png)

![Data Gaps](images/4/datagaps4.png)

![Data Gaps](images/4/datagaps5.png)

![Data Gaps](images/4/datagaps6.png)


## Time, Terrain and Behavior Considerations

![Data Gaps](images/4/datagaps7.png)

![Data Gaps](images/4/datagaps8.png)

![Data Gaps](images/4/datagaps9.png)

![Data Gaps](images/4/datagaps10.png)


## Developing a Sensor Strategy

![Data Gaps](images/4/datagaps11.png)

![Data Gaps](images/4/datagaps12.png)

![Data Gaps](images/4/datagaps13.png)

![Data Gaps](images/4/datagaps14.png)

![Data Gaps](images/4/datagaps15.png)

![Data Gaps](images/4/datagaps16.png)

![Data Gaps](images/4/datagaps17.png)

![Data Gaps](images/4/datagaps18.png)

![Data Gaps](images/4/datagaps19.png)


## Using Alternative Data Sources and Analytics

![Data Gaps](images/4/datagaps20.png)

![Data Gaps](images/4/datagaps21.png)

![Data Gaps](images/4/datagaps22.png)

![Data Gaps](images/4/datagaps23.png)

![Data Gaps](images/4/datagaps24.png)

![Data Gaps](images/4/datagaps25.png)

![Data Gaps](images/4/datagaps26.png)

![Data Gaps](images/4/datagaps27.png)

![Data Gaps](images/4/datagaps28.png)


## Communicating with Network Managers

![Data Gaps](images/4/datagaps29.png)

## Validating Configuration






# Implementing and Testing Analytics

## Implementing Analytics

## Validating Analytics

## Improving Performance, Precision, Recall

## Expanding Time, Terrain, Behavior

## Exploring the Three Dimensions

## Updating Analytics


















