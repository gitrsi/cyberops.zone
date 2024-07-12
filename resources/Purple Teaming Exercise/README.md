
![Cyber security resources](images/purpleteaming.jpg "Cyber security resources")

> :bulb: Purple Teaming Exercise"


# Flow

![Cyber security resources](material/PT_Exercise.drawio.png "Cyber security resources")

[PT_Exercise.drawio](https://github.com/gitrsi/cyberops.zone/blob/main/resources/Purple%20Teaming%20Exercise/material/PT_Exercise.drawio)

[Draw.io App](https://app.diagrams.net)
# Notes

## CTI

[CTI Blueprint](https://mitre-engenuity.org/cybersecurity/center-for-threat-informed-defense/our-work/cti-blueprints/) based on Behavior seen in relevant Campaigns by relevant Adversaries

## Defensive Scope

Purpose -> Objectives -> Scope

Criteria:
- Functionality: Detection/Mitigation/Investigation/Response
- Platforms
- Implementations

Example Scope: "Detect Execution, Defense Evasion and Lateral Movement Tactics (as used by Blackbasta Ransomware) on Windows Endpoints"


## Prioritization

Criteria:
- Risk = Likelyhood x Impact
- Technology Stack
- Current defensive Gaps
- Datasources
- Complexity
- Variance

Example:
- Risk: n/a
- Tech Stack: Windows Endpoints
- Current Defensive Gaps: Good Detection Coverage in Network, too few Use Cases on Windows Endpoints
- Datasources: Windows Security Event Logs
- Complexity: low
- Variance: low

## Extract TTPs
- Procedure Level
- Create an [Adversary Profile](material/adversaryprofile.md)
- For a more detailed Profile the [Capability Abstraction Template](material/capabilityabstraction.md) can be used
- Create a high Level Adversary Emulation Plan

## Table Top
- Review high Level Plan
- Choose the TTPs to emulate
- Visibility/Expectations matrix
- Create an [Adversary Emulation Plan](https://attack.mitre.org/resources/adversary-emulation-plans/) afterwards

## Exercise
- Kick Off
- Preflight Checks
- Execution
- Tracking

## Results
- Benign vs. malicous
- Refine Hypothesis
- Confidence

# Resources

| What | URL | Description |
| ----------- | ----------- | ----------- |
| MAD20 Purple Teaming Fundamentals | https://github.com/gitrsi/cyberops.zone/tree/main/learning/MAD20%20Purple%20Teaming%20Fundamentals |  |
| Detection Rules Development Framework | https://ipurple.team/2024/02/21/detection-rules-development-framework/ | |
| Summiting the Pyramid of Pain: The TTP Pyramid | https://scythe.io/library/summiting-the-pyramid-of-pain-the-ttp-pyramid |  |
| Purple Team Exercise Framework (PTEF) | https://github.com/scythe-io/purple-team-exercise-framework/blob/master/PTEFv3.md |  |
| Purple Team Exercise Framework Templates | https://github.com/scythe-io/purple-team-exercise-framework/tree/master/templates |  |
| Capability Abstraction | https://posts.specterops.io/capability-abstraction-fbeaeeb26384 |  |
| CTI Templates | https://github.com/center-for-threat-informed-defense/cti-blueprints/wiki/CTI-Templates#user-content-know-threat-actor-report |  |
| CTI Blueprints | https://mitre-engenuity.org/cybersecurity/center-for-threat-informed-defense/our-work/cti-blueprints/ |  |
| Attack Flow | https://mitre-engenuity.org/cybersecurity/center-for-threat-informed-defense/our-work/attack-flow/ |  |
| Micro Emulation Plans | https://mitre-engenuity.org/cybersecurity/center-for-threat-informed-defense/our-work/micro-emulation-plans/ |  |
| MAPPING ATT&CK TO CVE FOR IMPACT | https://mitre-engenuity.org/cybersecurity/center-for-threat-informed-defense/our-work/mapping-attck-to-cve-for-impact/ |  |
| Campaigns | https://attack.mitre.org/campaigns/ |  |
| CISA Advisory: Black Basta | https://www.cisa.gov/sites/default/files/2024-05/aa24-131a-joint-csa-stopransomware-black-basta_1.pdf |  |
| Threat Actor Groups, Including Black Basta, are Exploiting Recent ScreenConnect Vulnerabilities | https://www.trendmicro.com/en_us/research/24/b/threat-actor-groups-including-black-basta-are-exploiting-recent-.html |  |
| Black Basta Ransomware | https://www.sentinelone.com/labs/black-basta-ransomware-attacks-deploy-custom-edr-evasion-tools-tied-to-fin7-threat-actor/ |  |
| Ransomware Attackers May Have Used Privilege Escalation Vulnerability as Zero-day | https://symantec-enterprise-blogs.security.com/threat-intelligence/black-basta-ransomware-zero-day |  |
| Black Basta - Technical Analysis | https://www.kroll.com/en/insights/publications/cyber/black-basta-technical-analysis |  |
| Black Basta Ransomware Analysis, Simulation, and Mitigation | https://www.picussecurity.com/resource/blog/black-basta-ransomware-analysis-cisa-alert-aa24-131a |  |
| Black Basta | https://attack.mitre.org/software/S1070/ |  |
| Ransomware Spotlight: Black Basta | https://www.trendmicro.com/vinfo/us/security/news/ransomware-spotlight/ransomware-spotlight-blackbasta |  |
| Ongoing Social Engineering Campaign Linked to Black Basta Ransomware Operators | https://www.rapid7.com/blog/post/2024/05/10/ongoing-social-engineering-campaign-linked-to-black-basta-ransomware-operators/ |  |
| Threat actors misusing Quick Assist in social engineering attacks leading to ransomware | https://www.microsoft.com/en-us/security/blog/2024/05/15/threat-actors-misusing-quick-assist-in-social-engineering-attacks-leading-to-ransomware/ |  |
| xxx | https |  |

