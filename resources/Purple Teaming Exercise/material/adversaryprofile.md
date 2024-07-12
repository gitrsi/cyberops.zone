

# Adversary Profile

## Description

Black Basta is considered a ransomware-as-a-service (RaaS) variant and was first identified in April
2022. Black Basta affiliates have impacted a wide range of businesses and critical infrastructure in
North America, Europe, and Australia. As of May 2024, Black Basta affiliates have impacted over 500
organizations globally.

Black Basta affiliates use common initial access techniques—such as phishing and exploiting known
vulnerabilities—and then employ a double-extortion model, both encrypting systems and exfiltrating
data

## Objective

As most campaigns are financially motivated, victims are usually given between 10 and 12 days to pay the ransom before the ransomware group publishes their data on the Black Basta TOR site, Basta News.

## TTPs

| Tactic | Technique | Procedure | CTI Reference |
| ----------- | ----------- | ----------- | ----------- |
| Reconnaissance | xxx | xxx | xxx |
| Resource Development | xxx | xxx | xxx |
| Initial Access | xxx | xxx | xxx |
| Execution | T1047 Windows Management Instrumentation | Has been observed to use Windows Management Instrumentation (WMI) to spread and execute files over the Network. | https://www.trendmicro.com/vinfo/us/security/news/ransomware-spotlight/ransomware-spotlight-blackbasta |
| Persistence | xxx | xxx | xxx |
| Privilege Escalation | xxx | xxx | xxx |
| Defense Evasion | T1562.001 Impair Defenses: Disable or Modify Tools | Performed a defense evasion mechanism by attempting to disable Windows Defenders’ real-time monitoring via PowerShell | https://www.cisa.gov/sites/default/files/2024-05/aa24-131a-joint-csa-stopransomware-black-basta_1.pdf, https://www.trendmicro.com/en_us/research/24/b/threat-actor-groups-including-black-basta-are-exploiting-recent-.html |
| Credential Access | xxx | xxx | xxx |
| Discovery | xxx | xxx | xxx |
| Lateral Movement | T1021.001 | Uses RDP to spread and execute the malware across the network. | https://www.trendmicro.com/vinfo/us/security/news/ransomware-spotlight/ransomware-spotlight-blackbasta |
| Lateral Movement | T1021.002 Remote Services: SMB/Windows Admin Shares | Infected machines communicate with the threat actor-controlled device via an SMB channel | https://www.kroll.com/en/insights/publications/cyber/black-basta-technical-analysis |
| Lateral Movement | T1570 Lateral tool transfer | Impacket was used to move payloads between compromised systems. | https://www.rapid7.com/blog/post/2024/05/10/ongoing-social-engineering-campaign-linked-to-black-basta-ransomware-operators/ |
| Lateral Movement | T1570 Lateral tool transfer | Uses tools like PsExec and BITSAdmin to spread the malware laterally across the network. | https://www.microsoft.com/en-us/security/blog/2024/05/15/threat-actors-misusing-quick-assist-in-social-engineering-attacks-leading-to-ransomware/ |
| Collection | xxx | xxx | xxx |
| Command and Control | xxx | xxx | xxx |
| Exfiltration | xxx | xxx | xxx |
| Impact | xxx | xxx | xxx |




