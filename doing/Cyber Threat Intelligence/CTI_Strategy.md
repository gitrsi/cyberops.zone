
![Cyber Threat Intelligence Strategy](images/cti_strategy.jpg "Cyber Threat Intelligence Strategy")

# Cyber Threat Intelligence Strategy

## What is Cyber Threat Intelligence?

Cyber Threat Intelligence is the collection, analysis and dissemination of information about current and potential cyber threats. CTI helps to understand:

- Who might attack (threat actors)
- Why (motivations & objectives)
- How (tactics, techniques and procedures - TTPs)
- What they are targeting (assets, vulnerabilities)
- When and where attacks are occurring

CTI helps answering strategic questions like:

- Who might target us and why?
- What trends could affect our industry?
- Where should we invest in cyber defense?
- How do geopolitical shifts affect our cyber risk?
- Are we prepared for the next 12-24 months of threats?

## Types of Cyber Threat Intelligence

| Type       | Description                                            | Consumers                     |
|------------|--------------------------------------------------------|-------------------------------|
| **Strategic**  | High-level insights on trends, actors, motivations     | Executives, risk officers, CISO     |
| **Operational**| Campaign-level details, attack chains, tools used      | CISO, IR teams, threat hunters               |
| **Tactical**   | Specific IOCs and signatures                           | SOC, security analysts        |
| **Technical**  | Data about specific infrastructure (IP addresses, malware hashes) | Security tools, firewalls, IDS/IPS/NDR |

### Strategic CTI

> :question: "Who is out there threatening us, what do they want and what should we do about it?"

Strategic cyber threat intelligence (strategic CTI) provides a high-level, long-term view of the threat landscape. It focuses on understanding adversaries motives, trends and risk implications to inform executive decision-making, risk management and security investment planning.

#### Characteristics

- Understand who might target the organization and why
- Align cybersecurity with business strategy and risk appetite
- Guide resource allocation and policy creation
- Influence executive-level decisions and board reporting

#### Tasks

|Task | Description|
|---|---|
|**Threat Landscape Monitoring** | Tracks global and sector-specific threat trends, geopolitical factors and emerging risks.|
|**Adversary Profiling** | Builds high-level profiles of threat actors (APT groups, criminal organizations), including motives, targets and capabilities.|
|**Risk Impact Analysis** | Maps potential cyber threats to business risks (e.g. supply chain compromise, IP theft, ransomware disruption).|
|**Industry & Sector Threat Analysis** | Identifies threats relevant to your specific industry or geographic region.|
|**Strategic Reporting** | Prepares periodic threat intelligence reports for executives, boards and regulators.|
|**Security Program Influence** | Advises on security initiatives, budget priorities and long-term defensive strategy.|
|**Policy & Regulation Tracking** | Monitors new cyber regulations and frameworks that may affect the organization.|

#### Outputs

|Output | Description | Consumers|
|---|---|---|
|**Executive Threat Reports** | High-level summaries of the evolving threat landscape, tailored to business leadership. | CISO, management, noard|
|**Threat Actor Overviews** | Strategic dossiers on threat groups: motivations, geopolitical ties, targets etc. | Risk management, legal, executives|
|**Risk Forecasts** | Intelligence-driven assessments of future threat scenarios and organizational exposure. | Risk Management, business units|
|**Security Investment Justification** | CTI-informed rationale for new technologies, staff or programs. | Budget holders, CISO|
|**Geopolitical Risk Updates**| Cyber implications of political conflicts, sanctions or instability. | Legal, compliance, executives|
|**Regulatory Intelligence** | Summaries of upcoming legal or regulatory changes affecting cybersecurity. | Compliance, legal, CISO|

### Operational CTI

> :question: "How are attackers operating and how do we prepare for or disrupt them?"

Operational threat intelligence collection and analysis is the ongoing, real-time gathering and processing of threat data to detect, understand and respond to cyber threats at the operational level. It bridges the gap between raw technical indicators and high-level strategic insights by focusing on threat actor behaviors, tools, infrastructure and campaigns that directly impact an organization`s day-to-day security operations.

#### Characteristics

- Campaign-Level Focus
- Threat actor TTPs
- Bridges tactical & strategic CTI
- Supports response and defense planning
- Time-relevant
- Context-rich
- Feeds hunt and detection engineering
- Often human-analyzed

#### Tasks

|Task | Description|
|---|---|
|**Continuous Collection** | Ongoing acquisition of threat data from internal and external sources (e.g. SIEM, honeypots, OSINT, dark web, commercial threat feeds).|
|**Operational Focus** | Prioritizes campaign-level data, adversary infrastructure and TTPs used in real-world attacks.|
|**Correlation & Enrichment** | Aligns threat data with internal telemetry (e.g. logs, alerts) and enriches it with context like MITRE ATT&CK techniques or threat actor attribution.|
|**Threat Prioritization**| Assesses threats based on relevance, risk and potential impact to the organization.|
|**Real-Time Analysis**| Supports SOCs and IR teams by detecting active campaigns, creating hunting hypotheses and driving detection engineering.|

#### Outputs

| Output | Description | Consumers |
|--------|-------------|-----------|
| **Threat Actor Campaign Reports** | Detailed reports describing ongoing or recent attack campaigns, including actor behaviors, objectives and patterns. | Incident response, SOC analysts |
| **TTP (Tactics, Techniques and Procedures) Analysis** | Descriptions of adversary tactics and techniques, often mapped to frameworks like MITRE ATT&CK. | SOC, detection engineering, threat hunting |
| **Attack Chain Mapping** | Diagrams or reports showing the stages of a specific attack (e.g. initial access, lateral movement, exfiltration). | SOC, incident response, threat hunting |
| **IOC (Indicators of Compromise) Lists** | Curated and updated lists of IOCs (e.g. IP addresses, domains, file hashes) tied to active threat campaigns. | SOC, SIEM, EDR, IDS/IPS/NDR tools |
| **Vulnerability and Exploit Analysis** | Analysis on 0-day or actively exploited vulnerabilities, providing insights into immediate patching needs. | IT teams, vulnerability management |
| **Adversary Playbooks** | Operational playbooks outlining common attack strategies, tactics and tools used by specific threat groups. | Incident response, security operations |
| **Advisories and Warnings** | Timely alerts about ongoing campaigns, targeted industries and newly identified threat actors or tactics. | CISO, management, incident response |
| **Detection and Response Recommendations** | Actionable recommendations for improving detection, response and blocking based on observed threat activity. | SOC, incident response, detection engineering |

### Tactical CTI

> :question: "What signs of compromise should we monitor for right now?"

Tactical CTI focuses on short-term, actionable intelligence about the methods and artifacts used by threat actors - typically expressed as Indicators of Compromise (IOCs) and detection signatures. Its primary goal is to support detection, alerting and blocking in real-time security operations.

#### Characteristics

- Highly granular and focused on machine-consumable artifacts
- Short-term relevance - IOCs may change frequently or become obsolete quickly
- Strongly supports incident detection and response efforts
- Often the first layer of defense fed directly into security tools

#### Tasks

| Task | Description |
|------|-------------|
| **IOC Collection & Validation** | Gather IPs, URLs, file hashes, domain names and validate them for accuracy and relevance. |
| **Signature Generation** | Create or adapt YARA, Snort, Sigma or other detection rules to identify malicious activity. |
| **Threat Indicator Enrichment** | Add context to IOCs (e.g. associated threat actor, campaign, kill chain stage). |
| **IOC Dissemination** | Share curated IOCs with SOC tools like SIEM, IDS/IPS/NDR, EDR etc. |
| **Short-Term Threat Tracking** | Monitor known malicious infrastructure or artifacts that pose immediate threats. |
| **Detection Feedback Loop** | Gather feedback from SOC/IR on the effectiveness and false positives of shared indicators. |

#### Outputs

| Output | Description | Consumers |
|--------|-------------|-----------|
| **Curated IOC Feeds** | Timely and relevant IPs, domains, hashes and other artifacts. | SOC, IR, SIEM/EDR/IDS/NDR tools |
| **Detection Signatures** | Custom YARA, Sigma, Suricata or Snort rules based on threat intelligence. | Detection engineering, SOC |
| **IOC Enrichment Reports** | IOC context (e.g. actor attribution, TTPs, kill chain stage). | SOC, incident response |
| **Threat Hunt Pivot Data** | Known infrastructure or artifacts to guide threat hunting. | Threat hunters, blue teams |
| **False Positive Reduction** | Refined indicators and rules based on operational feedback. | SOC, detection engineering |


### Technical CTI

> :question: "What are the exact technical elements being used in current or past attacks?"

Technical CTI refers to highly specific, machine-readable data about threat actor infrastructure, malware characteristics, vulnerabilities and exploits. It sits between tactical and operational CTI and provides the technical depth needed to feed detection systems, automate defenses and understand threats at a binary or protocol level.

#### Characteristics

- Highly Specific and Machine-Readable
- Short Lifespan
- Infrastructure-Centric
- Supports Detection and Automation
- Extracted from Technical Sources
- Feeds IOC Repositories
- Limited Context
- Requires Continuous Validation


#### Tasks

| Task | Description |
|------|-------------|
| **Infrastructure Analysis** | Monitor and analyze malicious IP addresses, domains, C2 servers etc. |
| **Malware Reverse Engineering** | Dissect malware samples to extract behavior patterns, signatures and communication methods. |
| **Exploit and Vulnerability Tracking** | Analyze exploits in use, CVE targeting trends and technical exploit chains. |
| **Protocol and Payload Analysis** | Study command-and-control (C2) protocols and payload delivery mechanisms. |
| **Automated Feed Generation** | Create machine-consumable threat feeds for integration with defense tools. |
| **Sandboxing and Behavioral Analysis** | Use automated tools or sandboxes to study malware behavior and extract IOCs/TTPs. |


#### Outputs

| Output | Description | Consumers |
|--------|-------------|-----------|
| **Malware Behavior Reports** | Technical details of malware functionality, persistence, obfuscation etc. | Incident response, malware analysts |
| **C2 Infrastructure Maps** | IPs, domains, ports and communication behavior of adversary infrastructure. | SOC, detection engineering |
| **Exploit Analysis Reports** | CVE exploitation methods, affected platforms and patch guidance. | IT, vulnerability management |
| **Automated IOC Feeds** | High-fidelity threat data for use in SIEMs, EDR, IDS/IPS/NDR etc. | SOC, security tools |
| **Decryption or Extraction Scripts** | Tools or scripts to decode or extract artifacts from malware samples. | Reverse engineers |
| **Technical Threat Notes** | Bulletins covering specific malware families, toolkits or frameworks (e.g. Cobalt Strike, Emotet). | Blue teams, incident response|


### Strategic vs Operational vs Tactical CTI

|Type | Focus | Consumers | Content|
|---|---|---|---|
|**Strategic** | Trends, actors, motivations, risk | Executives, CISO | Reports, forecasts, actor profiles|
|**Operational** | Campaigns, attack chains, TTPs | SOC, IR, Detection Engineering | Playbooks, detection plans|
|**Tactical** | IOCs, signatures | SOC, Analysts | IPs, hashes, YARA, Snort rules|



## Integration of CTI into the Cyber Risk Management Process
### Governance and Framework Alignment

- Strategic CTI Input
    - CTI informs leadership and governance teams about global threat trends, geopolitical risks and emerging attack vectors relevant to the business.
- Risk Appetite Definition
    - Helps executives understand the real-world cyber threat landscape and align risk tolerance accordingly.
- Framework Mapping
    - CTI supports alignment with frameworks like NIST by contextualizing threats within industry verticals.

### Risk Assessment & Identification

- Threat Actor Profiling
    - CTI helps identify threat actors targeting your sector (e.g. APT groups in financial services, ransomware gangs in healthcare).
- TTP Analysis
    - Understanding how attackers operate (e.g. phishing campaigns, credential stuffing, exploiting zero-days) enables more precise identification of potential risk vectors.
- Threat Intelligence Feeds
    - Use commercial, open-source and information sharing organization feeds to uncover active and emerging threats.
- Dark Web Monitoring
    - Identifies exposed credentials, leaked data or mentions of your organization, providing early indicators of potential attacks.
- Attack Surface Intelligence
    - Helps identifying, monitoring and analyzing all the possible entry points that an attacker can potentially exploit
- Threat Modeling Support
    - CTI enables more realistic and dynamic threat models by incorporating current adversarial TTPs.    
- Contextual Risk Scoring
    - CTI enriches vulnerability data by showing real-world exploitation and actor interest-helping prioritize patching and mitigation.

### Risk Treatment

- Prioritization of Controls & Detections
    - CTI-driven insights focus investment on controls that protect against relevant threats.
    - Use threat intelligence to build or refine detections and alert logic.
- Tailored Awareness Training
    - CTI helps tailor security awareness to the most relevant attack methods.

### Communication and Reporting

- Executive Briefings
    - Strategic CTI provides high-level threat summaries to leadership, showing how threats affect business units and what`s being done to mitigate them
- Risk Reports with Threat Context
    - Risk dashboards can integrate CTI to explain why certain risks are prioritized
- Regulatory & Compliance Alignment
    - CTI supports evidence reporting for regulations and frameworks like ISG, DSG, GDPR, NIS2, DORA ISO 27001 etc.

### Continuous Improvement

- Threat Landscape Tracking
    - CTI ensures the organization adapts as new threats emerge.
- Maturity Progression
    - As threat intelligence capabilities evolve, they contribute to cyber maturity models and roadmaps.


## Roles & Responsibilities

### Roles

#### Threat Intelligence Team

- Collection
    - Gather threat data from internal logs, OSINT, feeds, dark web, ISACs/ISAOs/ISAOs etc.
    - Collect IOCs, TTPs and vulnerabilities.
    - Monitor for threats targeting the organization, industry or region.
- Processing & Normalization
    - Clean, structure, de-duplicate and tag incoming data.
    - Map raw data to frameworks like MITRE ATT&CK, STIX/TAXII etc.
- Analysis
    - Analyze threat actor behavior, motives, capabilities and infrastructure.
    - Correlate threat data with internal telemetry (e.g. SOC logs).
    - Perform threat actor attribution, trend analysis and risk scoring.
- Production
    - Generate actionable intelligence reports (strategic, operational, tactical, technical).
    - Tailor intel products for different audiences (CISO, SOC, management).
    - Create detection rules or IOC feeds for tools like SIEM, IDS/IPS/NDR, firewalls.
- Dissemination
    - Share intelligence to internal teams (SOC, IR, IT, CISO) and external partners or threat-sharing communities.
    - Maintain communication channels with ISACs/ISAOs, government CERTs, vendors etc.
- Feedback & Collaboration
    - Work with the SOC/IR team to improve intel quality based on actual incidents.
    - Collaborate with red/blue teams to simulate or understand threats.
    - Ingest feedback from consumers to tune collection and reporting.
- Threat Modeling & Anticipation
    - Identify potential threats before they materialize.
    - Support threat modeling for new systems, apps or business initiatives.
- Tooling & Automation
    - Maintain CTI platforms (e.g. OpenCTI, MISP etc.).
    - Automate ingestion and sharing via STIX/TAXII, APIs, custom parsers.
- Compliance & Policy Support
    - Ensure intelligence supports regulatory compliance (e.g. ISG, DSG, NIS2, DORA, ISO 27001 etc.).
    - Assist with reporting requirements for regulators or legal teams.

#### CISO

- Strategic Oversight
    - The CISO is responsible for ensuring that the CTI program aligns with the organization`s risk management and business objectives.
- Decision Maker
    - They prioritize actions based on threat intelligence reports and provide strategic direction on mitigating identified risks.
- Stakeholder Communication
    - The CISO communicates threat intelligence findings to upper management and other key stakeholders, explaining potential risks and impacts.
- Resource Allocation
    - Determines budget and resources for threat intelligence capabilities.
- Incident Response Oversight
    - Ensures that the CTI program feeds into incident response plans, helping to guide the response to detected threats.

#### Executive Management

- Risk Management
    - Management receives CTI reports from the CISO to assess strategic risk and allocate resources to mitigate these risks.
- Decision Making
    - They decide on investment in cybersecurity tools, staff and external services based on the threat intelligence provided.
- Compliance
    - Ensure that the CTI program complies with regulatory requirements (e.g. ISG, DSG, GDPR, NIS2, DORA ISO 27001 etc.) and legal obligations related to cybersecurity.
- Support & Endorsement
    - The executive team ensures that cybersecurity and CTI initiatives are supported at all organizational levels.
- Crisis Management
    - In the event of a significant cyber incident, management helps steer the organization's response strategy and public communication.

#### SOC

- Real-Time Threat Monitoring
    - The SOC continuously monitors the organization`s network and systems for suspicious activity, utilizing CTI feeds for up-to-date threat information.
- Incident Detection & Escalation
    - They use CTI to detect and respond to security incidents (e.g. identifying an attacker using known techniques from the ATT&CK framework).
- Threat Hunting
    - SOC teams proactively search for signs of compromise or latent threats, using CTI data to guide their efforts.
- Alert Triage
    - The SOC assesses and prioritizes alerts, ensuring that high-fidelity intelligence gets immediate attention.
- Feedback to CTI
    - Provides real-time feedback to CTI teams based on the tactics and techniques observed in live incidents.

#### Incident Response Team

- Incident Handling
    - When a security incident occurs, the IR team uses CTI to understand the adversary`s tactics and techniques, helping to contain, eradicate and recover from the attack.
- Forensics & Analysis
    - IR teams use CTI data to perform detailed forensic analysis, looking for evidence of threat actor activity and understanding the scope of the attack.
- Collaboration with SOC
    - Works closely with the SOC to ensure that CTI is leveraged to contain and respond to incidents effectively.
- Post-Incident Reporting
    - After an incident, the IR team contributes to post-mortem analysis, using CTI to inform lessons learned and improve defenses.

#### IT

- Infrastructure Protection
    - The IT department applies the intelligence provided by the CTI program to strengthen the organization's technical defenses.
- Patch Management
    - Uses CTI to identify vulnerabilities that adversaries are likely to exploit and ensures timely patching of those vulnerabilities.
- System Hardening
    - Implements system configurations and controls to mitigate specific attack techniques identified in the CTI, such as disabling unused services or restricting administrative privileges.
- Collaboration with SOC/IR
    - Works with the SOC and IR teams to ensure that security controls are properly configured to block threats in real-time.
- Business Continuity
    - Supports business continuity planning by using CTI to anticipate potential risks to the infrastructure, helping the organization plan for disruptions.

#### Suppliers/Vendors

- Risk Sharing
    - Suppliers must share threat intelligence related to their services, especially if they are part of a critical supply chain in order to better understand the risks associated with third-party relationships.
- Compliance & Security Practices
    - Ensure that they adhere to cybersecurity best practices and regulatory requirements, reducing the risk posed by third-party vulnerabilities.
- Incident Notification
    - If a third-party vendor is compromised and it affects your organization, they should provide timely, actionable CTI to the organization`s CTI and IR teams.
- Security Alignment
    - Work closely with the organization to ensure that security measures and threat intelligence are aligned across the supply chain (e.g. aligning on cybersecurity frameworks).

#### Customers

- Data Protection
    - Customers should understand the security measures you have in place, including how threat intelligence is used to protect their data.
- Alerting & Reporting
    - If customers encounter or suspect threats on their end, they should report any anomalies or issues quickly. This could provide valuable intelligence on emerging threats.
- Feedback to Security Posture
    - Customers may share feedback on how they perceive your cybersecurity, which can help the organization enhance its CTI and security programs.
- Supply Chain Risk
    - In certain cases, customers may also be part of a larger supply chain or critical partner network, where the CTI team can learn from their security challenges and ensure that threats are mitigated across the customer ecosystem.

### Responsibilities

| **Role** | **Consumes** | **Provides** |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **SOC (Security Operations Center)** | - Curated IOCs (IPs, hashes, domains) for detection<br>- Enriched context for alerts (e.g. actor profiles, TTPs)<br>- Threat hunting hypotheses and guidance<br>- Detection rules and SIEM tuning recommendations | - Alerts and logs from SIEM and EDR tools<br>- TTPs observed during threat detection<br>- False positive/negative feedback<br>- Alert correlation info |
| **Incident Response (IR)** | - Attack chain intelligence and adversary playbooks<br>- Threat actor background and behavior patterns<br>- Intelligence to guide containment and eradication<br>- Attribution support and malware family analysis | - Forensic data from actual incidents<br>- Malware samples, artifacts, attack chains<br>- Incident timelines and recovery actions<br>- Lessons learned reports |
| **IT Department** | - Vulnerability prioritization based on active exploitation<br>- Intelligence on attack surface risks (e.g. misconfigured services)<br>- Guidance for hardening systems and patching timelines | - Vulnerability and patching data<br>- Asset inventory and system configurations<br>- Logs from firewalls, DNS and proxies<br>- Operational changes that impact security posture |
| **CISO / Management** | - Strategic threat landscape reports<br>- Risk trend analysis aligned to business sectors/regions<br>- KPI-driven metrics on threat exposure and mitigation<br>- Support for board-level decisions and security investment justification | - Strategic business priorities and risk appetite<br>- Legal/regulatory reporting obligations<br>- Decisions on threat model focus (e.g. sectors, regions) |
| **Red/Blue Teams** | - Threat actor simulation data (real-world TTPs)<br>- Updated adversary emulation plans (e.g. from MITRE ATT&CK)<br>- Enriched context for purple teaming exercises | - Insights from internal security testing<br>- Adversary emulation data<br>- Recommendations for detection improvement |
| **Suppliers/Vendors** | - Shared IOCs and threat reports relevant to mutual risk<br>- Coordinated disclosure of vulnerabilities and exploits<br>- Sector-specific threat insights to improve partner resilience | - Threat feeds and intel reports<br>- Shared IOCs and malware analysis<br>- Vendor-specific threat data<br>- SLAs and exposure information |
| **Other 3rd Party (ISACs/ISAOs, CERTs)** | - Contributions of anonymized threat data and incident insights<br>- Collaboration on active threat campaigns<br>- Cross-sector intelligence sharing and trend analysis | - Sector-specific threat intelligence<br>- Early warnings and alerts<br>- Peer-shared incident data<br>- Collaborative analysis of threats |
| **Customers** | - Notifications about relevant cyber threats or supply chain risks<br>- Security transparency (e.g. threat modeling summaries or red team results) | - Suspicious activity reports<br>- Threats seen on their side that may involve an organization's infrastructure<br>- Vulnerability disclosures or complaints |


## CTI Data Sources

Cyber Threat Intelligence (CTI) data sources are the raw inputs - such as logs, alerts, threat feeds and human insights - that provide evidence and context about malicious activity, threat actors, vulnerabilities and emerging threats. 
These sources are collected, analyzed and enriched by CTI teams to transform data into actionable intelligence that supports decision-making across security operations, incident response, risk management and strategy.

The following list contains the most common but also includes some advanced/unconventional CTI data sources:

| **Source Type**       | **Examples**                                                                                   | **Value to CTI**                                                                 |
|-----------------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| **Internal Logs & Telemetry** | - SIEM logs<br>- Firewall logs<br>- DNS, proxy, email logs<br>- EDR/XDR events          | Baseline behavior, anomaly detection and IOC correlation                       |
| **Network Infrastructure Data** | - NetFlow<br>- IP geolocation<br>- BGP route anomalies                                       | Network behavior profiling, threat actor infrastructure detection               |
| **Incident Response Reports** | - Forensics data<br>- Malware samples<br>- Timeline analysis                          | Real attack insights, malware behavior and attacker TTPs                       |
| **Security Operations Center (SOC)** | - Alert data<br>- Escalated cases<br>- Threat hunting results                       | Context for detections and evidence of active threats                           |
| **Vulnerability Management** | - CVE data<br>- Patch status<br>- Asset criticality                                    | Threat prioritization and risk scoring                                          |
| **Threat Feeds (Commercial)** | - Recorded Future<br>- Flashpoint<br>- Intel471<br>- Anomali                            | Curated, timely IOCs, TTPs, actor profiles                                      |
| **Threat Feeds (Free/Open Source)** | - AlienVault OTX<br>- Abuse.ch<br>- MalwareBazaar<br>- PhishTank                    | Crowd-sourced and community-driven intelligence                                 |
| **Industry-Specific Feeds**      | - Healthcare: H-ISAC<br>- Finance: FS-ISAC<br>- Energy: E-ISAC                                     | Sector-unique threats and mitigation strategies                                 |
| **Threat Sharing Communities** | - ISACs/ISAOs<br>- CERTs<br>- Government portals (e.g. CISA, ENISA, Europol)               | Sector-specific intel, collaboration, early warnings                            |
| **Dark Web & Deep Web** | - TOR forums<br>- Criminal marketplaces<br>- Paste sites                                   | Actor chatter, breach data, threat actor intent                                 |
| **OSINT (Open Source Intelligence)** | - News articles<br>- Blogs<br>- GitHub<br>- Social media<br>- VirusTotal, Shodan     | Discovery of emerging threats, exploits or exposed data                        |
| **Human Intelligence (HUMINT)** | - Insider reports<br>- Security researchers<br>- Trusted contacts in the field       | Early insights into threats before public exposure                              |
| **Malware Analysis Tools** | - Sandboxes (e.g. Cuckoo, Any.Run)<br>- YARA rules<br>- Static/dynamic analysis         | Understanding malware behavior and creating detection artifacts                 |
| **Red/Blue/Purple Team Exercises** | - Adversary simulations<br>- Penetration testing<br>- Emulation reports            | Insights into detection blind spots and realistic attacker paths                |
| **MITRE ATT&CK Framework** | - TTP mapping<br>- Adversary emulation plans<br>- Defensive gap analysis                 | Standardized reference for attacker techniques and behaviors                    |
| **Sensor Networks / Honeynets**  | - Project HoneyNet<br>- T-Pot<br>- Custom cloud honeynets                                           | Captures real-world attacker TTPs and live malware payloads                     |
| **DevOps & CI/CD Monitoring**   | - Git repositories<br>- Pipeline logs<br>- Secrets scanning (e.g. GitLeaks)                | Detect leaked credentials, misconfigurations or exposed tokens                 |
| **Custom Threat Simulation Labs**| - Internal threat actor emulation environments                                                      | Simulates attacks for tool evaluation, detection gap analysis                   |
| **Threat Modeling Tools**        | - Microsoft Threat Modeling Tool<br>- OWASP Threat Dragon                                          | Design-time risk identification and actor mapping                              |
| **Supply Chain Telemetry**       | - Partner SIEM feeds (via contracts)<br>- API logs<br>- Git repos                                  | Extended visibility into shared attack surfaces                                 |
| **Visual Recon Tools**          | - Shodan (screenshots)<br>- Censys snapshots<br>- EyeWitness                                        | Detect branding abuse, unpatched services and attacker reconnaissance targets  |
| **OT/ICS Threat Intelligence**   | - Dragos, Claroty, Nozomi feeds                                                                     | Intelligence for industrial systems (SCADA, PLCs etc.)                         |
| **Public Repositories** | - GitHub IOC lists<br>- JSON/STIX repositories<br>- Shared YARA/Suricata rules              | Reusable technical artifacts and community research                             |
| **Cloud & SaaS Provider Intelligence** | - Microsoft Defender TI<br>- Google Chronicle<br>- AWS GuardDuty feeds            | Cloud-focused insights, abuse reports and platform-wide threats                |
| **Bug Bounty Programs**         | - HackerOne<br>- Bugcrowd<br>- Intigriti                                                     | Early insight into vulnerabilities and how attackers approach your assets       |
| **Phishing Intelligence**       | - Email honeypots<br>- Phishing campaign data<br>- Brand abuse monitoring                   | Detection of credential harvesting, brand impersonation and attack staging     |
| **Brand Monitoring Services**   | - Domain spoofing reports<br>- Logo abuse on fake websites                                   | Helps detect external impersonation or fraud campaigns                          |
| **DNS Intelligence**            | - Passive DNS<br>- Domain age and reputation<br>- WHOIS history                             | Tracks domain infrastructure used by threat actors                              |
| **TLS/SSL Certificate Logs**   | - Certificate Transparency (CT) logs                                                         | Detects malicious or suspicious domains using similar certs                     |
| **Social Media Monitoring**     | - Twitter<br>- Telegram<br>- Discord<br>- Forums                                             | Early chatter on vulnerabilities, exploits or breach disclosures               |
| **Exploit Markets & Leak Sites**| - Exploit.in<br>- BreachForums<br>- 0day marketplaces                                        | Access to stolen data, exploits and actor capabilities                         |
| **Academic & Research Papers**  | - ArXiv<br>- IEEE Xplore<br>- University security research                                   | Deep technical understanding of new attack methods and detection techniques     |
| **Security Conference Materials** | - DEF CON, Black Hat, SANS, FIRST talks<br>- Slide decks, videos, whitepapers              | Leading-edge threat research, tools and tactics shared by experts              |
| **Internal Deception Systems** | - Honeypots<br>- Honeytokens<br>- Deception grids                                            | Detecting attacker movement and techniques within your network                  |
| **Threat Intel Platforms (TIPs)**| - ThreatConnect<br>- MISP<br>- IBM X-Force Exchange                                          | Aggregates, enriches and shares intel across all internal and external feeds   |
| **Fraud & Financial Threat Intel** | - Payment fraud systems<br>- Account takeover detection tools                              | Insight into financially motivated attackers, botnets and social engineering   |
| **Threat Actor Profiling Tools**| - Intel 471 actor profiles<br>- Group-IB, Kaspersky reports                                  | Understand motivations, targets and capabilities of attacker groups            |
| **Cyber Insurance Partners**    | - Incident trend data<br>- Claim statistics                                                  | Real-world attack frequency and financial impact by sector                      |
| **Insider Threat Programs**      | - DLP alerts<br>- Access control logs<br>- HR flags                                                 | Helps identify malicious or negligent insiders and early warning behaviors      |
| **Insider HUMINT (Human Intel)** | - Interviews, exit interviews, disgruntled user chatter                                             | Provides context on internal risk and whistleblower threats                     |
| **Behavioral Biometrics**        | - Mouse movement, typing patterns, session behavior analytics                                       | Detects abnormal user behavior for insider threat intel                         |
| **Geopolitical Intelligence Feeds** | - Stratfor, Recorded Future geopolitical layer, government briefings                             | Aligns cyber activity with global events and nation-state threats               |
| **Psychographic & Linguistic Profiling** | - Language patterns on forums<br>- Behavioral fingerprinting                                    | Enhances actor attribution (especially on dark web or Telegram)                |
| **Threat Attribution Engines**   | - Proprietary actor tracking tools (e.g. ThreatConnect, Intel 471)                                | Clusters activity, TTPs, infrastructure and malware to groups                  |
| **Threat Actor Dossier Databases** | - FireEye, CrowdStrike, Group-IB actor databases                                                  | Enriched profiling with history, motives, targets and tools used               |
| **Deception-as-a-Service**      | - Cymmetria, TrapX, Illusive Networks                                                               | Outsourced attacker engagement for TTP collection                               |
| **Deepfake & Impersonation Monitoring** | - Tools that detect synthetic voice, video or manipulated content                             | Protects against misinformation, fraud and disinfo campaigns                   |
| **Language Translation Pipelines** | - Native speaker review of Chinese/Russian/Iranian hacker forums                                 | Avoids lost-in-translation issues with TTPs and cultural references             |
| **Threat Research Labs**         | - Cisco Talos<br>- Unit 42<br>- ESET Research                                                       | Cutting-edge malware, APT and infrastructure analysis                          |
| **Darknet Crawlers / AI Agents** | - NLP-enabled agents crawling dark web and encrypted messaging apps                                | Scalable actor tracking and threat discovery                                    |
| **Mobile Threat Intel Sources**  | - App store abuse detection<br>- SMS phishing (smishing) traps                                     | Detects mobile-specific malware and social engineering                         |
| **Public Certificate Authorities**| - Let`s Encrypt logs<br>- DigiCert, GlobalSign records                                              | Infrastructure fingerprinting for C2 domains                                   |


## MITRE ATT&CK Integration

Integrating the MITRE ATT&CK framework into a Strategic Cyber Threat Intelligence (CTI) program provides a structured and detailed way to understand and communicate threat actor tactics, techniques and procedures (TTPs). This can greatly enhance decision-making at the strategic level, ensuring that your cybersecurity posture is aligned with the latest threats and real-world adversary behavior.

### Strategic Input: Adding ATT&CK Tactics and Techniques

The MITRE ATT&CK framework can provide high-level insight into threat actor behavior. At the strategic level, you can use it to:
- Identify trends and adversary groups: By associating attacks with specific ATT&CK tactics (e.g. Initial Access, Persistence, Lateral Movement), you can observe emerging adversary tactics that could affect your industry.
- Prioritize defensive actions: Knowing the TTPs associated with common threat actors or attack methods allows you to make informed decisions on which defenses or mitigations should be prioritized, such as strengthening defenses around Initial Access or improving detection for Lateral Movement.
- Align with threat intelligence sources: Use MITRE ATT&CK mappings from threat reports to assess if your existing security controls address relevant adversary behaviors.

Example:

| **Strategic Input**                     | **MITRE ATT&CK TTP** |  **Usage** |
|-------------------------------|-----------------|-----------------|
|**Vulnerability Management**|Techniques related to exploitation for initial access (e.g. exploitation of remote services T1210)|Align vulnerability management with emerging tactics for Initial Access.|
|**Incident Response Trends**|Techniques related to persistence and privilege escalation (e.g. Hijack execution flow T1038)|Identify which persistence techniques are most relevant to your organization`s threat landscape.|

### Operationalizing ATT&CK Data for Risk and Asset Management

MITRE ATT&CK can inform strategic decisions about asset protection and risk management. For example, if you know that an adversary often uses phishing (T1566) for initial access, you can:
- Focus on employee training and email filtering as a strategic investment.
- Ensure that your endpoint security solutions are capable of detecting techniques like spearphishing and related follow-up activities.

### Enhancing Threat Intelligence Reports

The ATT&CK framework helps transform raw intelligence data into actionable insights. For a strategic CTI program, you can:
- Map threat intelligence feeds to the ATT&CK matrix, highlighting relevant tactics and techniques associated with specific threat actors.
- Use ATT&CK-based threat reports to provide higher-level strategic analysis to executives, showcasing the specific risks to your organization from particular threat actor groups.

| **Threat Actor**                     | **MITRE ATT&CK TTP** |  **Strategic Risk Implication** |
|-------------------------------|-----------------|-----------------|
|APT28|Initial Access: Phishing (T1566)|Phishing represents a significant risk to our organization, necessitating enhanced employee awareness training and advanced email filtering tools.|
|FIN7|Lateral Movement: SMB/Windows Admin Shares (T1021.002)|Emphasize monitoring for SMB traffic and enhance network segmentation to mitigate the risk from this tactic.|

### Enhance CTI with ATT&CK Data

The SOC and incident response teams may gather data on specific techniques and tactics used in active incidents. This ATT&CK-based data can be fed back into the strategic CTI program to:
- Adjust priorities based on real-world observed techniques.
- Update risk registers with specific ATT&CK techniques tied to your organization`s threat model.

Example: If a SOC identifies Credential Dumping (T1003) during an attack, this can help the strategic CTI team understand that attackers may focus on exploiting credential storage weaknesses in the future. This could lead to strategic investments in strengthening Multi-factor Authentication (MFA).

### Alignment with CISO & Executive Reporting

For CISOs and the executive team, integrating MITRE ATT&CK helps provide clarity on the organization`s defense against sophisticated threats. By presenting risk analysis in the form of ATT&CK techniques, you can:
- Show the organization's threat profile based on the MITRE ATT&CK matrix.
- Discuss resilience gaps in terms of specific adversary TTPs and their impact on your organization's strategic assets.

### Example Use Cases

#### Use Case 1: Aligning Security Strategy with Threat Landscape

| Scenario | MITRE ATT&CK usage | Strategic actions |
|----|----|----|
|The CISO needs to ensure the security program is aligned with real-world threats|CTI analysts map top threat actors targeting the sector and region to ATT&CK techniques|Adapt cybersecurity roadmap: prioritize investments in controls that mitigate frequently-used TTPs|

#### Use Case 2: Reporting to the Board with Threat Context
| Scenario | MITRE ATT&CK usage | Strategic actions |
|----|----|----|
|The CISO needs to communicate cyber risk to the board in a business-relevant manner|CTI provides information around active threats and the ATT&CK techniques they use. ATT&CK enables visualizing how many of those techniques the org is prepared to detect or prevent|- Create ATT&CK heat maps showing coverage gaps<br>- Report on risk posture with respect to relevant threats for the organization<br>-Justify security budget based on threat data.|

#### Use Case 3: Threat-Driven Control Assessment

| Scenario | MITRE ATT&CK usage | Strategic actions |
|----|----|----|
|The CISO wants to evaluate whether the existing security stack effectively mitigates real-world threats|- CTI maps adversaries to techniques<br>- Security tools and processes are evaluated against their ability to detect or prevent those techniques|- Launch a threat-informed defense assessment (e.g. purple teaming)<br>- Identify and close control gaps|

#### Use Case 4: Incident Readiness and Playbook Development

| Scenario | MITRE ATT&CK usage | Strategic actions |
|----|----|----|
|The organization lacks tailored incident response playbooks|CTI evaluates the most likely attack paths based on threat actors use of ATT&CK TTPs. These can guide the design of incident scenarios.|- Develop incident response playbooks mapped to specific TTPs<br>- Conduct exercises based on real-world attacker behaviors|

#### Use Case 5: Security Training and Awareness

| Scenario | MITRE ATT&CK usage | Strategic actions |
|----|----|----|
|The CISO wants to raise organizational awareness around cyber threats|CTI provides ATT&CK techniques used in real-world breaches (e.g., phishing using T1566 or abuse of RDP via T1021)|- Tailor employee training around specific techniques and scenarios relevant to the organization<br>- Educate teams on how threat actors operate|

#### Use Case 6: Post-Incident Analysis and Lessons Learned

| Scenario | MITRE ATT&CK usage | Strategic actions |
|----|----|----|
|After an incident, IR needs to document and improve based on gaps and adversary behavior|Use ATT&CK to classify attacker activity by technique, helping to identify what worked and where gaps were|- Enrich post-incident reports with ATT&CK mappings<br>- Derive actions such as improved detection, better logging or updated playbooks for missed techniques.|

#### Use Case 7: Enhancing Alert Context and Prioritization

| Scenario | MITRE ATT&CK usage | Strategic actions |
|----|----|----|
|SOC analysts receive high volumes of alerts with minimal context.|CTI maps IOCs and behavioral patterns in alerts to ATT&CK techniques.|-Enrich alerts with ATT&CK technique IDs and threat actor context<br>- Apply threat-based risk scoring to prioritize response efforts.|

#### Use Case 8: Threat-Informed Detection Engineering

| Scenario | MITRE ATT&CK usage | Strategic actions |
|----|----|----|
|SOC needs to improve detection coverage in response to evolving threats.|CTI identifies trending or sector-specific TTPs. ATT&CK provides standardized technique references for building new detections.|Develop and tune SIEM/EDR rules targeting specific ATT&CK techniques.|

#### Use Case 9: Threat Hunting Hypothesis Generation

| Scenario | MITRE ATT&CK usage | Strategic actions |
|----|----|----|
|Threat hunters need direction for proactive search missions.|CTI maps relevant adversaries ATT&CK TTPs used in ongoing campaigns.|Formulate hunt hypotheses around specific techniques and scope telemetry sources to support those hunts.|

#### Use Case 10: Detection Coverage Reporting

| Scenario | MITRE ATT&CK usage | Strategic actions |
|----|----|----|
|Cyber security leadership must demonstrate how well current defenses align with threat actor TTPs.|ATT&CK provides a visual, structured framework to assess and report coverage.|Use ATT&CK Navigator to present heatmaps of covered and uncovered techniques<br>- Guide SOC investment in detection rule development.|

#### Use Case 11: Threat-Informed Vulnerability Prioritization

| Scenario | MITRE ATT&CK usage | Strategic actions |
|----|----|----|
|Organization has thousands of CVEs, but limited patching windows|- CTI analysts map relevant threat actors to ATT&CK techniques.<br>- These techniques (e.g. T1059 - Command and Scripting Interpreter) are linked to software vulnerabilities (via CVEs) |VM team prioritizes on CVEs that are actively exploited using these techniques.|


#### Use Case 12: Mitigate patching gaps

| Scenario | MITRE ATT&CK usage | Strategic actions |
|----|----|----|
|Patching everything immediately is not feasible|CTI maps unpatched CVEs to techniques|Apply mitigating/compensating measures and deploy detection rules around high-risk, unpatched techniques|

#### Use Case 13: Attack Surface Reduction Strategy

| Scenario | MITRE ATT&CK usage | Strategic actions |
|----|----|----|
|Limited resources or time requires reducing exposure instead of full remediation.|-Use CTI and ATT&CK to identify most relevant TTPs for the organization<br>- Assess software/configs enabling those techniques|Harden configurations, disable high-risk features, reduce unnecessary exposed services|

#### Use Case 14 Board-Level Strategic Reporting

| Scenario | MITRE ATT&CK usage | Strategic actions |
|----|----|----|
|Justify security investments to executives|CTI shows trending techniques and maps those to current internal CVEs and exposures.|-Build business cases showing threat actor behavior vs. internal gaps<br>-Justify expenses on improvements or staffing|

#### Use Case 15: Supporting Cross-Team Collaboration

| Scenario | MITRE ATT&CK usage | Strategic actions |
|----|----|----|
|IR needs to coordinate with CISO, SOC, Threat Hunting and Threat Intel teams.|ATT&CK provides a common language to collaborate across teams.|Use ATT&CK to structure incident debriefs, handoffs, and documentation. Ensure all teams are aligned on what was seen and how to respond.|
