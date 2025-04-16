
# What is Cyber Threat Intelligence?
Cyber Threat Intelligence is the collection, analysis, and dissemination of information about current and potential cyber threats. CTI helps to understand:

- Who might attack them (threat actors)
- Why (motivations & objectives)
- How (tactics, techniques, and procedures – TTPs)
- What they’re targeting (assets, vulnerabilities)
- When and where attacks are occurring

The goal is to answer strategic questions like:

Who might target us and why?
What trends could affect our industry?
Where should we invest in cyber defense?
How do geopolitical shifts affect our cyber risk?
Are we prepared for the next 12–24 months of threats?

# Types of Threat Intelligence

| Type       | Description                                            | Consumers                     |
|------------|--------------------------------------------------------|-------------------------------|
| Strategic  | High-level insights on trends, actors, motivations     | Executives, Risk Officers     |
| Operational| Campaign-level details, attack chains, tools used      | CISOs, IR teams               |
| Tactical   | Specific IOCs and signatures                           | SOC, Security Analysts        |
| Technical  | Data about specific infrastructure (IP addresses, malware hashes) | Security tools, firewalls, IDS |

# Integration of CTI into the Cyber Risk Management Process
## 1. Governance and Framework Alignment
- Strategic CTI Input: CTI informs leadership and governance teams about global threat trends, geopolitical risks, and emerging attack vectors relevant to the business.
- Risk Appetite Definition: Helps executives understand the real-world cyber threat landscape and align risk tolerance accordingly.
- Framework Mapping: CTI supports alignment with frameworks like NIST CSF (e.g., ID.RA – Risk Assessment) by contextualizing threats within industry verticals.

## 2. Risk Identification
- Threat Actor Profiling: CTI helps identify threat actors targeting your sector (e.g., APT groups in financial services, ransomware gangs in healthcare).
- TTP Analysis: Understanding how attackers operate (e.g., phishing campaigns, credential stuffing, exploiting zero-days) enables more precise identification of potential risk vectors.
- Threat Intelligence Feeds: Use commercial, open-source, and ISAC/ISAO feeds to uncover active and emerging threats.
- Dark Web Monitoring: Identifies exposed credentials, leaked data, or mentions of your organization, providing early indicators of potential attacks.

## 3. Risk Assessment
- Contextual Risk Scoring: CTI enriches vulnerability data by showing real-world exploitation and actor interest—helping prioritize patching and mitigation. Example: A CVE with a low CVSS score but being actively exploited by ransomware groups = high risk.
- Attack Surface Intelligence: External reconnaissance (e.g., shadow IT, exposed services) is informed by threat actor behaviors and tactics.
- Threat Modeling Support: CTI enables more realistic and dynamic threat models by incorporating current adversarial TTPs.

## 4. Risk Treatment
- Prioritization of Controls: CTI-driven insights focus investment on controls that protect against relevant threats (e.g., endpoint hardening if targeting includes ransomware with initial access via phishing).
- Detection Engineering: Use threat intelligence to build or refine SIEM detections and alert logic (based on known TTPs).
- Tailored Awareness Training: CTI helps tailor security awareness to the most common attack methods (e.g., business email compromise tactics used in your sector).

## 5. Risk Monitoring and Review
- Real-Time Threat Monitoring: CTI feeds into SOC/SIEM systems to enrich alerts with threat context and prioritize response.
- Indicators of Compromise (IOCs): CTI provides actionable IOCs (IPs, hashes, domains) that can be used for blocking or alerting.
- Threat Hunting: CTI supports proactive hunting missions by giving analysts clues on where and what to look for.

## 6. Incident Response & Recovery
- Enrichment of Alerts: During incidents, CTI helps teams understand attacker motives, methods, and escalation potential.
- Playbook Enhancement: Response playbooks are improved with intelligence about adversary dwell time, lateral movement patterns, and exfiltration techniques.
- Collaboration with External Entities: CTI supports coordinated incident handling and information sharing with ISACs, law enforcement, and third parties.

## 7. Communication and Reporting
- Executive Briefings: Strategic CTI provides high-level threat summaries to leadership, showing how threats affect business units and what’s being done to mitigate them.
- Risk Reports with Threat Context: Risk dashboards can integrate CTI to explain why certain risks are prioritized.
- Regulatory & Compliance Alignment: CTI supports reporting and evidence for frameworks like GDPR, SEC cyber disclosure rules, and NIS2 Directive.


## 8. Continuous Improvement
- Threat Landscape Tracking: CTI ensures the organization adapts as new threats emerge (e.g., AI-driven phishing, OT attacks).
- Red Team/Blue Team Tuning: Intelligence-based red teaming simulates real threat actors, and CTI helps blue teams close the gaps.
- Maturity Progression: As threat intelligence capabilities evolve (from tactical to strategic), they contribute to cyber maturity models and roadmaps.


# Roles & Relationships

## Responsibilities
### 1 Threat Intelligence Team
- Collection
    - Gather threat data from internal logs, OSINT, commercial feeds, dark web, ISACs, etc.
    - Collect IOCs (Indicators of Compromise), TTPs (Tactics, Techniques, Procedures), and vulnerabilities.
    - Monitor for threats targeting the organization, industry, or region.
- Processing & Normalization
    - Clean, structure, de-duplicate, and tag incoming data.
    - Map raw data to frameworks like MITRE ATT&CK, STIX/TAXII, etc.
- Analysis
    - Analyze threat actor behavior, motives, capabilities, and infrastructure.
    - Correlate threat data with internal telemetry (e.g., SOC logs).
    - Perform threat actor attribution, trend analysis, and risk scoring.
- Production
    - Generate actionable intelligence reports (strategic, operational, tactical, technical).
    - Tailor intel products for different audiences (CISO, SOC, management).
    - Create detection rules or IOC feeds for tools like SIEM, IDS, firewalls.
- Dissemination
    - Share intelligence to internal teams (SOC, IR, IT, CISO) and external partners or threat-sharing communities.
    - Maintain communication channels with ISACs, government CERTs, vendors, etc.
- Feedback & Collaboration
    - Work with the SOC/IR team to improve intel quality based on actual incidents.
    - Collaborate with red/blue teams to simulate or understand threats.
    - Ingest feedback from consumers to tune collection and reporting.
- Threat Modeling & Anticipation
    - Identify potential threats before they materialize (predictive intelligence).
    - Support threat modeling for new systems, apps, or business initiatives.
- Tooling & Automation
    - Maintain CTI platforms (e.g., MISP, ThreatConnect, Recorded Future).
    - Automate ingestion and sharing via STIX/TAXII, APIs, custom parsers.
- Compliance & Policy Support
    - Ensure intelligence supports regulatory compliance (e.g., NIS2, ISO 27001, etc.).
    - Assist with reporting requirements for regulators or legal teams.

### 2. CISO
- Strategic Oversight
    - The CISO is responsible for ensuring that the CTI program aligns with the organization’s risk management and business objectives.
- Decision Maker
    - They prioritize actions based on threat intelligence reports and provide strategic direction on mitigating identified risks.
- Stakeholder Communication
    - The CISO communicates threat intelligence findings to upper management and other key stakeholders, explaining potential risks and impacts.
- Resource Allocation
    - Determines budget and resources for threat intelligence capabilities (tools, teams, external services).
- Incident Response Oversight
    - Ensures that the CTI program feeds into incident response plans, helping to guide the response to detected threats.

### 3. Management (Executive Leadership)
- Risk Management
    - Management receives CTI reports from the CISO to assess strategic risk and allocate resources to mitigate these risks.
- Decision Making
    - They decide on investment in cybersecurity tools, staff, and external services based on the threat intelligence provided.
- Compliance
    - Ensure that the CTI program complies with regulatory requirements (e.g., GDPR, HIPAA) and legal obligations related to cybersecurity.
- Support & Endorsement
    - The executive team ensures that cybersecurity and CTI initiatives are supported at all organizational levels.
- Crisis Management
    - In the event of a significant cyber incident, management helps steer the organization's response strategy and public communication.

### 4. SOC (Security Operations Center)
- Real-Time Threat Monitoring
    - The SOC continuously monitors the organization’s network and systems for suspicious activity, utilizing CTI feeds for up-to-date threat information.
- Incident Detection & Escalation
    - They use CTI to detect and respond to security incidents (e.g., identifying an attacker using known techniques from the ATT&CK framework).
- Threat Hunting
    - SOC teams proactively search for signs of compromise or latent threats, using CTI data to guide their efforts.
- Alert Triage
    - The SOC assesses and prioritizes alerts, ensuring that high-fidelity intelligence gets immediate attention.
- Feedback to CTI
    - Provides real-time feedback to CTI teams based on the tactics and techniques observed in live incidents.

### 5. Incident Response (IR) Team
- Incident Handling
    - When a security incident occurs, the IR team uses CTI to understand the adversary’s tactics and techniques, helping to contain, eradicate, and recover from the attack.
- Forensics & Analysis
    - IR teams use CTI data to perform detailed forensic analysis, looking for evidence of threat actor activity and understanding the scope of the attack.
- Collaboration with SOC
    - Works closely with the SOC to ensure that CTI is leveraged to contain and respond to incidents effectively.
- Post-Incident Reporting
    - After an incident, the IR team contributes to post-mortem analysis, using CTI to inform lessons learned and improve defenses.

### 6. IT Department
- Infrastructure Protection
    - The IT department applies the intelligence provided by the CTI program to strengthen the organization's technical defenses (e.g., firewalls, network segmentation, endpoint security).
- Patch Management
    - Uses CTI to identify vulnerabilities that adversaries are likely to exploit and ensures timely patching of those vulnerabilities.
- System Hardening
    - Implements system configurations and controls to mitigate specific attack techniques identified in the CTI, such as disabling unused ports or restricting administrative privileges.
- Collaboration with SOC/IR
    - Works with the SOC and IR teams to ensure that security controls are properly configured to block threats in real-time.
- Business Continuity
    - Supports business continuity planning by using CTI to anticipate potential risks to the infrastructure, helping the organization plan for disruptions.

### 7. Suppliers/Third-Party Vendors
- Risk Sharing
    - Suppliers must share threat intelligence related to their services, especially if they are part of your critical supply chain, so you can better understand the risks associated with your third-party relationships.
- Compliance & Security Practices
    - Ensure that they adhere to cybersecurity best practices and regulatory requirements, reducing the risk posed by third-party vulnerabilities.
- Incident Notification
    - If a third-party vendor is compromised and it affects your organization, they should provide timely, actionable CTI to the organization’s CTI and IR teams.
- Security Alignment
    - Work closely with the organization to ensure that security measures and threat intelligence are aligned across the supply chain (e.g., aligning on cybersecurity frameworks).

### 8. Customers
- Data Protection
    - Customers should understand the security measures you have in place, including how threat intelligence is used to protect their data.
- Alerting & Reporting
    - If customers encounter or suspect threats on their end, they should report any anomalies or issues quickly to your team. This could provide valuable intelligence on emerging threats.
- Feedback to Security Posture
    - Customers may share feedback on how they perceive your cybersecurity, which can help the organization enhance its CTI and security programs.
- Supply Chain Risk
    - In certain cases, customers may also be part of a larger supply chain or critical partner network, where the CTI team can learn from their security challenges and ensure that threats are mitigated across the customer ecosystem.

## Collaboration

-> Drawing

### Contributions from CTI
| **Recipient Role** | **Contribution**|
|---------------------------|----------------------------------------------------------------------------------------------------------------------------|
| **SOC (Security Operations Center)** | - Curated IOCs (IPs, hashes, domains) for detection<br>- Enriched context for alerts (e.g. actor profiles, TTPs)<br>- Threat hunting hypotheses and guidance<br>- Detection rules and SIEM tuning recommendations |
| **Incident Response (IR)** | - Attack chain intelligence and adversary playbooks<br>- Threat actor background and behavior patterns<br>- Intelligence to guide containment and eradication<br>- Attribution support and malware family analysis |
| **IT Department**         | - Vulnerability prioritization based on active exploitation<br>- Intelligence on attack surface risks (e.g. misconfigured services)<br>- Guidance for hardening systems and patching timelines |
| **CISO / Management**     | - Strategic threat landscape reports<br>- Risk trend analysis aligned to business sectors/regions<br>- KPI-driven metrics on threat exposure and mitigation<br>- Support for board-level decisions and security investment justification |
| **Red/Blue Teams**        | - Threat actor simulation data (real-world TTPs)<br>- Updated adversary emulation plans (e.g. from MITRE ATT&CK)<br>- Enriched context for purple teaming exercises |
| **Third-Party Vendors**   | - Shared IOCs and threat reports relevant to mutual risk<br>- Coordinated disclosure of vulnerabilities and exploits<br>- Sector-specific threat insights to improve partner resilience |
| **External Communities (ISACs, CERTs)** | - Contributions of anonymized threat data and incident insights<br>- Collaboration on active threat campaigns<br>- Cross-sector intelligence sharing and trend analysis |
| **Customers (where relevant)** | - Notifications about relevant cyber threats or supply chain risks<br>- Assurances of proactive threat monitoring<br>- Security transparency (e.g., threat modeling summaries or red team results) |


### Contributions to CTI
| **Provider Roler** |**Contributions**                                                      |
|------------------------|------------------------------------------------------------------------------------------------------------------------|
| **SOC (Security Operations Center)** | - Alerts and logs from SIEM and EDR tools<br>- TTPs observed during threat detection<br>- False positive/negative feedback<br>- Alert correlation info |
| **Incident Response (IR)** | - Forensic data from actual incidents<br>- Malware samples, artifacts, attack chains<br>- Incident timelines and recovery actions<br>- Lessons learned reports |
| **IT Department**      | - Vulnerability and patching data<br>- Asset inventory and system configurations<br>- Logs from firewalls, DNS, and proxies<br>- Operational changes that impact security posture |
| **CISO / Management**  | - Strategic business priorities and risk appetite<br>- Legal/regulatory reporting obligations<br>- Decisions on threat model focus (e.g., sectors, regions) |
| **Red/Blue Teams**     | - Insights from internal security testing (e.g., simulated attacks)<br>- Adversary emulation data<br>- Recommendations for detection improvement |
| **Third-Party Vendors**| - Threat feeds and intel reports<br>- Shared IOCs and malware analysis<br>- Vendor-specific threat data<br>- SLAs and exposure information |
| **External Communities (ISACs, CERTs)** | - Sector-specific threat intelligence<br>- Early warnings and alerts<br>- Peer-shared incident data<br>- Collaborative analysis of threats |
| **Customers (when applicable)** | - Suspicious activity reports<br>- Threats seen on their side that may involve your infrastructure<br>- Vulnerability disclosures or complaints |

# Inputs for a Strategic CTI Program

### External Inputs

| **Source**                     | **Description** |
|-------------------------------|-----------------|
| **Threat Reports from Vendors** | Reports from companies like Mandiant, Recorded Future, CrowdStrike, etc. offering analysis of global threats and APT activity. |
| **Government & ISAC Alerts**   | National cyber agencies (e.g., CISA, ENISA) and sector-specific Information Sharing and Analysis Centers provide high-level warnings and threat landscape trends. |
| **Open Source Intelligence (OSINT)** | Publicly available data on geopolitical tensions, emerging cybercrime groups, hacktivism, etc. |
| **Industry Threat Sharing Groups** | Peer organizations collaborating on threat trends and challenges. |
| **Academic & Research Papers** | Insight into emerging technologies, future attack vectors, and defensive strategies. |
| **Dark Web Monitoring**        | Intel on planned attacks, leaked credentials, and chatter about industry-specific targets. |
| **Media & News**               | Broader geopolitical or economic developments (e.g., conflicts, sanctions) that may drive cyber activity. |


### Internal Inputs

| **Source**                     | **Description** |
|-------------------------------|-----------------|
| **Incident & SOC Data**        | Trends from internal alerts and incidents — what attackers are actually doing to your organization. |
| **Vulnerability Management Reports** | Data on where internal weaknesses exist and how they relate to external threats. |
| **Business Strategy & Risk Registers** | Understanding what’s most critical to the business and aligning threat analysis with organizational priorities. |
| **Executive Concerns**         | Questions or strategic directions from the C-suite (e.g., "Are we at risk due to rising tensions in region X?"). |
| **IT & Asset Inventory**       | Knowledge of what you’re protecting — the “crown jewels.” |
| **Compliance & Regulatory Requirements** | Understanding what standards you must meet can shape what strategic CTI must monitor and inform. |

# MITRE ATT&CK Integration
Integrating the MITRE ATT&CK framework into a Strategic Cyber Threat Intelligence (CTI) program provides a structured and detailed way to understand and communicate threat actor tactics, techniques, and procedures (TTPs). This can greatly enhance decision-making at the strategic level, ensuring that your cybersecurity posture is aligned with the latest threats and real-world adversary behavior.

## 1. Strategic Input Layer: Adding ATT&CK Tactics and Techniques
The MITRE ATT&CK framework can provide high-level insight into threat actor behavior. At the strategic level, you can use it to:
- Identify trends and adversary groups: By associating attacks with specific ATT&CK tactics (e.g., Initial Access, Persistence, Lateral Movement), you can observe emerging adversary tactics that could affect your industry.
- Prioritize defensive actions: Knowing the TTPs associated with common threat actors or attack methods allows you to make informed decisions on which defenses or mitigations should be prioritized, such as strengthening defenses around Initial Access or improving detection for Lateral Movement.
- Align with threat intelligence sources: Use MITRE ATT&CK mappings from threat reports to assess if your existing security controls address relevant adversary behaviors.

Example:

| **Strategic Input**                     | **MITRE ATT&CK TTP** |  **Usage** |
|-------------------------------|-----------------|-----------------|
|Vulnerability Management|Techniques related to Exploitation for Initial Access (e.g., Exploitation of Remote Services T1203)|Align vulnerability management with emerging tactics for Initial Access.|
|Incident Response Trends|Techniques related to Persistence and Privilege Escalation (e.g., DLL Search Order Hijacking T1038)|Identify which persistence techniques are most relevant to your organization’s threat landscape.|

## 2. Operationalizing ATT&CK Data for Risk and Asset Management
MITRE ATT&CK can inform strategic decisions about asset protection and risk management. For example, if you know that an adversary often exploits Phishing (T1566) for Initial Access, you can:
- Focus on employee training and email filtering as a strategic investment.
- Ensure that your endpoint security solutions are capable of detecting techniques like Spearphishing and related follow-up activities.

## 3. Enhancing Threat Intelligence Reports
The ATT&CK framework helps transform raw intelligence data into actionable insights. For a strategic CTI program, you can:
- Map threat intelligence feeds to the ATT&CK matrix, highlighting relevant tactics and techniques associated with specific threat actors.
- Use ATT&CK-based threat reports to provide higher-level strategic analysis to executives, showcasing the specific risks to your organization from particular threat actor groups (e.g., APT29, FIN7).

| **Threat Actor**                     | **MITRE ATT&CK TTP** |  **Strategic Risk Implication** |
|-------------------------------|-----------------|-----------------|
|APT28|Initial Access: Phishing (T1566)|Phishing represents a significant risk to our organization, necessitating enhanced employee awareness training and advanced email filtering tools.|
|FIN7|Lateral Movement: SMB/Windows Admin Shares (T1021.002)|Emphasize monitoring for SMB traffic and enhance network segmentation to mitigate the risk from this tactic.|

## 4. Feedback Loop: Enhance CTI with ATT&CK Data
The SOC and Incident Response Teams may gather data on specific techniques and tactics used in active incidents. This ATT&CK-based data can be fed back into the strategic CTI program to:
- Adjust priorities based on real-world observed techniques.
- Update risk registers with specific ATT&CK techniques tied to your organization’s threat model.

Example: If a SOC identifies Credential Dumping (T1003) during an attack, this can help the strategic CTI team understand that attackers may focus on exploiting credential storage weaknesses in the future. This could lead to strategic investments in strengthening Multi-factor Authentication (MFA) across critical systems.

## 5. Alignment with CISO & Executive-Level Reporting
For CISOs and the executive team, integrating MITRE ATT&CK helps provide clarity on the organization’s defense against sophisticated threats. By presenting risk analysis in the form of ATT&CK techniques, you can:
- Show the organization's threat profile based on the MITRE ATT&CK matrix.
- Discuss resilience gaps in terms of specific adversary TTPs and their impact on your organization's strategic assets.

## Example Use Cases
### Example 1: Vulnerability Management
- Initial Access (T1071.001) – Application Layer Protocols (e.g., web traffic)
- Vulnerability management teams can monitor and patch vulnerabilities that could be exploited by adversaries to gain Initial Access. For instance, the adversary may exploit a vulnerable web application or public-facing server.
- Strategic Action: Align your patching strategy with MITRE ATT&CK by prioritizing vulnerabilities tied to the Initial Access techniques. Utilize CVEs tied to T1071 techniques that might be actively exploited.

| **Strategic Input**              | **MITRE ATT&CK Technique**                | **Usage**                                                       |
|-----------------------------------|------------------------------------------|-------------------------------------------------------------|
| **Vulnerability Management**      | **Exploitation of Public-Facing Applications** (T1071.001) | Prioritize patching of web application vulnerabilities based on real-time intelligence (CVEs related to T1071). |
| **Business Risk Analysis**        | **Drive-by Downloads** (T1071.002)       | Focus efforts on vulnerabilities that can lead to **Drive-by Downloads**, enhancing website defenses. |


### Example 2: SOC Data Integration
- Lateral Movement (T1021.001) – SMB/Windows Admin Shares
- SOC teams often see attackers trying to move laterally across the network. By using ATT&CK to map these movements, SOC analysts can detect abnormal activity tied to Windows Admin Shares.
- Strategic Action: Strengthen internal network segmentation and monitor for lateral movement techniques that could be used to pivot across critical systems.

| **SOC Data**                       | **MITRE ATT&CK Technique**               | **Strategic Action**                                               |
|-------------------------------------|-----------------------------------------|-------------------------------------------------------------------|
| **Lateral Movement Alerts**         | **SMB/Windows Admin Shares** (T1021.001) | Use network segmentation to prevent lateral movement and focus detection on SMB traffic. |
| **Incident Analysis**               | **Pass-the-Hash** (T1075)                | Focus on **Pass-the-Hash** techniques to limit credential reuse across critical systems. |

### Example 3: Incident Response Integration
- Persistence (T1070) – Indicator Removal from Tools
- Incident response teams may observe evidence of adversaries deleting logs or clearing indicators of compromise (IoCs). MITRE ATT&CK provides specific techniques to detect these activities (e.g., T1070).
- Strategic Action: Ensure the log management strategy includes tamper detection mechanisms and alerts on suspicious deletion or overwriting activity.

| **Incident Response**             | **MITRE ATT&CK Technique**                | **Strategic Action**                                                 |
|-----------------------------------|------------------------------------------|---------------------------------------------------------------------|
| **Incident Data**                 | **Indicator Removal from Tools** (T1070)  | Implement tamper-resistant logging and monitor for log deletion activity. |
| **Root Cause Analysis**           | **Process Injection** (T1055)            | Focus defensive strategy on detecting **Process Injection** to prevent further escalation of attacks. |


### Example 4: Executive Risk Reporting
- Credential Dumping (T1003) – NTLM Hash Dumping
- Executive-level discussions on cybersecurity risk can focus on Credential Dumping techniques such as NTLM Hash Dumping. By integrating MITRE ATT&CK into risk reporting, the CISO can clearly show how credential theft contributes to wider organizational risks.
- Strategic Action: Highlight risks related to Credential Dumping and the need for multi-factor authentication (MFA) to prevent unauthorized access.

| **Executive Risk**                | **MITRE ATT&CK Technique**               | **Risk Implication**                                               |
|-----------------------------------|-----------------------------------------|--------------------------------------------------------------------|
| **Credential Theft**              | **NTLM Hash Dumping** (T1003)           | Implement **MFA** to mitigate risks associated with **NTLM Hash Dumping** and credential theft. |
| **Lateral Movement**              | **Remote Desktop Protocol** (T1076)     | Prioritize securing remote access points and monitoring RDP sessions to prevent lateral movement. |


