![Certificate of Competence in Zero Trust (CCZT)](images/cczt.jpg "Certificate of Competence in Zero Trust (CCZT)")

> :bulb: Notes on "Certificate of Competence in Zero Trust (CCZT)"

# Zero Trust Planning

## Five-Step Process for ZT Implementation
As outlined in the 2022 U.S. National Security Telecommunications Advisory
Committee (NSTAC) Report to the President 1 :
1. Define the protect surface: Identify the data, applications, assets, and services (DAAS) elements to protect.
2. Map the transaction flows: Understand how the networks work by mapping the transaction flows to and from the protect surface, including how various DAAS components interact with other resources on the network. These transaction flows provide insight to help determine where to place proper controls.
3. Build a Zero Trust Architecture (ZTA): Design your ZTA, tailored to the protect surface, determined in steps 1 and 2. The way traffic moves across the network specific to the data in the protect surface determines design. The architectural elements cannot be predetermined, though a good rule of thumb is to place the controls as close as possible to the protect surface. 
4. Create a ZT policy: Instantiate ZT as an application layer policy statement. Use the Kipling Method 2 of ZT policy writing to determine who or what can access your protect surface. Consider person and non-person (services, applications, and bots) entities. 
5. Monitor and maintain the network: Inspect and log all traffic, all the way through the application layer. The telemetry gathered and processed from this process helps prevent significant cybersecurity events and provides valuable security improvement insights over the long term. As a result, each subsequent protect surface can become more robust and better protected over time. 

## CISA High-Level Zero Trust Maturity Model

The five pillars are:
- Identity
- Devices
- Networks
- Applications & Workloads
- Data

The four maturity stages, along with a brief example of their criteria, are as follows:
- Traditional - Utilizes multi-factor authentication (MFA), employs manual deployment of threat protection, and maintains an on-premises network 
- Initial - Implements MFA with passwords, tracks all physical assets, initiates isolation of critical workloads, and employs formal deployment mechanisms through the CI/CD pipeline 
- Advanced - Implements phishing-resistant MFA, tracks most physical and virtual assets, makes most mission-critical apps available over public networks, automates data inventory with tracking, and encrypts data at rest 
- Optimal - Engages in continuous validation and risk analysis, grants resource access based on real-time device risk analysis, establishes distributed micro perimeters with just-in-time (JIT) and just-enough access controls, conducts continuous data inventorying, and encrypts data in use

## Planning Considerations
### Stakeholders
Stakeholders include, but are not limited to:
- Business/service owners
- Application owners
- Infrastructure owners
- Service architecture owners
- CISO/security teams
- Legal officers
- Compliance officers
- Procurement officers

Setup Stakeholder responsibilities
- Responsible, Accountable, Consulted, and Informed (RACI) chart
- Communications plan

Stakeholder Communications
- Communication strategy, including tools and any required guidance
- Establish cadence (e.g., forums, format, etc.)
- Incorporate mechanisms for setting proper expectations with interested parties
- Include a means to communicate and document key decisions

### Technology Strategy
Organizations should be asking themselves the following essential
questions:
- How does the ZT strategy fit into the organization’s technology strategy?
- How does the ZT strategy need to be updated to incorporate the technology strategy?
- How does the ZT strategy impact existing plans, processes, and procedures?
- How does the ZT strategy affect existing budgets and investments?
- How does the ZT strategy affect existing internal standards and best practices?

### Business Impact Assessment
A BIA provides organizations with a list of assets followed by their relative values and owners, valuable information like recovery point objective(RPO) and recovery time objective (RTO), interdependencies and priorities, and an assessment of resources required to restore and maintain each asset. Based on this information, organizations can establish more comprehensive and accurate service level agreements (SLAs), business continuity/ disaster recovery (BC/DR) plans, third-party risk management (TPRM) programs, as well as streamline prioritization and stakeholder identification efforts for ZT planning.

### Risk Register
Have developed a risk register containing an inventory of potential risk events, recorded and tracked by likelihood, impact, and description. The risk register should also contain controls for reducing risk levels within
the organization-defined risk appetite thresholds, along with the risk owner and the control owner.
The risk register will require continuous updating as the organization adopts new technologies and its infrastructure evolves.

### Supply Chain Risk Management
As a result, an organization’s visibility into its supply chain is limited by nature, since many components are outside the organization’s control.

ZT planning considerations should address supply chain risk, since lack of visibility into potential third-party exposures and security glitches.

The National Institute of Standards and Technology (NIST) Special Publication (SP) 800-207 presents ZT tenets that apply to a supply chain and its supplier organizations across all ZT pillars, namely Identity, Devices, Networks, Applications and Workloads, and Data.

Pursue the concept of a software bill of materials (SBOM) as a tool for advancing supply chain risk management.

Additionally, the following non-exhaustive list of tools and resources can help organizations in
determining supply chain risk:
- CSA STAR Program 7 (STAR Level 1 and STAR Level 2)
- ISO 27001 assessments
- SOC 1 and 2 assessments
- Systems audits
- Bridge letters & attestations
- Supplier organization and service offering reputation research

### Organizational Security Policies
The most relevant policies will fit into roughly three categories:
1. Policies that dictate or constrain the ZT initiative
2. Policies that require updating due to ZT
3. Policies that need to be created to support ZT

The following are common policy types for a ZT initiative:
- General IT and security
- ZT
- Data governance
- Cloud
- Key management policy
- Incident response
- User and IAM
- Monitoring
- Disaster recovery (DR)
- Business continuity (BC)

### Architecture
Planners should identify the relevant architecture capabilities and components that could impact ZT or require updating due to ZT.

These capabilities may include architectural frameworks such as The Open Group Architecture Framework (TOGAF), Sherwood Applied Business Security Architecture (SABSA), CSA’s Enterprise Architecture Reference Guide.

### Compliance
U.S. government agencies have produced artifacts that provide critical ZT guidance like the NSTAC Report to the President on Zero Trust and Trusted Identity Management, and the NIST Cyber Security White Paper (CSWP) 20 11 , to name a few. Other jurisdictions like Europe and Asia are also preparing ZT guidance or regulations.

A ZT approach will be helpful in two ways:
- First, it will increase control over regulated data by enforcing controls that foster accountability and by segregating data within dedicated micro-segments. 
- Second, it will drive better overall cybersecurity, which in many cases exceeds most existing legal and regulatory requirements.

### Workforce Training
ZT principles should be integrated into the security awareness program in all phases — onboarding, role changes, yearly reviews, drip feed, and termination.

Special attention needs to be paid to the training of the:
- Staff who determines access controls
- Staff who configures the access control rules
- Support Team, including the Help Desk, who need to be ready to handle the paradigm shift is paramount to a smooth transition
- Staff who audit what has been done, including IT audit and security audit
- Upper management who need to fully embrace the cultural shift that ZT might impose

Finally, it is important that the board of directors and CEO have the necessary level of awareness to be able to fully understand the progress and challenges of the ZT project.

## Scope, Priority, & Business Case


![Cyber Threat Intelligence](images/cczt20.png)
![Cyber Threat Intelligence](images/cczt21.png)
![Cyber Threat Intelligence](images/cczt22.png)
![Cyber Threat Intelligence](images/cczt23.png)
![Cyber Threat Intelligence](images/cczt24.png)
![Cyber Threat Intelligence](images/cczt25.png)
![Cyber Threat Intelligence](images/cczt26.png)
![Cyber Threat Intelligence](images/cczt27.png)
![Cyber Threat Intelligence](images/cczt28.png)

















