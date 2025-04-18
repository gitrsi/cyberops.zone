![Zero Trust Security Fundamentals for IT Professionals](images/zt.jpg "Zero Trust Security Fundamentals for IT Professionals")

> :bulb: Notes on "Zero Trust Security Fundamentals for IT Professionals"

# Zero Trust Security Fundamentals for IT Professionals

## Resources
- https://www.udemy.com/course/zero-trust-security-fundamentals-for-it-professionals/

## Zero Trust Fundamentals

Zero Trust is
- security model, strategy and framework
    - Never trust, always verify
    - Assume breach
    - Verify explicitly
    - Least privileged access

Assumptions
- Network is assumed to be hostile
- External and internal threats are always present
- Network locality isn't sufficient for determining trust
- Every single device, user and network flow is authenticated and authorized with dynymic policies


Never trust, always verify
- Trust isn't implicit
- Trust is a vulnerability
- every device, user, request is treated as potential threat
- just-in-time and just-enough-access least privilege access controls


Zero trust architecture (ZTA)
- cybersecurity plan to utilize zero trust concepts
- encompasses component relationships, workflow planning and access policies

$$
Zero Trust Enterprise = Zero Trust (ZT) + Zero trust architecture (ZTA)
$$

7 Tenets of Zero Trust Architecture
- consider every data source and computing device as a resource
- keep all communication secured, regardless of network location
- grant resource access on a per-session basis
- moderate access with a dynamic policy
- maintain data integrity
- rigorously enforce authentication and authorization
- gather data for improved security

5 Pillars of Zero Trust
- Users & Identity
    - identification, authentication and access control policies
- Devices
    - validation to determince acceptable trustworthiness
- Network & Environment
    - segment, isolate, control the network environment
- Application & Workloads
    - secure everything from applications to hypervisors, vms and containers
- Data
    - secure and enforce access control
    - data categorization and classification
    - data isolation

2 Foundations of Zero Trust
- Visibility & Analytics
    - provide insight into user and system behavior
    - observe real-time communication
- Automation & Orchestration
    - automate security and network operation
    - orchestrate functions between systems and applications

![Zero Trust Security Fundamentals for IT Professionals](images/zt1.png)

## Why We Need Zero Trust

Perimeter security pittfalls
- Outside-in
- trust by verify approach
- static policies
- insider threats
- lateral movement

Digital transformation
- Cloud
- Remote/hybrid work
- BYOD
- Blurred traditional network boundaries
- Complex IT infrastructure environments
- Increased attack surface
- Attackers shifting to identity based attacks

Forrester Case study
- 92% ROI
- 50% reduction of data breach risk
- $11.6M net benefits
- less than 6 months payback

Gartner survey reported benefits
- 75% reported improved risk management
- 65% reported improved secure remote access
- 41% reported a reduced nuber of IT security incidents
- 34% reported a reduced network complexity
- 26% reported lower overall security costs

Gartner survey adoption concerns
- 56% Cost
- 51% Technology gaps
- 51% Skills gaps
- 39% shadow IT
- 38% BYOD

## Zero Trust Architecture (ZTA) Fundamentals

### NIST Zero Trust Architectural Model

![Zero Trust Security Fundamentals for IT Professionals](images/zt2.png)

- Untrusted requestor
    - requestor is untrusted by default
- Policy enforcement point (PEP)
    - enables, monitors and terminates connections between a subject and a resource
- Policy administrator (PA)
    - executes the policy engine's decision by signaling it to the PEP
- Policy engine (PE)
    - brain of the model
    - inputs signals and compares them with access policies to determine wheter access should be granted
- Policy decision point (PDP)
    - conjuction of PE and PA

Data sources
- Continuous diagnostics and mitication (CDM) system
    - collect information regarding systems to determine their current state and apply configuration and sofware updates as needed
- Industrial compliance
    - ensures the enterprise remains compliant with regulatory requirements
- Threat intelligence
    - information regarding newly discovered attacks and vulnerabilities helping to unterstand threats
- Activity logs
    - aggregated real-/near time information on the security posture
- Data access policy
    - attributes, rules and policies to determine how access is granted on trusted resources
- PKI
    - generating and logging certificates
- ID management
    - responsible for creating, storing and managing accounts and identiy records
- SIEM system
    - collect, aggregate and analyze security-centric information

### NIST ZTA architecture approaches
- Enhanced identity governance
    - identity as main source of policy creation
- Micro-segmentation
    - network segments to protect resources
- Software defined perimeters
    - commonly used in the cloud

### NIST ZTA deployment models
#### Device agent/gateway
- PEP agent is deployed on all enterprise systems
- PEP agent communicates with the PA
- if approved, the PA establishes a communication channel between the user agent and the resource gateway

![Zero Trust Security Fundamentals for IT Professionals](images/zt4.png)

#### Enclave based
- variation of agent/gateway model
- gateway protects several resources instead of one -> resource enclave

![Zero Trust Security Fundamentals for IT Professionals](images/zt5.png)

#### Resource portal
- Agentless (no PEP)
- Gateway web portal to access resources
- Access to single source or enclave

![Zero Trust Security Fundamentals for IT Professionals](images/zt6.png)

### Examples
#### Netskope private access

![Zero Trust Security Fundamentals for IT Professionals](images/zt7.png)

#### Microsoft's internal ZTA

![Zero Trust Security Fundamentals for IT Professionals](images/zt3.png)

### Trust algorithms & Policies Fundamentals
#### Overview
- PE ist the brains of the PDP
- PE uses trust algorithms to determine whether to grant or deny access
- PE inputs from multiple data sources
- PE leverages policy database
    - observable information about subjects
    - subject attributes and roles
    - historical subject behavior patterns
    - threat intelligence sources
    - other metadata sources

![Zero Trust Security Fundamentals for IT Professionals](images/zt8.png)

#### Trust algorithm inputs

![Zero Trust Security Fundamentals for IT Professionals](images/zt9.png)

#### Attribute based access controls (ABAC)
- role-based: based on group membership
- attribute-based: based on attributes and information from multiple data sources
- ZT uses a combination of both, providing dynamic and contextual information

![Zero Trust Security Fundamentals for IT Professionals](images/zt10.png)

#### Kipling method for developing policies
- who is requesting?
- what application is used to access?
- when is the request happening?
- where is the requestor requesting access from?
- why is the requestor requesting access?
- how should the requestor be allowed to access?

![Zero Trust Security Fundamentals for IT Professionals](images/zt11.png)

## Zero Trust Architectural Pillars
![Zero Trust Security Fundamentals for IT Professionals](images/zt12.png)
### Users & Identity
Focuses on user identification, authentication and access control policies using dynamic and contextual data analysis.

- User inventory (Centralized identity and access management (IAM)
- Multi-factor authentication (Behavioral, contextual, biometrics)
- Least privileged access (Just-in-time and just-enough-access)
- Privileged access management (PAM)

### Devices
Performs validation of user-controlled and autonomous devices to determine acceptable cybersecurit posture and trustworthiness.
- Device inventory (IT asset management ITAM)
- Patch management
- Endpoint detection and response (EDR)
- Mobile device management (MDM)

### Network & Environment
Segments, isolates and controls the network environment with granular policy and access controls.

- Data flow mapping
- Micro-segmentation
- Software defined perimeter (SDP), effective replacement for VPN that enables micro-segmentation
- Encryption

### Application & Workloads
Secures everything from applications to hypervisors, including containers and virtual machines.
- Application inventory
- Secure software development
- Continuous monitoring & ongoing authorizations (monitoring and re-authorize access)
- Workload isolation

### Data
Focuses on securing and enforcing access to data based on a data's categorization and classification to isolate the data from everyone exept those that need access.

- Data classification (inventory, classify and label)
- Data encryption
- Data loss protection (DLP)
- Data access control (Just-in-time and just-enough-access)

### Foundational components

#### Visibility & Analytics
Provide insight into user and system behavior by observing real-time communications between all Zero Trust components.

- Log all traffic
- Continuous monitoring
- Threat intelligence
- Security Information & Event Management (SIEM)

#### Automation & Orchestration
Automates security and network operational processes across the ZTA by orchestrating functions between similar and disparate security systems and applications.

- Machine learning & artificial intelligence (ML & AI)
- Security orchestration, automation & response (SOAR)
- Policy decision point (PDP) orchestration
- Security operations center (SOC) & incident response (IR) integration

## Designing a Zero Trust Architecture
### No right way to ZT
- strategy and framework
- no two organizations are the same
    - size
    - mission
    - budget
    - resources
    - risk environment
    - regulatory requirements
    - IT architecture
    - cyber security capabilities

### Design principles
- focus on business outcomes
- design from inside out
- determine who and what needs access
- inspect and log all traffic

### Five steps Zero Trust design methodology
![Zero Trust Security Fundamentals for IT Professionals](images/zt13.png)

- Define your protect surfaces
    - identify most critical organizational data, assets, applications, services (DAAS)
    - prioritize based on each protect surface's criticality
- Map transaction flows
    - understand how data flows
    - understand ho DAAS components interact
    - make current state visible
    - design target state
- Design your ZTA
- Create Your ZTA policies
    - who/what needs access
    - Kipling method
- Monitor & maintain

### Forrester's five steps to Zero Trust
![Zero Trust Security Fundamentals for IT Professionals](images/zt14.png)

- Data centric approach
- micro segmentation as a primary aspect
- embrace automation and orchestration

## Migrating to Zero Trust
### Building a Zero Trust business case
- communicates the business value
- obtain executive support

Business benefits:
- seamless supports work from anywhere
- enables secure and rapid cloud migration
- cost savings through simplified security
- improves brand reputation and public trust
- strategic and competitive differentiation

### Challenge of change
- main hurdle is cultural, not technological
- changes the way people work
- shifts the security focus from network to identity
- requires eliminating information technology silos
- enterprise architecture effort requires coordination across business and IT
- resistance to change is inevitable

![Zero Trust Security Fundamentals for IT Professionals](images/zt15.png)

Success factors:
- assess the current state
- clearly define goal
- not a one-person job
- communicate constantly
- benchmark against best-in-class
- be prepared for resistance
- plan for failures
- dedicated Zero Trust team
- focus on low-hanging fruits
- simulate before going into production

### Zero Trust team
- executive champion: CIO, CISO
- project sponsor: Zero Trust program management office
- cross functional working group: IT and business units

## ZTA Use Cases

### VPN-less implementation
![Zero Trust Security Fundamentals for IT Professionals](images/zt16.png)

### East-West segmentation
![Zero Trust Security Fundamentals for IT Professionals](images/zt17.png)

### Secure access from anywhere
![Zero Trust Security Fundamentals for IT Professionals](images/zt18.png)

### Conditional authentication & authorization
![Zero Trust Security Fundamentals for IT Professionals](images/zt19.png)

### Microsoft ZTA step-by-step
![Zero Trust Security Fundamentals for IT Professionals](images/zt20.png)

## Zero Trust Maturity Models

### NSA Zero Trust maturity model
![Zero Trust Security Fundamentals for IT Professionals](images/zt21.png)

### Microsoft Zero Trust maturity model
![Zero Trust Security Fundamentals for IT Professionals](images/zt22.png)

### CISA Zero Trust maturity model
![Zero Trust Security Fundamentals for IT Professionals](images/zt23.png)

### DoD target & advanced Zero Trust maturity model 
![Zero Trust Security Fundamentals for IT Professionals](images/zt24.png)















