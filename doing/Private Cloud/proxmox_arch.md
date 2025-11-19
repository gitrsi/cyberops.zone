--![Private Cloud](images/proxmox_architecture.jpg "Private Cloud")

# Proxmox Detailed Architecture
On-prem admin zone is the trusted control plane; the cloud is untrusted and strongly segmented.

```
                           ON-PREM ADMIN ZONE (CONTROL PLANE)
┌────────────────────────────────────────────────────────────────────────────┐
│  Git       Terraform      Ansible          Vault            Boundary       │
│   │           │             │                │                 │           │
│   │           │             │                │                 │           │
│   └───────┬───┴────────┬────┴────────┬───────┴───────┬─────────┴───────────│
│           │            │             │               │                     │
│       Admin Jump Host ─┴─────────────┴───────────────┘                     │
│          (SSO/MFA)                                                         │
│                                                                            │
│                             Proxmox Backup Server (PBS)                    │
└────────────────────────────────────────────────────────────────────────────┘
                     |   OUTBOUND ONLY (ZT tunnels, API access)   |
                     v                                            v
===============================================================================
                          PRIVATE CLOUD (PROXMOX CLUSTER)
===============================================================================
   SEGMENT A: PROXMOX MGMT           SEGMENT C: CORE INFRA
 ┌─────────────────────────┐     ┌─────────────────────────────────────┐
 │ Proxmox UI/API          │     │ IAM                                 │
 │ Ceph mgmt               │     │ DNS/DHCP                            │
 │ SDN Controllers         │     │ Vault Agent (for workload mTLS)     │
 └─────────────────────────┘     │ Internal PKI Clients                │
            |                    └─────────────────────────────────────┘
            |                                    |
            v                                    v
   SEGMENT B: SECURITY SERVICES      SEGMENT D: OBSERVABILITY
 ┌─────────────────────────┐     ┌─────────────────────────────────────┐
 │ SIEM                    │     │ Logging                             │
 │ SOAR                    │     │                                     │
 │ CTI                     │     │                                     │
 └─────────────────────────┘     └─────────────────────────────────────┘
                 SEGMENT E: WORKLOAD / APPLICATION MICROSEGMENTS
                 (per-app VXLAN / OVN with default-deny policies)
===============================================================================
```

All access from on-prem -> cloud goes through Boundary, optionally using Vault-issued short-lived credentials.


# Network Segmentation + Firewall Policy Matrix
## VLAN / Zone Layout

| Zone                          | Description                                         | Trust Level |
| ----------------------------- | --------------------------------------------------- | ----------- |
| **ON-PREM-ADMIN**             | Git, Terraform, Ansible, HashiCorp Vault, HashiCorp Boundary, Jump Host | High        |
| **PROXMOX-MGMT**              | Proxmox API/UI, Ceph mgmt                           | Medium      |
| **CORE-INFRA**                | IAM, DNS, NTP, internal CA                          | Medium      |
| **SECURITY-SERVICES**         | SIEM, SOAR, CTI                                | Low/Medium  |
| **OBSERVABILITY**             | Logging, Metrics, Tracing                           | Low/Medium  |
| **WORKLOAD-ZONES (MULTIPLE)** | Individual app microsegments                        | Low         |

## Firewall Matrix (Zero Trust Default Deny)

Legend
-> Allowed
X Blocked
0 Only via Boundary
E Only with mTLS (Vault)

| SRC -> DST            | Proxmox Mgmt               | Core Infra            | Security Services                  | Observability                     | Workloads                    |
| --------------------- | -------------------------- | --------------------- | ---------------------------------- | --------------------------------- | ---------------------------- |
| **ON-PREM ADMIN**     | 0 Proxmox API only         | 0 IAM token endpoints | 0 Config API only                  | 0 Metrics dashboards via Boundary | 0 Only for provisioning      |
| **Boundary**          | -> Proxmox API, Jump to VMs | -> IAM (OIDC), Vault   | -> SIEM/SOAR admin ports            | -> Observability dashboards        | -> Controlled access          |
| **Vault**             | -> Issue certs (mTLS)       | -> PKI distribution    | X                                  | X                                 | -> Issue per-VM/service certs |
| **Proxmox Mgmt**      | -> Ceph                     | -> DNS/NTP             | X                                  | X                                 | X                            |
| **Security Services** | X                          | -> IAM for auth        | -> SIEM/SOAR/CTI East-West (narrow) | -> Logging                         | X                            |
| **Core Infra**        | -> DNS/NTP                  | internal only         | -> Auth for SIEM/SOAR               | -> Logging/metrics                 | -> IAM token validation       |
| **Workloads**         | X                          | -> IAM/DNS             | -> Send logs to SIEM                | -> Send metrics                    | -> Only same microsegment     |


# Integration Guide (How Boundary, Vault, Terraform, Ansible Work Together)
## 1. Git -> Terraform / Ansible (IaC Workflow)
Git contains:
- Terraform code to create VMs, networks, SDN
- Ansible roles for configuration
- Boundary config (targets, roles)
- Vault policies

## 2. Terraform -> Vault
Terraform retrieves secrets dynamically:
- Vault`s Proxmox secret engine (API creds)
- Vault PKI to create short-lived mTLS for each VM
- Vault KV for encrypted static secrets (rare)

Terraform Provider Example
- provider "vault" { address = ... }
- data "vault_kv" "proxmox_credentials"

## 3. Terraform -> Proxmox (via Boundary or direct restrict)

Terraform deploys:
- VMs
- Networks
- Tags
- Firewall rules

Access path:
Terraform -> Boundary (for ZT identity) -> Proxmox API
OR
Terraform -> Proxmox API (direct, IP-restricted, mTLS)

## 4. Ansible -> Boundary -> VMs

Boundary grants:
- TCP session brokering (SSH, RDP, WinRM)
- No network layer access

Ansible runs:
- ansible_connection: ssh
- SSH key issued from Vault (SSH CA) or ephemeral Boundary credentials

## 5. Boundary -> Vault

- Boundary asks Vault for:
- Dynamic DB credentials
- Ephemeral SSH certs
- mTLS certs for service identities
- Revocation on session end

## 6. Vault -> Workloads

Agents in cloud VMs:
- auto-renew mTLS certs
- authenticate via JWT/OIDC to Vault
- fetch secrets if allowed

## 7. PBS -> Proxmox Cloud

Strictly backup traffic:
- PBS pulls backups via Proxmox Backup protocol
- No cloud-initiated traffic

# Recommended Git Repository Layout
Git (monorepo model — simple & highly recommended):
```
infrastructure/
├── terraform/
│   ├── global/
│   ├── proxmox/
│   ├── network/
│   ├── security-services/
│   └── modules/
│
├── ansible/
│   ├── inventories/
│   ├── roles/
│   │   ├── siem/
│   │   ├── soar/
│   │   ├── cti/
│   │   ├── iam/
│   │   └── base/
│   ├── playbooks/
│   └── group_vars/
│
├── boundary/
│   ├── scopes/
│   ├── roles/
│   ├── targets/
│   └── workers/
│
├── vault/
│   ├── policies/
│   ├── pki/
│   ├── secret_engines/
│   └── auth_methods/
│
└── docs/
    ├── architecture/
    ├── threat-model/
    └── procedures/
```
