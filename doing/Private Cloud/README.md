
![Private Cloud](images/private_cloud.jpg "Private Cloud")

> :bulb: Private Cloud"


# Architecture

## Physical Layer (Bare Metal)

### Server

High-performance x86_64 machineswith AMD-V hardware virtualization 

### Network

Dedicated private LAN

### Storage

Local SSD

### Host OS

Proxmox VE

## Virtualization Platform Proxmox VE

- KVM for virtual machines
- LXC for lightweight containers (optional)
- Built-in cluster management for multiple Proxmox nodes
- ZFS support for snapshotting, replication
- Software-defined networking OVS or Linux bridges

## Infrastructure as Code & Configuration Management

### Terraform (Infrastructure as Code)
- Terraform Proxmox provider
- Manage:
    - Virtual machines (VMs)
    - Networks
    - Storage pools
    - Proxmox users, roles, and permissions
- Terraform state stored locally or remotely (e.g., git or S3-compatible service)

Example:

```
provider "proxmox" {
  pm_api_url = "https://proxmox.local:8006/api2/json"
  pm_user    = "root@pam"
  pm_password = var.pm_password
  pm_tls_insecure = true
}

resource "proxmox_vm_qemu" "web_vm" {
  name = "web01"
  target_node = "proxmox1"
  ...
}
``` 
### Ansible (Configuration Management)
- Provision base configuration on VMs (e.g., packages, users, ssh)
- Set up services (e.g., web servers, databases, internal DNS)
- Manage updates, hardening, firewall rules (ufw/iptables), and monitoring agents
- Can be triggered post-Terraform deployment or via cloud-init

Example Ansible roles:
- common/: basic packages, timezone, users
- application/: nginx, certbot, firewall
- monitoring/: Prometheus node exporter, logging

## Network Design
- Internal VLANs:
    - mgmt: for Proxmox API, SSH, monitoring
    - infra: for internal services (DNS, NTP, etc.)
    - vm-net: for VM traffic
- Optional external interface for NAT/gateway access
- Use Linux bridges (vmbr0, vmbr1) or Open vSwitch for Proxmox networking

## Storage Design
- Local ZFS on SSD/NVMe for fast VMs and snapshots
- Ceph cluster for distributed block storage
- NFS for shared data volumes.

## Secrets Management
- HashiCorp Vault for secrets management.

## Monitoring & Maintenance
- ELK stack for logging
- Grafana dashboards
- Zabbix for monitoring
- Prometheus + Node Exporter for metrics.

## GitOps Automation

### Git repo
Terraform modules and state
Ansible roles/playbooks
CI pipeline for validation/linting

### Automation Workflow
- Update Git (infra or config)
- Trigger Terraform plan
- Run Ansible playbook on updated VMs

# Setup

## Preparation

### Requirements
- Number of physical servers
- VM types and workloads
- Storage needs (ZFS, Ceph, NFS?)
- Network layout (VLANs, bridges)
- Security requirements (VPN, firewall, 2FA)

### Prepare Your Network
- Set static IPs for servers
- Configure VLANs
- Plan for management and VM networks (e.g., vmbr0, vmbr1)

## Installation

### Base System
- Install Proxmox VE (https://www.proxmox.com/en/downloads)
- Configure Proxmox VE
- Set up web UI access

Configure:
- Storage (ZFS if desired)
- Linux bridges for networking
- Enable Proxmox API access and generate credentials for Terraform

### Infrastructure Provisioning with Terraform

#### Set Up Terraform
- Install Terraform CLI
- Add the Proxmox provider

#### Create Terraform Modules
Define modules for:
- VM creation (proxmox_vm_qemu)
- Storage config
- Networking (bridge assignments)

#### Terraform Project Layout

```
terraform/
|-- main.tf
|-- variables.tf
|-- outputs.tf
|-- modules/
|   |-- vm/
|   '-- network/
```
#### Deploy VMs

```bash
terraform init
terraform plan
terraform apply
```

### VM Configuration with Ansible

#### Prepare Ansible
- Install Ansible
- Create roles for:
    - Base system setup (users, ssh, updates)
    - Services (e.g., Nginx, Docker, apps)
- Inventory can be dynamic or static from Terraform outputs

#### Use Cloud-Init or Ansible for Initial Bootstrapping
- Add cloud-init templates in Terraform

#### Run Ansible Playbooks

Example:

```bash
ansible-playbook -i inventory.ini site.yml
```

### Automate & Maintain
- Set up Git repo (IaC and Ansible)
- Use GitHub Actions/GitLab CI to run Terraform & Ansible


# Resources
| What | URL | Description |
| ----------- | ----------- | ----------- |
| xxx | https |  |
| xxx | https |  |
| xxx | https |  |
| xxx | https |  |
| xxx | https |  |
| xxx | https |  |
| xxx | https |  |

