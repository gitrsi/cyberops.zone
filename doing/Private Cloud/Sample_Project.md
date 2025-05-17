# Sample Project

## Project Directory Structure

```
private-cloud/
|-- terraform/
|   |-- main.tf
|   |-- variables.tf
|   |-- outputs.tf
|   |-- cloud-init.tpl
|   |-- terraform.tfvars
|   '-- vault_provider.tf              # Vault integration here
|
|-- ansible/
|   |-- inventories/
|   |   '--- static.ini
|   |-- playbooks/
|   |   |-- site.yml
|   |   '--- roles/
|   |       |-- common/
|   |       |-- webserver/
|   |       '--- monitoring/
|   '-- vault_lookup.yml               # Example of using Vault in Ansible
|
|-- vault/
|   |-- vault-policy.hcl               # Vault policy for Terraform/Ansible access
|   |-- secrets/                       # Structure for storing secrets
|   |   '--- proxmox/
|   |       '--- root-creds            # Stored as key-value in Vault
|   '-- README.md                      # Vault usage notes
|
|-- .gitignore
'-- README.md
```

## Terraform Example

### terraform/main.tf

```
provider "proxmox" {
  pm_api_url      = "https://192.168.1.10:8006/api2/json"
  pm_user         = "root@pam"
  pm_password     = var.pm_password
  pm_tls_insecure = true
}

resource "proxmox_vm_qemu" "vm01" {
  name        = "vm01"
  target_node = "proxmox1"
  clone       = "ubuntu-template"  # Create a template VM with cloud-init first
  cores       = 2
  memory      = 2048
  disk {
    size    = "20G"
    type    = "scsi"
    storage = "local-lvm"
  }
  network {
    bridge = "vmbr0"
    model  = "virtio"
  }

  ipconfig0 = "ip=192.168.1.100/24,gw=192.168.1.1"
  sshkeys   = file("~/.ssh/id_rsa.pub")

  cloudinit_cdrom_storage = "local-lvm"
}
```
### terraform/variables.tf

```
variable "pm_password" {
  type        = string
  description = "Proxmox root password"
}
```

### terraform/cloud-init.tpl

```yaml
#cloud-config
hostname: ${hostname}
users:
  - name: ubuntu
    ssh-authorized-keys:
      - ${ssh_key}
    sudo: ['ALL=(ALL) NOPASSWD:ALL']
    shell: /bin/bash
```

## Ansible Example

### Ansible Inventory

#### ansible/inventories/static.ini

```
[web]
192.168.1.100 ansible_user=ubuntu ansible_ssh_private_key_file=~/.ssh/id_rsa
```

### Ansible Playbook

#### ansible/playbooks/site.yml

```yaml
- name: Configure all servers
  hosts: all
  become: yes
  roles:
    - common

- name: Configure webserver
  hosts: web
  become: yes
  roles:
    - webserver
```
### Role

#### roles/common/tasks/main.yml

```yaml
- name: Update and upgrade apt packages
  apt:
    update_cache: yes
    upgrade: dist

- name: Install base packages
  apt:
    name:
      - curl
      - htop
      - ufw
    state: present
```

## HashiCorp Vault Integration



# Usage

Install dependencies:
- Terraform, Ansible and SSH keys
- Set up your Proxmox server and cloud-init template

Initialize and apply Terraform:
```bash
cd terraform
terraform init
terraform apply
```
Run Ansible:
```bash
cd ../ansible
ansible-playbook -i inventories/static.ini playbooks/site.yml
```