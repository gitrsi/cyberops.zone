# Minisforum UM890 Pro Proxmox VE Setup Guide

## 1. BIOS Configuration: Enable Virtualization & IOMMU

1. **Enter BIOS/UEFI**  
   - On boot, press `DEL` or `F2` (check your manual).

2. **Enable SVM (AMD Virtualization):**  
   - Navigate to `Advanced` &rarr; `CPU Configuration` &rarr; `SVM Mode` &rarr; set to **Enabled**

3. **Enable IOMMU:**  
   - `Advanced` &rarr; `AMD IOMMU` &rarr; **Enabled**  
   (Important for PCI passthrough, improves device virtualization)

4. **Disable Secure Boot:**  
   - `Boot` &rarr; `Secure Boot` &rarr; set to **Disabled** (Proxmox doesn't support Secure Boot by default)

5. **Set Boot Mode to UEFI:**  
   - `Boot` &rarr; `Boot Mode` &rarr; select **UEFI only**

6. **Adjust Power Settings (Optional but recommended):**  
   - Disable CPU C-States or set them to a minimum to prevent sleep-related instability under VM workloads.

7. **Save and Exit BIOS**

## 2. Prepare Installation Media

- Download the latest stable Proxmox VE ISO from:  
  [https://www.proxmox.com/en/downloads/category/iso-images-pve](https://www.proxmox.com/en/downloads/category/iso-images-pve)

- Create a bootable USB with [balenaEtcher](https://www.balena.io/etcher/) or `dd` command.

## 3. Install Proxmox VE

1. Boot UM890 Pro from USB installer
2. Follow installation prompts:
   - Select target disk (1TB NVMe)
   - Choose hostname and network config (static IP recommended)
   - Set strong root password and email
3. Reboot when done, remove USB

## 4. Post-Installation Setup

### Access Web UI  
- From another PC on your private network, go to:  
  `https://<Proxmox-IP>:8006`  
- Login as root with your password

### Update System  
```bash
apt update && apt full-upgrade -y
reboot
```

## 5. Storage Setup

- Use local-lvm on NVMe for VM disks by default
- Optionally add external NVMe or USB storage as ZFS pool for backups and redundancy

    Example to create ZFS pool:

    ```bash
    zpool create -f mypool mirror /dev/nvme1n1 /dev/nvme2n1
    zfs create mypool/vms
    ```
- Add this pool in Proxmox Storage GUI

## 6. Network Configuration Tips

- Confirm 2.5GbE NIC is detected:

    ```bash
    ip a
    ```
- Set up a Linux Bridge in Proxmox UI for VM network connectivity (usually vmbr0)
- Optionally configure VLANs or bond interfaces if you add more NICs

## 7. Optimize Proxmox for K3s + Multiple VMs

- Enable Nested Virtualization for VMs running containers:
    ```bash
    cat /sys/module/kvm_amd/parameters/nested
    ```
    If not Y, enable with:
    ```bash
    echo "options kvm-amd nested=1" >> /etc/modprobe.d/kvm-amd.conf
    update-initramfs -u -k all
    reboot
    ```
- Tune hugepages and CPU pinning if needed for performance (optional)

## 8. Backup and Security

- Set up scheduled backups in Proxmox for VMs/containers
- Configure firewall with Proxmox GUI or ufw/iptables
- Consider integrating HashiCorp Vault later for secret management
