![Industrial Cyber Security Basics](images/icsb.jpg "Industrial Cyber Security Basics")

> :bulb: Notes on "Industrial Cyber Security Basics"


# Industrial Cyber Security Basics

## Introduction

### Shodan IO
Explore Industrial Control Systems
https://www.shodan.io/search?query=port%3A102+Siemens

Images
https://images.shodan.io/?query=screenshot.label%3A%22ics%22

## Grundlagen IT und OT

### Safety vs. Security

Safety
- Funktionale Sicherheit
- Betriebssicherheit
- Schützt Umgebung vor dem Sytem

Security
- Informationssicherheit
- Angriffssicherheit
- Schützt das System vor der Umgebung

### IT vs. OT
#### OT Operational Technology
Hardware und Software zur (echtzeitfähigen) Erfassung/Regelung/Steuerung eines (industriellen) Prozesses
- SPS
- HMI
- Servocontroller
- Greifroboter
- Visionsystems
- Webserver

![AWS Essentials](images/icsb1.png)

Einsatzdauer der Hardware
IT: 3-5 Jahre
OT: bis 20 Jahre

Verfügbarkeit und Performance
IT: Ausfälle tolerierbar, weiche Echtzeit
OT: Keine Ausfälle, harte Echtzeit

Updates und Patches:
IT: regelmässig und automatisiert
OT: Nie...selten

Security Tools (AV, FW, Backup)
IT: etabliert
OT: selten

Security Bewusstsein
IT: gut
OT: eher schlecht, wachsend


### Schutzziele IT vs. OT
- Confidentiality
- Integrity
- Availability

IT Systeme
Prio: C -> I -> A

OT Systeme:
Prio: A -> I -> C


### Datenverkehr im Netzwerk
![AWS Essentials](images/icsb2.png)

### Verkapselte Feldbusprotokolle
![AWS Essentials](images/icsb3.png)

### Normen und Richtlinien
IT
- ISO 27000
- IT Grundschutz (BSI)

OT
- IEC 62443
- ICS-Security Kompendium (BSI)

Rollen IEC 62443-3-3
![AWS Essentials](images/icsb4.png)

Level des Angreifers
- SiL 1-4 (Safety)
    - Technologie
    - Funds
    - Skills
    - Motiv

![AWS Essentials](images/icsb5.png)

Security Levels
SL-T: Target Level (gefordert)
SL-C: Capable Level (theoretisch erreichbar)
SL-A: Achieved Level (effektiv, durch Audit ermittelt)

Systemanforderungen IEC 62443

![AWS Essentials](images/icsb6.png)







```
sudo netdiscover -r 192.168.56.0/24

sudo nmap 192.168.56.100 ... -Pn -p 1-65535 -T5

```

### ...







![AWS Essentials](images/icsb7.png)
![AWS Essentials](images/icsb8.png)
![AWS Essentials](images/icsb9.png)


![AWS Essentials](images/icsb10.png)
![AWS Essentials](images/icsb11.png)
![AWS Essentials](images/icsb12.png)
![AWS Essentials](images/icsb13.png)
![AWS Essentials](images/icsb14.png)
![AWS Essentials](images/icsb15.png)
![AWS Essentials](images/icsb16.png)
![AWS Essentials](images/icsb17.png)
![AWS Essentials](images/icsb18.png)
![AWS Essentials](images/icsb19.png)


![AWS Essentials](images/icsb20.png)
![AWS Essentials](images/icsb21.png)
![AWS Essentials](images/icsb22.png)
![AWS Essentials](images/icsb23.png)
![AWS Essentials](images/icsb24.png)
![AWS Essentials](images/icsb25.png)
![AWS Essentials](images/icsb26.png)
![AWS Essentials](images/icsb27.png)
![AWS Essentials](images/icsb28.png)
![AWS Essentials](images/icsb29.png)

