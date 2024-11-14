![Industrial Cyber Security Basics](images/icsb.jpg "Industrial Cyber Security Basics")

> :bulb: Notes on "Industrial Cyber Security Basics"


# Industrial Cyber Security Basics

## OT Security

**Operational Technology (OT)** sind in der industriellen Automatisierung und Steuerung (z. B. in Produktionsanlagen, Energieversorgung, Verkehrsinfrastruktur) im Einsatz. Sie unterscheiden sich von klassischen IT-Systemen durch ihre direkte Verbindung mit physischen Prozessen und Maschinen.

Zentrale Aspekte:
1. **Verfügbarkeit vor Vertraulichkeit**:
   - In OT-Systemen steht die **Verfügbarkeit** an erster Stelle, da Unterbrechungen den Betrieb physischer Systeme beeinträchtigen können.
   - *Beispiel*: Ein Produktionsstopp in einer Fabrik kann enorme wirtschaftliche Schäden verursachen.

2. **Sicherheitsziele erweitern**:
   Neben den klassischen IT-Sicherheitszielen (Vertraulichkeit, Integrität, Verfügbarkeit) kommen bei OT folgende hinzu:
   - **Sicherheit von Menschen und Umwelt**: Physische Gefahren, wie Explosionen oder chemische Lecks, müssen verhindert werden.
   - **Prozessintegrität**: Sicherstellung, dass industrielle Prozesse wie geplant ablaufen.

3. **Segmentierung und Isolation**:
   - OT-Systeme sollten möglichst von IT-Netzwerken isoliert werden, um das Risiko eines Übergriffs (z. B. durch Malware) zu minimieren.
   - Einsatz von **Zonen** und **Konduiten** basierend auf Standards wie der **IEC 62443**.

4. **Schwachstellen und Lebenszyklen**:
   - OT-Systeme haben oft eine längere Lebensdauer als IT-Systeme, sodass ältere, nicht gepatchte Software oder Hardware anfällig bleiben kann.
   - Sicherheit muss daher über den gesamten Lebenszyklus des Systems gewährleistet sein.

5. **Physische Sicherheit**:
   - Da OT-Systeme oft direkt mit der physischen Infrastruktur verbunden sind, spielt der Schutz vor physischem Zugriff eine wichtige Rolle.

6. **Incident Response in OT**:
   - Im Falle eines Sicherheitsvorfalls in OT-Systemen müssen die Prozesse zur Vorfallsbehandlung auf die spezifischen Bedingungen der OT abgestimmt sein.
   - Die Priorität liegt darauf, den Betrieb schnellstmöglich wiederherzustellen.


| **IT Security**                     | **OT Security**                     |
|-------------------------------------|-------------------------------------|
| Fokus auf **Daten** (Vertraulichkeit, Integrität) | Fokus auf **Prozesse** (Verfügbarkeit, Sicherheit) |
| Regelmäßige **Updates und Patches** | Updates schwierig wegen **Betriebsunterbrechungen** |
| Schnelllebige Technologien          | **Langzeitbetrieb** von Anlagen     |
| Hauptgefahr: **Datenverlust**       | Hauptgefahr: **Anlagenausfall**, **physische Schäden** |


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

**Security Levels**
In der OT-Sicherheit (Operational Technology) beschreibt die IEC 62443-Norm drei wesentliche Security Levels (SL), um das Sicherheitsniveau in industriellen Steuerungssystemen zu bewerten und zu verbessern:

1. **SL-T (Target Security Level)**
   - **Definition**: Das angestrebte Sicherheitsniveau, das ein System oder eine Zone erreichen soll. Es dient als Planungsziel und berücksichtigt die spezifischen Anforderungen und Risiken des Unternehmens.
   - **Anwendung**: Während der Planung wird ein Ziel-SL festgelegt, um die notwendigen Schutzmaßnahmen zu definieren.

2. **SL-C (Capability Security Level)**
   - **Definition**: Das Sicherheitsniveau, das ein System oder eine Komponente tatsächlich technisch erreichen kann, basierend auf den aktuellen Funktionen und Sicherheitsmaßnahmen.
   - **Anwendung**: Dieses Level hilft, die Fähigkeit der eingesetzten Systeme zu bewerten und eventuelle Lücken zwischen dem Ziel (SL-T) und der aktuellen Kapazität aufzuzeigen.

3. **SL-A (Achieved Security Level)**
   - **Definition**: Das tatsächlich im Betrieb erreichte Sicherheitsniveau. Es wird durch die Implementierung und den Betrieb von Sicherheitsmaßnahmen bestimmt.
   - **Anwendung**: Nach der Implementierung wird überprüft, ob die Maßnahmen ausreichen, um das angestrebte SL-T zu erreichen.

Diese drei Level sind essenziell für die Sicherheitsbewertung und -entwicklung in OT-Umgebungen. Sie ermöglichen es, gezielt Maßnahmen zu planen, die Risiken zu mindern und die Sicherheitsanforderungen systematisch umzusetzen, ohne die Verfügbarkeit der Systeme zu gefährden. Die Bewertung erfolgt im Rahmen von Risikoanalysen und schließt kontinuierliche Überprüfungen ein, um das Sicherheitsniveau an neue Bedrohungen und technologische Entwicklungen anzupassen

Systemanforderungen IEC 62443

![AWS Essentials](images/icsb6.png)


## Offense

### OSINT

#### Google Dorks

[Goolge Dorks Cheatsheet](https://gist.github.com/sundowndev/283efaddbcf896ab405488330d1bbc06)

```
openplc intext:"password"
beckhoff cx intext:"default password"
siemens S7 intext:"default password"
siemens wincc intext:"default password"
SCADA intext:"default password"
```

#### Default Credentials
[SCADA Passwords](https://github.com/scadastrangelove/SCADAPASS/blob/master/scadapass.csv)

#### Shodan IO
[Explore Industrial Control Systems](https://www.shodan.io/search?query=port%3A102+Siemens)
[Shodan Images](https://images.shodan.io/?query=screenshot.label%3A%22ics%22)


### Primär- und Folgeangriffe
Primärangriff auf IT
OT Security geht davon aus, dass der Primärangriff bereits erfolgreich war und befasst sich mit den Folgeangriffen auf die OT Systems.


### MITRE ICS ATT&CK Framework

[ICS Matrix](https://attack.mitre.org/matrices/ics/)

![AWS Essentials](images/icsb7.png)


### Angriffsanalyse auf Wasseraufbereitungsanlage Florida
[A Hacker Tried to Poison a Florida City's Water Supply](https://www.wired.com/story/oldsmar-florida-water-utility-hack/)


### Analyse der Triton/Trisis/Hatman-Malware
[HatMan—Safety System Targeted Malware](https://www.cisa.gov/sites/default/files/documents/MAR-17-352-01%20HatMan%20-%20Safety%20System%20Targeted%20Malware%20%28Update%20B%29.pdf)

[How Does Triton Attack Triconex Industrial Safety Systems?](https://blogs.cisco.com/security/how-does-triton-attack-triconex-industrial-safety-systems)

![AWS Essentials](images/icsb8.png)
![AWS Essentials](images/icsb9.png)


### Analyse der Stuxnet-Malware
[The Real Story of Stuxnet](https://spectrum.ieee.org/the-real-story-of-stuxnet)

![AWS Essentials](images/icsb10.png)

### Angriff Methtode 1
#### Dynamische Systeme
![AWS Essentials](images/icsb11.png)

| **Systemtyp**   | **Potentialgröße**  | **Flussgröße**        | **Verbraucher** | **Energiespeicher (Potential)** | **Energiespeicher (Fluss)** |
|------------------|--------------------|-----------------------|-----------------|---------------------------------|-----------------------------|
| **Elektrisch**   | Spannung           | Strom                 | Widerstand      | Kondensator                     | Spule                       |
| **Translation**  | Kraft              | Geschwindigkeit       | Dämpfer         | Feder                           | Masse                       |
| **Rotation**     | Drehmoment         | Winkelgeschwindigkeit | Reibung         | Torsionsfeder                   | Trägheitsmoment             |
| **Fluid**        | Druck              | Volumenstrom          | Drossel         | Kompressionsblase               | Flüssigkeitsmasse           |
| **Thermisch**    | Temperatur         | Wärmestrom            | Wärmeleiter     | Wärmekapazität                  | —                           |

Erklärungen:
- Potentialgröße: Beschreibt das "Potenzial", das eine Bewegung oder einen Fluss antreibt.
- Flussgröße: Beschreibt den tatsächlichen Fluss der Energie oder Materie.
- Verbraucher: Elemente, die Energie verbrauchen oder umwandeln.
- Energiespeicher (Potential): Speichert Energie basierend auf der Potentialgröße.
- Energiespeicher (Fluss): Speichert Energie basierend auf der Flussgröße (wenn zutreffend).


#### HAZOP Leitworte
HAZOP (Hazard and Operability Study) ist eine systematische Methode zur Identifikation von Gefährdungen und Betriebsproblemen in Prozessen. 
Die Leitworte helfen dabei, mögliche Abweichungen von den Design- oder Prozessbedingungen zu identifizieren.

[HAZOP: Hazard and Operability](https://safetyculture.com/topics/hazop/)

| **Leitwort**    | **Erläuterung**                                | **Beispiele für Abweichungen**            |
|------------------|------------------------------------------------|-------------------------------------------|
| **Kein/Keine**   | Es findet keine oder eine unvollständige Funktion statt. | Kein Durchfluss, keine Reaktion, kein Signal. |
| **Mehr**         | Eine Funktion oder Größe ist höher als vorgesehen. | Erhöhter Druck, zu hohe Temperatur, zu großer Durchfluss. |
| **Weniger**      | Eine Funktion oder Größe ist niedriger als vorgesehen. | Verminderter Durchfluss, zu niedrige Temperatur, unvollständige Reaktion. |
| **Teilweise**    | Eine Funktion wird nur teilweise ausgeführt.   | Nur ein Teil des Produkts erreicht die nächste Stufe, unvollständige Reinigung. |
| **Frühzeitig**   | Eine Funktion tritt früher als vorgesehen ein. | Frühzeitiger Beginn einer Reaktion, frühzeitiger Druckanstieg. |
| **Spät**         | Eine Funktion tritt später als vorgesehen ein. | Verzögerte Reaktion, verzögerter Start eines Prozesses. |
| **Umgekehrt**    | Eine Funktion oder Strömung läuft in entgegengesetzter Richtung. | Rückfluss, Umkehr des Stromflusses. |
| **Anders als**   | Abweichungen von den erwarteten Bedingungen, nicht spezifizierte Änderung. | Falsche Konzentration, ungeeignetes Material, falsches Produkt. |
| **Zusätzlich**   | Es tritt etwas auf, das nicht vorgesehen ist.  | Unerwünschte Nebenprodukte, zusätzliche Wärmeentwicklung. |
| **Zusätzlich zu**| Ein zusätzlicher Prozess oder Zustand, der nicht vorgesehen war. | Zwei Reaktionen gleichzeitig, zusätzliche Lasten. |
| **Kombiniert**   | Kombination von mehreren Abweichungen.         | Erhöhter Druck und Temperatur, Kombination verschiedener Produkte. |


### Angriff Methode 2

Betrachtung der Anwendugsfälle und Ableitung der Missbrauchsfälle.

Szenarien
![AWS Essentials](images/icsb12.png)

Sensormanipulation
![AWS Essentials](images/icsb13.png)


### Pentest Tools

#### netdiscover


#### nmap


#### Wireshark


#### Ettercap


#### Metasploit


#### Metasploitable



## Defense

### Security by Design

Die Integration von Security by Design in OT-Projekten kombiniert Sicherheits- und Engineering-Praktiken und unterstützt dabei, sowohl die Betriebskontinuität als auch die Cybersicherheit zu gewährleisten. Diese Methode wird von führenden Standards und Frameworks wie NIST und IEC 62443 unterstützt, die eine durchgängige Sicherheitsstrategie fördern.

**ICS Security Kompendium (BSI)**

Das *ICS Security Kompendium* des Bundesamtes für Sicherheit in der Informationstechnik (BSI) bietet eine umfassende Einführung in die Cybersicherheit für industrielle Kontrollsysteme (Industrial Control Systems, ICS). Es richtet sich an Betreiber, Integratoren, Maschinenbauer und Hersteller von Steuerungssystemen, die in Operational Technology (OT) eingesetzt werden. Ziel des Dokuments ist es, die Grundlagen der Sicherheit in der OT zu vermitteln und praktische Schritte zur Absicherung von Systemen aufzuzeigen.

Ziele des Kompendiums
- **Grundlagen der Cybersicherheit**: Vermittlung von Basiswissen über Cybersicherheit und Besonderheiten von OT.
- **Prozesse und Maßnahmen**: Unterstützung bei der Etablierung von Sicherheitsprozessen und Umsetzung erster Maßnahmen.
- **Standards und Normen**: Verknüpfung mit relevanten Sicherheitsstandards und Normen.
- **Auditing-Methodik**: Anleitung zur Durchführung von Risikoanalysen und Audits in OT-Umgebungen.
  
Inhalte im Überblick
1. **Einführung in ICS**: Basiswissen über OT und ICS, speziell für IT-Sicherheitsfachleute.
2. **Schwachstellen und Angriffe**: Überblick über typische Bedrohungen, Angriffsmuster und spezifische Schwachstellen in OT.
3. **Organisationen und Standards**: Vorstellung relevanter nationaler und internationaler Standards sowie Best Practices.
4. **Good Practices und Maßnahmenkatalog**: Konkrete Sicherheitsmaßnahmen zur Risikominderung.
5. **Audit- und Kontrollprozesse**: Hilfsmittel für die Bewertung und Sicherstellung der Sicherheit in OT-Systemen.

Praktische Relevanz
Das Kompendium soll Organisationen dabei unterstützen, Sicherheitsrisiken zu identifizieren und durch gezielte Maßnahmen ein akzeptables Restrisiko zu erreichen. Es betont die Notwendigkeit regelmäßiger Risikoanalysen und der Implementierung von Sicherheitsstrategien auf Grundlage bewährter Verfahren.

**CSAF**
Das **Common Security Advisory Framework (CSAF)** ist ein internationaler Standard für die maschinenlesbare Kommunikation und Verteilung von Sicherheitsinformationen. Er wurde von der Organisation OASIS Open entwickelt und zielt darauf ab, die Verarbeitung von Sicherheitswarnungen zu automatisieren. Durch die Nutzung von JSON-Dokumenten ermöglicht CSAF eine schnelle und effiziente Erfassung sowie den Vergleich von Sicherheitsinformationen mit einer Datenbank von IT-Assets oder Software Bills of Materials (SBOMs)&#8203;:contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}.

Hauptmerkmale des CSAF:
1. **Maschinenlesbarkeit**: Sicherheitswarnungen werden in einem standardisierten JSON-Format bereitgestellt, was die automatisierte Analyse und Integration in Sicherheitsprozesse erleichtert.
2. **Profile**: CSAF bietet verschiedene Profile, wie etwa „Security Advisory“ oder „Vulnerability Exploitability eXchange (VEX)“, die je nach Bedarf spezifische Informationen zu Schwachstellen und deren Behebung liefern&#8203;:contentReference[oaicite:2]{index=2}.
3. **Produktstrukturierung**: CSAF erfordert eine klare und hierarchische Struktur von betroffenen Produkten, einschließlich detaillierter Versionsangaben, um einen präzisen Abgleich mit Asset-Datenbanken zu ermöglichen&#8203;:contentReference[oaicite:3]{index=3}.

Das **BSI** unterstützt CSAF und bietet Tools wie den **Secvisogram Editor**, der die Erstellung und Verwaltung von CSAF-Dokumenten erleichtert. Dieses Tool richtet sich an Organisationen, die Sicherheitswarnungen in einem maschinenlesbaren Format erstellen und veröffentlichen möchten&#8203;:contentReference[oaicite:4]{index=4}. 

Weitere Details zu CSAF und dessen Einsatz finden sich in der entsprechenden technischen Richtlinie des BSI (&#8203;:contentReference[oaicite:5]{index=5}.


Referenzen
[Integrated Safety and Security by Design in the IT/OT Convergence of Industrial Systems: A Graph-Based Approach](https://ieeexplore.ieee.org/document/10664281)

[ICS Security Kompendium](https://www.bsi.bund.de/SharedDocs/Downloads/DE/BSI/ICS/ICS-Security_kompendium_pdf.html)
[Common Security Advisory
Framework (CSAF)](https://www.bsi.bund.de/SharedDocs/Downloads/EN/BSI/Publications/TechGuidelines/TR03191/BSI-TR-03191.pdf?__blob=publicationFile&v=5))

#### OT Security im Projektablauf

1. **Akquisitionsphase**
In dieser Phase liegt der Fokus darauf, den Projektauftrag zu gewinnen. Das Unternehmen reagiert auf Anfragen oder Ausschreibungen des Kunden, die oft als Lastenheft formuliert werden. Die Akquisitionsarbeit beinhaltet die Angebotserstellung, bei der technische und kommerzielle Aspekte eng abgestimmt werden. Dies geschieht in Zusammenarbeit von Vertriebsabteilungen und technischen Experten, wie Projektierungsingenieuren. Abschließend führen beide Parteien Vergabeverhandlungen, um den Auftrag zu sichern.

2. **Abwicklungsphase**
Nach Auftragserteilung beginnt die Abwicklung, die die technische Realisierung des Projekts umfasst. Zu den zentralen Schritten zählen:
- **Detail-Engineering**: Hier werden detaillierte Projektierungsunterlagen erstellt, wie das Pflichtenheft.
- **Fertigung und Tests**: Komponenten werden hergestellt und im Factory Acceptance Test (FAT) überprüft.
- **Montage und Inbetriebnahme**: Vor Ort erfolgt die Installation der Anlage, gefolgt vom Site Acceptance Test (SAT), bei dem die Funktionalität im Betriebsumfeld getestet wird.

3. **Servicephase**
Nach der Inbetriebnahme bietet das Unternehmen Serviceleistungen an, um den laufenden Betrieb zu unterstützen. Dazu gehören Wartung, Fehlerbehebung und Updates der Automatisierungssysteme, um die Langlebigkeit und Effizienz der Anlage sicherzustellen. Diese Phase ist entscheidend für die Kundenzufriedenheit und die Pflege langfristiger Geschäftsbeziehungen.

#### Zu bearbeitende Komponenten

Die Komponenten werden entsprechend dem geforderten Security Profil gewählt.

- Interne Design Richtlinien, Security Management
- Standard Credentials löschen, Nutzerverwaltung
- Lebensdauer der Anlage berücksichtigen
- Out of the Box Lösungen
- Netzwerkkonfiguration
- Testszenarien
- Sichere Geräte und Komponenten
- Software Libraries
- CERT, CSAF



### Defense in Depth

### Verteidigungstrategie mit ICS ATT&CK Framework

```
sudo netdiscover -r 192.168.56.0/24

sudo nmap 192.168.56.100 ... -Pn -p 1-65535 -T5

```

### ...


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


