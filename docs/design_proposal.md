# Design Proposal: Evaluating Data Attribution with TRAK on Targeted Layers of Residual Networks

**Project Goal:** Zbadanie skuteczności metody TRAK (Tracing with the Randomly-Projected After Kernel) w określaniu wpływu danych treningowych na predykcje, przy ograniczeniu obliczeń do wybranych warstw. Badanie zostanie przeprowadzone na zbiorze **CIFAR-10** z wykorzystaniem trzech architektur o różnej głębokości i strukturze. Jakość atrybucji zostanie zweryfikowana za pomocą metryki LDS (Linear Datamodeling Score).

**Project Scope:**
* Przygotowanie pipeline'u danych dla zbioru CIFAR-10.
* **Trening modeli bazowych:** Wytrenowanie trzech modeli (ResNet-18, ResNet-34, MobileNetV2) na pełnym zbiorze danych.
* Implementacja TRAK z selektywnym wyborem parametrów — zunifikowane konfiguracje warstw: Head-only, Late, Mid+Late, Early.
* **Budowa Ensemble'u do LDS:** Wytrenowanie po 20 modeli pomocniczych dla każdej architektury na losowych podzbiorach (50% danych).
* Analiza porównawcza TREK-LDS w zależności od architektury i zakresu śledzonych warstw.

---

## 1. Project Schedule

| Tydzień | Daty | Etap / Zadania | Kluczowe Daty |
| :--- | :--- | :--- | :--- |
| **W1** | 23.03 – 29.03 | Finalizacja koncepcji i bazowych modeli. Przygotowanie środowiska. | **25.03 (Śr): Design Proposal** |
| **W2** | 30.03 – 05.04 | Trening głównych modeli na pełnym zbiorze. Implementacja bazowej metody TRAK (gradienty). | **02.04 (Cz): PROTOTYP** |
| **W3** | 06.04 – 12.04 | Generowanie masek podziału danych. Start treningu ensemble (modele pomocnicze do LDS). | Intensywne użycie GPU |
| **W4** | 13.04 – 19.04 | Kontynuacja treningu modeli pomocniczych. Pierwsze testy korelacji LDS dla ResNet-18. | **16.04 (Cz): Standup 1** |
| **W5** | 20.04 – 26.04 | Przeliczanie LDS dla różnych konfiguracji warstwwe wszystkich modelach. | Zbieranie wyników |
| **W6** | 27.04 – 03.05 | **MAJÓWKA – Odpoczynek i regeneracja.** | **Czas wolny** |
| **W7** | 04.05 – 10.05 | Analiza porównawcza: wpływ głębokości sieci na stabilność metody TRAK w róznych warstwach. | **07.05 (Cz): Standup 2** |
| **W8** | 11.05 – 17.05 | Finalizacja eksperymentów, generowanie wykresów zbiorczych i map wpływu (attribution maps). | Dokumentacja |
| **W9** | 18.05 – 24.05 | Polerowanie kodu, czyszczenie repozytorium i przygotowanie raportu końcowego. | **21.05 (Cz): Termin zwolnienia** |

---

## 2. Methodology & Architectures

Badanie skupia się na zbiorze **CIFAR-10** (10 klas, obrazy $32 \times 32 \times 3$).

Wykorzystamy trzy architektury o różnej głębokości, aby sprawdzić, czy wraz ze wzrostem głębokości modelu informacja o wpływie danych "ucieka" z końcowych warstw:

1.  **ResNet-18:** (8 BasicBlocks w 4 grupach) — ~11.2M parametrów, baseline rezydualna.
2.  **ResNet-34:** (16 BasicBlocks w 4 grupach) — ~21.3M parametrów, głębsza sieć z tym samym typem bloku.
3.  **MobileNetV2:** (19 inverted residual blocks) — ~3.4M parametrów, lekka architektura z inną strukturą rezydualną — grupa kontrolna.

**Zunifikowane konfiguracje warstw do TRAK:**

Każda architektura ma hierarchiczną ekstrakcję cech. Definiujemy 4 konfiguracje przez pozycję w sieci:

| Konfiguracja | ResNet-18 / ResNet-34 | MobileNetV2 |
| :--- | :--- | :--- |
| **Head-only** | `fc` | `classifier` |
| **Late** | `layer4` + `fc` | `features[14:]` + `classifier` |
| **Mid+Late** | `layer3` | `features[7:14]` |
| **Early** | `conv1` + `layer1` | `features[:7]` |

---

## 3. Planned Program Functionality

* **Automatic Masking:** Skrypt generujący losowe maski $0/1$ dla zbioru treningowego (wymagane do LDS).
* **Selective TRAKer:** Integracja z biblioteką `trak` pozwalająca na przekazywanie przefiltrowanych list parametrów (`named_parameters`).
* **LDS Engine:** Moduł porównujący predykcje z wielu modeli (logity) z wynikami TRAK scores w celu wyznaczenia korelacji.
* **Visualization Suite:** Generowanie wykresów korelacji (LDS plots) oraz wizualizacja próbek o najwyższym wpływie na błędne klasyfikacje.

---

## 4. Bibliography

* **TRAK Method:** S. M. Park et al., *"TRAK: Attributing Model Behavior at Scale"*, ICML 2023. [https://arxiv.org/abs/2303.14186](https://arxiv.org/abs/2303.14186)
* **LDS Metric:** A. Ilyas et al., *"Datamodels: Predicting Predictions from Training Data"*, 2022. [https://arxiv.org/abs/2202.00622](https://arxiv.org/abs/2202.00622)
* **Codebase:** Oficjalna implementacja TRAK: [https://github.com/MadryLab/trak](https://github.com/MadryLab/trak)

---

## 5. Planned Tech Stack

* **Deep Learning:** PyTorch, Torchvision.
* **Data Attribution:** `trak` library.
* **Monitoring:** Weights & Biases (do logowania treningu wielu modeli LDS).
* **Infrastructure:** Środowisko z akceleracją GPU (CUDA) / MPS (Apple Silicon).

---
