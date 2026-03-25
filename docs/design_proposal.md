# Design Proposal

## Autorzy

* Mateusz Ostaszewski
* Natalia Pieczko
* Michał Sadowski

## Cel projektu

Zbadanie skuteczności metody **TRAK** (Tracing with the Randomly-Projected After Kernel) w określaniu wpływu danych treningowych na predykcje, przy ograniczeniu obliczeń do wybranych warstw. Badanie zostanie przeprowadzone na zbiorze **CIFAR-10** z wykorzystaniem trzech architektur o różnej głębokości i strukturze. Jakość atrybucji zostanie zweryfikowana za pomocą metryki **LDS** (Linear Datamodeling Score).

### Zakres projektu

* Przygotowanie pipeline'u danych (z zachowaniem niemutowalności oryginalnego zbioru CIFAR-10).
* **Trening modeli bazowych:** Wytrenowanie trzech modeli (ResNet-18, ResNet-34, MobileNetV2) na pełnym zbiorze danych.
* Implementacja TRAK z selektywnym wyborem parametrów — zunifikowane konfiguracje warstw: Head-only, Late, Mid+Late, Early.
* **Budowa Ensemble'u do LDS:** Wytrenowanie po 40 modeli pomocniczych dla każdej architektury na losowych podzbiorach (50% danych).
* Analiza porównawcza TREK-LDS w zależności od architektury i zakresu śledzonych warstw.

## Harmonogram

**Tydzień 1 (23.03-29.03)**

* **25.03 - Design Proposal**
* Finalizacja koncepcji
* Przygotowanie pliku z tabelaryczną analizą literatury zgodnie z wymogami projektu
* Przygotowanie środowiska

**Tydzień 2 (30.03-05.04)**

* **02.04 - PROTOTYP**
* Trening głównych modeli na pełnym zbiorze
* Implementacja bazowej metody TRAK (gradienty)

**Tydzień 3 (06.04-12.04)**

* Generowanie masek podziału danych
* Start treningu ensemble (modele pomocnicze do LDS)

**Tydzień 4 (13.04-19.04)**

* Kontynuacja treningu modeli pomocniczych
* Pierwsze testy korelacji LDS dla ResNet-18.

**Tydzień 5 (20.04-26.04)**

* Przeliczanie LDS dla różnych konfiguracji warstw we wszystkich modelach
* Zbieranie wyników

**Tydzień 6 (27.04-03.05)**

* Majówka - czas na odpoczynek

**Tydzień 7 (04.05-10.05)**

* Analiza porównawcza: wpływ głębokości sieci na stabilność metody TRAK w róznych warstwach

**Tydzień 8 (11.05-17.05)**

* Finalizacja eksperymentów, generowanie wykresów zbiorczych i map wpływu (attribution maps)
* Przygotowanie dokumentacji

**Tydzień 9 (18.05-24.05)**

* **21.05 - Termin oddania projektu pozwalający na zwolnienie z kolokwium**
* Poprawki kodu, czyszczenie repozytorium i przygotowanie raportu końcowego
* Stworzenie filmiku prezentującego projekt

**Termin końcowy 21.05**

## Bibliografia

* **TRAK:** S. M. Park et al., *"TRAK: Attributing Model Behavior at Scale"*, ICML 2023. [https://arxiv.org/pdf/2303.14186](<https://arxiv.org/pdf/2303.14186>)
* **Oficjalna implementacja TRAK**: [https://github.com/MadryLab/trak](<https://github.com/MadryLab/trak>)
* **LDS Metric:** A. Ilyas et al., *"Datamodels: Predicting Predictions from Training Data"*, 2022. [https://arxiv.org/abs/2202.00622](<https://arxiv.org/abs/2202.00622>)

## Planowany zakres eksperymentów

1. **Wytrenowanie modeli ResNet / MobileNet**

Badanie skupia się na zbiorze **CIFAR-10** (10 klas, obrazy 32×32×3).

Wykorzystamy trzy architektury o różnej głębokości, aby sprawdzić, czy wraz ze wzrostem głębokości modelu informacja o wpływie danych "ucieka" z końcowych warstw:

* **ResNet-18:** (8 BasicBlocks w 4 grupach) - \~11.2M parametrów, baseline rezydualna.
* **ResNet-34:** (16 BasicBlocks w 4 grupach) - \~21.3M parametrów, głębsza sieć z tym samym typem bloku.
* **MobileNetV2:** (19 inverted residual blocks) - \~3.4M parametrów, lekka architektura z inną strukturą rezydualną - grupa kontrolna.

2. **Badanie konfiguracji warstw w TRAK**

Każda architektura ma hierarchiczną ekstrakcję cech. Definiujemy 4 konfiguracje przez pozycję w sieci:

| **Konfiguracja** | **ResNet-18 / ResNet-34** | **MobileNetV2** |
| -- | -- | -- |
| **Head-only** | `fc` | `classifier` |
| **Late** | `layer4` + `fc` | `features[14:]` + `classifier` |
| **Mid+Late** | `layer3` | `features[7:14]` |
| **Early** | `conv1` + `layer1` | `features[:7]` |

3. **Obliczenie metryki LDS** (Linear Datamodeling Score) poszczególnych wariantów poprzez wielokrotne retrenowanie modelu na podzbiorach danych treningowych.

4. **Porównanie wariantów** pod kątem jakości wyjaśnień (LDS) oraz zapotrzebowania na zasoby i czas obliczeń.

### **Szacowane zapotrzebowanie na czas i zasoby**

Obliczenia zostaną przeprowadzone z użyciem pojedynczej karty graficznej NVIDIA RTX 3080 (10GB VRAM). Ze względu na stosunkowo niewielki wymiar danych (zbiór CIFAR-10, obrazy 32x32 piksele) oraz wysoką przepustowość karty, szacowany czas operacji wynosi:

* **Trening modeli bazowych (100% danych, 30 epok):** 3 modele $\times$ (30 epok $\times$ 20s) = ok. **30 minut**.
* **Trening modeli pomocniczych do ewaluacji LDS** (120 modeli, 50% zbioru danych, 30 epok): 120 $\times$ (30 epok $\times$ 15s) = ok. **15 godzin**
* **Ekstrakcja wpływu (TRAK)**: 112 niezależnych przebiegów algorytmu (3 wytrenowane modele bazowe $\times$ 4 badane konfiguracje warstw). Operacja ta wykorzystuje metodę rzutowań losowych i opiera się głównie na szybkiej inferencji. Szacowany czas to ok. **1,5 do 2 godzin**.

**Całkowity szacowany budżet czasowy eksperymentów:** ok. 17,5 godziny ciągłej pracy GPU.

## Planowana funkcjonalność programu

* Skrypt generujący losowe maski $0/1$ dla zbioru treningowego (wymagane do LDS).
* Integracja z biblioteką `trak` pozwalająca na przekazywanie przefiltrowanych list parametrów (`named_parameters`).
* Moduł porównujący predykcje z wielu modeli (logity) z wynikami TRAK scores w celu wyznaczenia korelacji.
* Generowanie wykresów korelacji (LDS plots) oraz wizualizacja próbek o najwyższym wpływie na błędne klasyfikacje.

## Planowany stack technologiczny

* **Język:** Python `>=3.10`
* **Środowisko wirtualne i zarządzanie pakietami:** `uv`
* **Oskryptowane budowanie, testowanie i uruchamianie aplikacji:** `make`
* **Jakość kodu:** `ruff` (linter i autoformatter), kod zgodny z PEP8
* **Kontrola wersji:** Git z rygorem `Conventional Commits`
* **Struktura projektu:** `cookiecutter`
* **Śledzenie eksperymentów:** `Weights & Biases` (do logowania treningu wielu modeli LDS)
* **Wykorzystywane biblioteki:** `trak` (atrybucja danych)**,** `PyTorch`, `Torchvision`
* **Infrastruktura**: Środowisko z akceleracją GPU (CUDA) / MPS (Apple Silicon).
