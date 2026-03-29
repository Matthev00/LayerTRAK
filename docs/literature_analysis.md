# Datamodels

Praca Datamodels stanowi ważny punkt odniesienia dla całego projektu, ponieważ porządkuje sposób myślenia o wpływie danych treningowych na zachowanie modelu. Jej główna idea polega na tym, że dla ustalonego przykładu testowego można analizować nie tylko końcową predykcję modelu, ale również to, jak ta predykcja zmieniałaby się po zmianie zbioru treningowego. Innymi słowy, zamiast pytać jedynie, które przykłady treningowe wydają się podobne lub intuicyjnie istotne, podejście to kieruje uwagę na bardziej kontrfaktyczne pytanie: jak wyglądałaby odpowiedź modelu, gdyby trenował się na innym podzbiorze danych.

Z perspektywy tego repozytorium najistotniejsze jest to, że Datamodels dostarcza podstawy koncepcyjnej dla takiego właśnie kontrfaktycznego spojrzenia na attribution. To z tej pracy wynika intuicja, że dobra metoda atrybucji powinna być oceniana nie tylko przez jakościowe przykłady, ale również przez to, czy trafnie odzwierciedla zmiany zachowania modelu przy zmianie danych treningowych. Sama praca Datamodels jest więc tutaj ważna przede wszystkim jako źródło sposobu formułowania problemu oraz uzasadnienie, dlaczego ewaluacja attribution powinna być powiązana z retrenowaniem modeli na różnych podzbiorach danych.

W praktyce Datamodels pełni tu rolę zaplecza koncepcyjnego i ewaluacyjnego, a nie bezpośrednio używanej metody attribution. Pełny pipeline tej pracy jest bardzo kosztowny obliczeniowo, ponieważ opiera się na masowym trenowaniu modeli na wielu różnych podzbiorach zbioru treningowego. Z tego względu w projekcie nie jest on traktowany jako główna metoda eksperymentalna, lecz jako punkt odniesienia dla sposobu rozumienia wpływu danych treningowych

## Materiały:
- artykuł: https://arxiv.org/abs/2202.00622
- kod: https://github.com/MadryLab/datamodels

# TRAK

Praca TRAK stanowi bezpośrednią podstawę metodyczną części eksperymentalnej projektu. Jej znaczenie wynika z tego, że proponuje metodę attribution zaprojektowaną z myślą o znacznie lepszej skalowalności niż pełne podejścia kontrfaktyczne. Zamiast wielokrotnie odtwarzać kosztowny proces trenowania modeli na ogromnej liczbie podzbiorów danych, TRAK przybliża wpływ przykładów treningowych przy użyciu informacji gradientowej, lokalnej linearyzacji modelu oraz losowych rzutowań. Dzięki temu możliwe jest badanie wpływu danych treningowych na predykcje modelu w sposób dużo bardziej praktyczny obliczeniowo.

To właśnie w pracy TRAK kontrfaktyczna intuicja znana z Datamodels zostaje przełożona na bardziej bezpośredni protokół ewaluacyjny dla metod data attribution. Szczególnie ważne jest tutaj sformułowanie LDS (Linear Datamodeling Score) jako jawnej metryki oceniającej, na ile attribution scores potrafią przewidywać zmiany zachowania modelu przy trenowaniu na losowych podzbiorach danych. W tym sensie LDS można traktować jako rozwinięcie sposobu myślenia obecnego w Datamodels, ale jego sformalizowana postać jako metryki ewaluacyjnej pojawia się właśnie w TRAK.

W kontekście tego repozytorium TRAK jest traktowany jako właściwa metoda badawcza. Szczególnie interesujące jest tutaj to, że opiera się on na reprezentacjach gradientowych względem parametrów modelu, co naturalnie prowadzi do pytania, czy dla uzyskania dobrej jakości attribution konieczne jest wykorzystywanie całego zbioru parametrów, czy też wystarczające mogą być jedynie wybrane części sieci. Na tym właśnie opiera się główna linia eksperymentalna projektu: analiza działania TRAK przy ograniczeniu obliczeń do różnych zakresów warstw, takich jak Head-only, Late, Mid+Late oraz Early, oraz porównanie tych wariantów między architekturami ResNet-18, ResNet-34 i MobileNetV2.

W tym ujęciu TRAK nie jest jedynie gotową biblioteką używaną w niezmienionej formie, ale punktem wyjścia do bardziej szczegółowej analizy zależności między zakresem parametrów a jakością attribution mierzoną przez LDS. To właśnie ten aspekt nadaje projektowi własny kierunek eksperymentalny.

## Materiały:
- artykuł: https://arxiv.org/pdf/2303.14186
- kod: https://github.com/MadryLab/trak

# LDS (Linear Datamodeling Score)

Metryka LDS (Linear Datamodeling Score) pełni w projekcie rolę głównej miary jakości atrybucji. Jej znaczenie wynika z tego, że pozwala oceniać metody attribution nie tylko na podstawie jakościowych przykładów, ale przede wszystkim przez ich zgodność z rzeczywistym, kontrfaktycznym zachowaniem modelu.

Intuicja stojąca za LDS jest następująca: jeżeli metoda atrybucji poprawnie identyfikuje wpływ danych treningowych na predykcję modelu, to suma score’ów przypisanych do przykładów obecnych w danym podzbiorze treningowym powinna dobrze przewidywać, jak zachowa się model wytrenowany właśnie na tym podzbiorze. Innymi słowy, dobra metoda attribution powinna nie tylko wskazywać przykłady „istotne”, ale również odtwarzać zmiany wyjścia modelu w scenariuszach, w których skład danych treningowych ulega zmianie.

W tym sensie LDS stanowi pomost między estymacją wpływu a rzeczywistym zachowaniem modeli trenowanych na różnych podzbiorach danych. Wysoka wartość tej metryki oznacza, że uzyskane attribution scores są spójne z obserwowaną zmianą predykcji modelu i dobrze oddają strukturę zależności między danymi treningowymi a wynikiem dla przykładu testowego.

W praktyce obliczenie LDS w ramach projektu obejmuje kilka etapów. Najpierw trenowany jest ensemble modeli pomocniczych na losowych podzbiorach danych treningowych, reprezentowanych przez maski 0/1. Następnie dla ustalonego przykładu testowego wyznaczana jest wartość wyjścia modelu dla każdego z takich podzbiorów, na przykład w postaci wybranego logitu lub innej analizowanej wielkości. Równolegle, dla tego samego przykładu obliczane są attribution scores metodą TRAK. Ostatnim krokiem jest porównanie przewidywań wynikających z sumy score’ów w danym podzbiorze z rzeczywistym zachowaniem modeli pomocniczych, co w pracy TRAK realizowane jest przez korelację rang Spearmana.

W kontekście tego projektu LDS ma szczególne znaczenie, ponieważ pozwala porównać jakość atrybucji uzyskanej z różnych zakresów parametrów modelu. Dzięki temu możliwe jest sprawdzenie, czy wysoka jakość attribution utrzymuje się już dla konfiguracji obejmujących jedynie końcowe warstwy, takich jak Head-only lub Late, czy też wymaga uwzględnienia szerszego zakresu parametrów, obejmującego również warstwy wcześniejsze. Metryka ta stanowi więc główne narzędzie porównawcze pomiędzy badanymi wariantami TRAK

# Porównanie

| Pozycja | Artykuł | Kod | Pretrained / precomputed | Metryki / ewaluacja | Zasoby obliczeniowe | Rola w projekcie | Komentarz autorski |
|---|---|---|---|---|---|---|---|
| Datamodels | https://arxiv.org/abs/2202.00622 | https://github.com/MadryLab/datamodels | Kod: tak. Precomputed wyniki dla CIFAR-10: tak. Klasyczne pretrained modele: nie wskazano wprost. | Ewaluacja opiera się na przewidywaniu wyników modeli trenowanych na różnych podzbiorach danych; praca stanowi koncepcyjne tło dla kontrfaktycznej oceny attribution. | Repo dla przykładu CIFAR-10 zakłada maszynę z dostępem do 8 GPU; autorzy podkreślają też, że obliczenia wymagają trenowania dużej liczby modeli. | Podstawa koncepcyjna. Praca uzasadnia kontrfaktyczne spojrzenie na wpływ danych treningowych. | Najważniejsza z punktu widzenia zrozumienia, czym w ogóle jest „dobra” atrybucja danych treningowych. Zbyt kosztowna, by odtwarzać ją w pełnej skali w tym projekcie, ale bardzo ważna jako tło dla dalszych eksperymentów. |
| TRAK | https://arxiv.org/pdf/2303.14186 | https://github.com/MadryLab/trak | Kod: tak. Pre-computed TRAK scores dla CIFAR-10: tak. Klasyczne pretrained modele: repo skupia się głównie na metodzie i przykładach użycia. | W pracy formalnie pojawia się LDS jako metryka oceny attribution; autorzy porównują też skuteczność i czas działania względem innych metod. | W pracy czasy porównawcze raportowane są dla pojedynczego A100; repo udostępnia tutoriale i przykłady użycia. | Główna metoda badawcza w projekcie. Na jej podstawie badany jest wpływ ograniczenia attribution do wybranych warstw. | Najbardziej praktyczna pozycja dla tego projektu. Łączy sensowną jakość atrybucji z kosztem, który da się pogodzić z eksperymentami na CIFAR-10. To właśnie tutaj pojawia się LDS jako jawna metryka ewaluacyjna. |