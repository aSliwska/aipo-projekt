# Project for AiPO.

## Uruchamianie



## Wstęp
Projekt ten jest aplikacją pozwalającą na postawie wideo z dashcamu samochodowego automatycznie określić, gdzie na świecie ono powstało. Dokonuje on oszacowania na podstawie ruchu prawo/lewostronnego, widocznych na wideo znaków drogowych i rejestracji samochodowych, tekstu z szyldów sklepowych i języka, w którym jest on napisany. Zwraca uwagę na nazwy własne i odległości, które mogą z nimi występować na znakach drogowych i bilbordach. Po wyciągnięciu z wideo tych informacji tworzy on zapytania do Nominatim API (OpenStreetMaps) i szacuje położenie na bazie zwróconych list koordynatów.


## Opis poszczególnych części projektu

### 1. GUI


### 2. Główna pętla i komunikacja z OpenStreetMaps
1. Główna pętla - 
2. Szacowanie kraju pochodzenia - Każdy kraj rozważany przez aplikację dostaje pewien wynik na podstawie list krajów zwróconych z różnych źródeł (kierunek ruchu, znaki drogowe, rejestracje, języki). Każde źródło informacji ma swoją wagę, która jest dodawana do wyniku każdego kraju, jeżeli został on przez to źródło zwrócony. Kraje są następnie sortowane malejąco według tego wyniku, a wyniki z 0 są odfiltrowywane.
3. Komunikacja z OpenStreetMaps - Tworzony jest iloczyn kartezjański ze wszystkich zwróconych regionów pasujących do wybranego kraju oraz ze wszystkich znalezionych nazw własnych. Wybrany kraj to ten z największym wynikiem (lub kolejne z największym wynikiem, jeżeli dla danego nie dostaniemy wyniku). Zapytania w formacie "kraj, region, miejsce" są wysyłane do Nominatim API z jednosekundowym opóźnieniem, a wyniki są cache'owane w generowanym pliku osm_cache.db. Wyniki to listy znalezionych geolokacji z metadanymi, z których wyciągane są punkty na mapie w formacie (lat, lon).
4. Szacowanie położenia - Z otrzymanej listy geolokacji wyciągane są mediany szerokości i wysokości geograficznych (osobno). Następnie na podstawie tej mediany obliczany jest promień pewności wyniku - maksymalna wartość spośród odległości od mediany do innego punktu. Odległość ta jest powiększona o dystans znaleziony na bilbordzie/znaku drogowym, który zawierał daną nazwę miejsca (przed wyciągnięciem maksimum). Mediana położenia i promień są zwracane.

Wykorzystane narzędzia:
- sqlite3 + pickle - do cache'owania zapytań (wymaganie korzystania z Nominatim API)
- geopy - do obsługi Nominatim API z poziomu języka Python, opóźniania wysyłania do Nominatim API zapytań o 1s (kolejne wymaganie API) i obliczania odległości między koordynatami na mapie
- Nominatim API - internetowe REST API do geocodingu 
- pandas + numpy + functools - do operacji na danych 
- cv2 - do wczytywania pliku wideo

### 3. Wykrywanie ruchu prawo- i lewostronnego


### 4. Znajdywanie i klasyfikacja znaków drogowych


### 5. Klasyfikacja rejestracji pojazdów drogowych


### 6. Znajdywanie tekstu


### 7. OCR

#### Tryb klatki:
Klatka jest heurystycznie poddawana dwustopniowemu powiększaniu i wyostrzaniu, w celu uwypuklenia tekstu i sprowadzenia go do obszaru optymalnej pracy sieci OCR. Preprocessing zakłada następnie equalizację histogramu, konwersję do skali szarości i rozciągnięcie histogramu. Następnie klatka jest poddawana parsowaniu przez sieć OCR.

#### Tryb precyzyjny:
Ten tryb zakłada, że na wejście zostaje podany sam tekst wraz z pewnym jego rejonem (zestaw linii tekstu, obwód znaku). Pomiędzy stopiami przetwarzania zachodzi korekcja perspektywy i rotacji obrazu. Główna metoda opiera się o znalezienie punktu horyzontu na podstawie znalezionych linii obrysu (transformacja Hough), obliczenie macierzy homeografii i następnie korekcji. W przypadku nieznalezienia żadnego punktu, zostaje uruchomiona bardziej złożona i jednocześnie bardziej zawodna metoda obliczenia macierzy homeografii w oparciu o rozwiązanie układu równań liniowych nadokreślonego z użyciem rozkładu SVD (układ 8 równań z 9 niewiadomymi) dla 4 wierzchołków tła. Wierzchołki są dobierane automatycznie poprzez wykrywanie kwadratowych konturów.

Użyte narzędzia:
- OpenCV
- Pytesseract
- Numpy + Matplotlib
- EasyOCR (porzucony)

Znane problemy:
- Tryb klatki często myli ze sobą podobne alfabety (latin-cyrillic, chinese-japanese-korean, filipino-arabic, etc.).
- Oba tryby, a zwłaszcza tryb klatki, generują znaczny szum znakowy.


### 8. Rozpoznawanie języka i tłumaczenie


### 9. Wyciąganie słów kluczowych i odległości z tekstu


## Co nie działa


## Źródła
1. Dataset - linki do filmów YouTube znajdują się w sekcji "Testowy dataset".
2. [Klasa cache'ująca dane z Nominatim API](https://stackoverflow.com/questions/28397847/most-straightforward-way-to-cache-geocoding-data)
3. Lista krajów na świecie [1](https://www.britannica.com/topic/list-of-countries-1993160), [2](https://en.wikipedia.org/wiki/Left-_and_right-hand_traffic)
4. [Nominatim Usage Policy](https://operations.osmfoundation.org/policies/nominatim/)
5. [Dokumentacja GeoPy](https://geopy.readthedocs.io/en/stable/)
6.


## Podział zadań
- Aleksandra Śliwska – lider zespołu, tworzenie testowego datasetu, iteracja po klatkach filmu, komunikacja z OpenStreetMaps i obliczanie finalnego położenia + promienia niepewności z otrzymanych geolokacji
- Glib Bersutskyi - 
- Marcin Kiżewski - 
- Arkadiusz Korzeniak - 
- Kamil Krzysztofek - tworzenie testowego datasetu, rozpoznawanie języka
- Patryk Madej - 
- Adam Niewczas - GUI
- Arkadiusz Rudy - 
- Wiktor Szewczyk - 
- wszyscy - dokumentacja


## Testowy dataset

Dataset do testowania aplikacji został stworzony przez nas ręcznie z ogólnodostępnych materiałów na YouTube, ponieważ datasety z dashcamów dostępne w Internecie nie spełniały naszych wymagań na temat widoczności rejestracji samochodowych, jakości filmów, formatu plików, zróżnicowania kraju i miast pochodzenia materiałów oraz były niepotrzebnie duże.

[LINK DO FILMÓW](https://drive.google.com/drive/folders/1nmdyPdUMOLT9aDzbtC55K6U4akBT9X3L?usp=drive_link)

|ID |Link to source                              |Country     |City    |Minute start|Minute end|Lat                |Lon               |Coordinate accuracy|
|---|-------------------------------------------|---------|----------|------------|-------------|-------------------|------------------|----------------------|
|1  |https://www.youtube.com/watch?v=7HaJArMDKgI|USA      |New York  |00:22       |00:44        |40.742621472431026 |-73.98038747208632|close                |
|2  |https://www.youtube.com/watch?v=7HaJArMDKgI|USA      |New York  |09:15       |09:35        |40.742621472431027 |-73.98038747208633|close                |
|3  |https://www.youtube.com/watch?v=cpnOJaZlPaI|Poland   |Cracow    |03:03       |03:33        |50.06617189335998  |19.94408750109151 |exact              |
|4  |https://www.youtube.com/watch?v=cpnOJaZlPaI|Poland   |Cracow    |21:15       |21:35        |50.060822730895865 |19.924014674791177|exact              |
|5  |https://www.youtube.com/watch?v=rTO7sICY3JM|Germany   |Frankfurt |04:55       |05:15        |50.10918301852155  |8.695969407340556 |close                |
|6  |https://www.youtube.com/watch?v=rTO7sICY3JM|Germany   |Frankfurt |08:04       |08:24        |50.10918301852155  |8.695969407340556 |exact              |
|7  |https://www.youtube.com/watch?v=JSH22SdMnFQ|India    |Mumbai    |10:40       |11:00        |19.08170011129876  |72.8559963283084  |far                |
|8  |https://www.youtube.com/watch?v=JSH22SdMnFQ|India    |Mumbai    |18:35       |18:55        |19.08170011129876  |72.8559963283084  |far                |
|9  |https://www.youtube.com/watch?v=9pPBmcKvFOc|Australia|Brisbane  |06:35       |07:05        |-27.441502154471287|153.03896430550725|close                |
|10 |https://www.youtube.com/watch?v=9pPBmcKvFOc|Australia|Brisbane  |26:45       |27:05        |-27.441502154471288|153.03896430550726|close                |
|11 |https://www.youtube.com/watch?v=IM9uH-XoKq8|Russia    |Moscow    |16:25       |16:45        |55.762557230350254 |37.64390131662508 |exact             |
|12 |https://www.youtube.com/watch?v=kd7P8Xyxuf8|Ukraine  |Kyiv      |13:00       |13:15        |50.46222206518409  |30.49911807360477 |exact             |
|13 |https://www.youtube.com/watch?v=_hptbEVx5eM|Brasil |Sao Paulo |01:40       |02:05        |-23.569943099334843|-46.6598785426813 |exact             |
|14 |https://www.youtube.com/watch?v=fFQb1QGKxS4|Greece   |Chania    |07:33       |07:53        |35.51237244050218  |24.015556808864858|exact             |
|15 |https://www.youtube.com/watch?v=Y4lPaRPf9iU|Egypt    |Alexandria|04:45       |05:00        |31.262086361655292 |29.984206889679946|exact             |
|16 |https://www.youtube.com/watch?v=Y4lPaRPf9iU|Egypt    |Alexandria|13:00       |13:20        |31.219909948041767 |29.942202079961422|exact             |
|17 |https://www.youtube.com/watch?v=STFPIMa3mXo|Poland   |Kielce    |06:28       |06:54        |50.882209          |20.645231         |exact             |
|18 |https://www.youtube.com/watch?v=MAiltiE8tgI|China    |Shanghai  |01:29:43    |01:30:03     |31.248994669315067 |121.4871027357963 |close                |
|19 |https://www.youtube.com/watch?v=u7wxZKSbTZs|Japan  |Kyoto     |21:55       |22:15        |35.01116214130449  |135.77824168969553|exact             |

