# aipo-projekt
Project for AiPO.
---
## Dataset
Clips: https://drive.google.com/drive/folders/1nmdyPdUMOLT9aDzbtC55K6U4akBT9X3L?usp=drive_link

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

# Interfejst użytkownika

Interfejs graficzny w Pythonie umożliwiający analizę plików wideo, przewidywanie miejsca, w którym nagrano wideo, prezentację lokalizacji na mapie oraz wizualizację odtworzonego wideo.

## Funkcje

- Wybór pliku wideo (.mp4, .avi, .mov, .mkv)
- Analiza wideo z paskiem postępu
- Odtwarzanie wideo w aplikacji
- Wizualizacja lokalizacji (na podstawie analizy) na mapie w postaci czerwonego markera
- Rysowanie okręgu o zadanym promieniu wokół wykrytej lokalizacji
- Możliwość ustawienia analizy co N-tą klatkę

## Technologie

- Python 3.11+
- Tkinter – GUI
- [TkinterMapView](https://github.com/TomSchimansky/TkinterMapView) – mapa oparta na OpenStreetMap
- imageio – odczyt wideo
- threading – nieblokujące przetwarzanie
- PIL - odtwarzanie wideo

## Jak uruchomić

1. **Zainstaluj zależności**
   ```bash
   pip install -r requirements.txt
   python map.py
   ```
   
## Zrzuty ekranu

Po uruchomieniu programu zobaczymy prosty interfejs pozwalający użytkownikowi wpisać liczbę klatek i wgrać swój plik wideo

![początkowy widok GUI](screenshots/gui_start.png)

Po wgraniu pliku, ukazuje się pasek postępu odzwierciedlający ilość ukończonych obliczeń i predykcji

![ładowanie danych i analizowanie obrazu](screenshots/gui_load.png)

Po ukończonej analizie, można zobaczyć mapę wraz z zaznaczonym punktem i okręgiem pozwalający oszacować położenie jadącego samochodu. Obok mapy, odtwarzane jest wgrane wideo przez użytkownika.

![prezentacja wideo i danych na mapie](screenshots/gui_end.png)

## Opis działania GUI

 1. Użytkowik najpierw podaje liczbę klatek a następnie wybiera plik wideo.

 2. Program analizuje zawartość i pokazuje pasek postępu.

 3. Po zakończeniu analizy:

    - wyświetla współrzędne lokalizacji,

    - pokazuje je na mapie,

    - rysuje okrąg wokół punktu o przybliżonym obszarze,

    - uruchamia odtwarzanie wideo.

 4. Dostępny jest przycisk Reset, który przywraca aplikację do stanu początkowego.

## Ograniczenia interfejsu

 1. Program obsługuje tylko pliki lokalne
 2. Odtwarzane wideo jest tylko prezentacją, nie można go zatrzymać


