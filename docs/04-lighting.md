# 04. Освітлення

## Модель

Одна абстракція, яка покриває і LED-стрічку, і DMX-прилади:

> **Всесвіт (universe) — це масив кольорових каналів. Кадр — це
> знімок усього всесвіту в конкретний момент. Драйвер уміє одне:
> взяти кадр і відправити його в залізо.**

WS2812B на 300 пікселів — це 300 RGB-елементів. DMX-парканна голова —
це кілька каналів (dimmer, R, G, B, strobe, pan, tilt). Різниця лише
в тому, як **прилад** описує свої канали. Тому:

```
Track features  →  Cue Engine  →  Frame[]  →  Fixture mapping  →  Driver
   (band, beat)     (правила)     (RGB)        (розкладка)        (UDP)
```

## Вибір заліза — рекомендація

**Почни з ESP32 + WS2812B, не з DMX.** Причини:

| | ESP32 + WLED | DMX512 | Розумні лампи |
|---|---|---|---|
| Ціна старту | ~$15–25 | $80–300+ | вже є |
| Латентність | 5–20 мс | 5–15 мс | 100–300 мс |
| Per-pixel контроль | так, сотні пікселів | ні, канали приладу | ні |
| Складність підключення | Wi-Fi, прошивка за 10 хв | USB-DMX або Ethernet-нода, XLR-кабелі, термінатор | API кожного вендора |
| Придатність для музики | відмінна | відмінна | **погана** |

Розумні лампи (Hue/Yeelight/Tuya) відпадають одразу: 100–300 мс
затримки на кожну команду плюс rate limiting означає, що в такт
вони не блимають ніколи. Вони підходять для повільної зміни
атмосфери — і тільки як додатковий "фоновий" драйвер.

**Конкретний старт:**
- ESP32 DevKit (~$6) + WS2812B стрічка 5 м / 60 LED/м (~$12) +
  БЖ 5V 10A (~$12). Не живи стрічку від USB — 300 пікселів на повній
  яскравості це ~18 А.
- Прошити [WLED](https://install.wled.me/) прямо з браузера.
- Увімкнути в WLED **DDP** або **DNRGB** realtime.

DMX додаси на M7, коли захочеш справжні голови/парки. Абстракція вже
буде готова — це буде новий клас драйвера, і більше нічого.

Куплене залізо, IP стрічки, налаштування WLED і виміряну затримку
записуй у нотатку **«Wave — Залізо»** в Obsidian — там уже є чеклист
покупки і поля під калібрування.

## Протоколи

| Протокол | Порт UDP | Коли |
|---|---|---|
| **DDP** | 4048 | Основний вибір для WLED. Простий заголовок, до 480 пікселів у пакеті, підтримує зсув для довгих стрічок |
| DNRGB (WLED native) | 21324 | Запасний. 489 LED/пакет, є поле timeout, після якого WLED сам повертається до своїх ефектів |
| **Art-Net** | 6454 | Стандарт для DMX по Ethernet. 512 каналів на всесвіт |
| sACN / E1.31 | 5568 | Те саме, що Art-Net, але з мультикастом і пріоритетами. WLED теж уміє |

Практично: `DdpDriver` для стрічки, `ArtNetDriver` для DMX. Обидва —
UDP fire-and-forget, ~150 рядків кожен.

Дрібниця, яка з'їсть вечір, якщо не знати: у **DNRGB перший байт —
таймаут у секундах**, після якого WLED повертається до власних
ефектів. Постав 2, а не 0, інакше при паузі в потоці стрічка
"оживе" сама.

## Абстракція драйверів (.NET)

```csharp
public readonly record struct Rgb(byte R, byte G, byte B);

public sealed class LightFrame
{
    public double TimeSec { get; init; }
    public Rgb[] Pixels { get; init; }   // весь всесвіт
}

public interface ILightDriver : IAsyncDisposable
{
    string Name { get; }
    int PixelCount { get; }
    ValueTask ConnectAsync(CancellationToken ct);
    ValueTask SendAsync(LightFrame frame, CancellationToken ct);
    ValueTask BlackoutAsync();      // обов'язково — на стоп і на помилку
}
```

Реалізації:

- `VirtualDriver` — шле кадри у SignalR-хаб, браузер малює. **Пиши
  його першим.** Він дозволяє розробляти весь cue engine без заліза
  взагалі, і залишається назавжди як інструмент відладки.
- `DdpDriver` — UDP на IP стрічки.
- `ArtNetDriver` — UDP broadcast або unicast, DMX-всесвіти.
- `MulticastDriver` — обгортка, що дублює кадр у кілька драйверів.
  Потрібна, щойно з'явиться друга стрічка.

## Розкладка приладів (fixtures)

Всесвіт — плоский масив, але фізично це кілька об'єктів. Опис у JSON:

```json
{
  "name": "Кімната",
  "universe_size": 300,
  "fixtures": [
    { "id": "strip_left",  "type": "pixel_strip",
      "offset": 0,   "count": 150, "role": "ambient" },
    { "id": "strip_right", "type": "pixel_strip",
      "offset": 150, "count": 150, "role": "ambient" },
    { "id": "bar_front",   "type": "pixel_strip",
      "offset": 0,   "count": 60,  "role": "beat" }
  ],
  "latency_offset_ms": -40
}
```

`role` — це те, за що cue engine чіпляється: не "пікселі 0–150",
а "усе, що має роль `beat`". Тоді пресети переносяться між різними
розкладками без переписування.

`latency_offset_ms` від'ємний — світло треба слати **раніше** за звук,
бо звук у браузері має свій буфер. Підбирається на слух один раз:
зніми на телефон стрічку разом зі звуком, подивись покадрово.

## Cue Engine — правила

Вхід — покадрові фічі з `.npz` (band_energy 8×N, onset_env, beats,
downbeats, sections), вихід — `LightFrame[]` на 60 FPS.

### Базовий набір правил

```json
{
  "rules": [
    { "type": "band_to_color", "role": "ambient",
      "mapping": {
        "sub":        { "color": "#FF0033", "weight": 1.0 },
        "bass":       { "color": "#FF6600", "weight": 0.8 },
        "mid":        { "color": "#00CC66", "weight": 0.6 },
        "presence":   { "color": "#0066FF", "weight": 0.5 },
        "brilliance": { "color": "#FFFFFF", "weight": 0.3 }
      },
      "smoothing_ms": 80 },

    { "type": "beat_flash", "role": "beat",
      "source": "downbeats", "color": "#FFFFFF",
      "decay_ms": 120, "intensity": 0.9 },

    { "type": "section_palette",
      "palettes": {
        "intro":     ["#001133", "#003366"],
        "build":     ["#330066", "#660099"],
        "drop":      ["#FF0033", "#FF9900", "#FFFFFF"],
        "breakdown": ["#003344", "#006688"],
        "outro":     ["#110022", "#220044"]
      },
      "crossfade_ms": 2000 },

    { "type": "key_hue_base",
      "description": "базовий відтінок від тональності: 12 нот → 360°",
      "saturation": 0.7, "weight": 0.3 },

    { "type": "chase", "role": "ambient",
      "trigger": "onset_peak", "speed_px_per_sec": 400 }
  ]
}
```

### Як це рахується

```
для кожного кадру t (крок 1/60 сек):
    base   = палітра поточної секції, інтерпольована
    hue    = зсув від key_hue_base
    energy = зважена сума band_energy[t] по mapping
    color  = base * energy, згладжено за smoothing_ms
    якщо t близько до downbeat: додати спалах з decay
    записати у пікселі відповідної ролі
```

Дві речі, які роблять різницю між "блимає" і "виглядає добре":

1. **Асиметричне згладжування.** Атака має бути миттєвою, спад —
   повільним. `if new > cur: cur = new else: cur += (new-cur) * a`.
   Симетричне згладжування вбиває всю перкусивність.
2. **Гамма-корекція.** Око сприймає яскравість нелінійно. Перед
   відправкою: `out = round(255 * (v/255) ** 2.2)`. Без цього нижня
   половина діапазону виглядає однаково яскравою.

## Синхронізація

Оскільки мікс і світлова доріжка **відрендерені заздалегідь**,
синхронізація зводиться до одного лічильника:

```
Wave.Api тримає ShowClock:
    position_sec, is_playing, started_at_utc

Браузер грає <audio> і шле currentTime у SignalR кожні 100 мс.
Api коригує ShowClock (плавно, не стрибком — інакше світло смикається).
LightPlayer у .NET на таймері 60 FPS:
    idx = (int)((clock.Position + latency_offset) * 60)
    driver.SendAsync(timeline[idx])
```

Не роби таймер через `Task.Delay` — джитер на Windows до 15 мс.
Потрібен `PeriodicTimer` з `TimeSpan.FromMilliseconds(16.67)`, а краще
власний цикл із `Stopwatch` і `SpinWait` на останні 2 мс.

## API

```
GET  /api/lighting/drivers
     → [{name, type, connected, pixel_count}]

POST /api/lighting/drivers
     {type: "ddp", host: "192.168.1.50", pixel_count: 300}

GET  /api/lighting/fixtures
POST /api/lighting/fixtures        розкладка (json вище)

GET  /api/lighting/presets
POST /api/lighting/presets

POST /api/lighting/test
     {color: "#FF0000", fixture_id: "strip_left"}
     статичний колір — перевірити, що залізо взагалі відповідає

POST /api/shows/{setId}/render     {preset_id}   → 202 {job_id}
POST /api/shows/{setId}/start      {driver_ids}
POST /api/shows/{setId}/pause
POST /api/shows/{setId}/seek       {position_sec}
POST /api/shows/{setId}/stop       → завжди робить blackout

hub  /hub/transport                позиція, стан
hub  /hub/lights                   кадри для віртуального емулятора
```

## Формат світлової доріжки

`data/mixes/<set_id>.lights` — бінарний, не JSON:

```
header:  magic "WVLT" | version u16 | fps u16 | pixels u32 | frames u32
body:    frames × pixels × 3 байти (RGB, uint8)
```

Година шоу на 300 пікселів / 60 FPS = 216000 кадрів × 900 байт =
**194 МБ**. Багато, але це послідовне читання з mmap — жодних проблем.
Якщо заважатиме — стисни zstd (RGB-кадри стискаються втричі-вчетверо)
або пиши 30 FPS: різниці на око майже немає.

## Живий режим (пізніше)

Коли захочеться реагувати на живий звук (мікрофон, лінійний вхід):

- FFT на буфері 1024 семпли, 43 кадри/сек — цього досить для
  band_energy у реальному часі.
- Beat detection у реальному часі — окрема задача, значно важча
  за офлайновий beat_track. Найпростіше: адаптивний поріг на
  спектральному потоці в sub-смузі.
- Секції й тональність у реальному часі — не роби, воно того не варте.

Цей режим — окремий `ILightSource`, паралельний до
`PrerenderedTimelineSource`. Cue engine і драйвери не змінюються.

## Безпека і дрібниці

- **Blackout на будь-якому виході.** Стоп, помилка, розрив з'єднання,
  зупинка процесу — стрічка має згаснути. Інакше вона лишиться на
  останньому кадрі і буде світити всю ніч.
- Обмеж загальну потужність: сума всіх каналів × струм на канал.
  При 300 пікселях на білому — 18 А. Або став обмеження яскравості
  у WLED (там є "Maximum Current"), або не роби повний білий.
- Стробоскопічні ефекти в діапазоні 5–30 Гц можуть спровокувати
  фотосенситивний напад. Постав у пресетах жорсткий ліміт частоти
  спалахів (не частіше 3 Гц) і винеси зняття ліміту в явну
  налаштовувану опцію з попередженням.
