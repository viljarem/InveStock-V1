# InveStock V1

En moderne aksjeporteføljeapplikasjon bygget med React, TypeScript og Vite.

## 📋 Innholdsfortegnelse

- [Oversikt](#oversikt)
- [Funksjoner](#funksjoner)
- [Teknologier](#teknologier)
- [Installasjon](#installasjon)
- [Bruk](#bruk)
- [Prosjektstruktur](#prosjektstruktur)
- [Utvikling](#utvikling)
- [Testing](#testing)
- [Bygging](#bygging)
- [Lisens](#lisens)

## 🎯 Oversikt

InveStock V1 er en brukervennlig applikasjon for å administrere og følge med på aksjeporteføljer. Applikasjonen gir sanntidsdata, porteføljeanalyse og detaljert oversikt over investeringer.

## ✨ Funksjoner

- 📊 Sanntids aksjekurser
- 💼 Porteføljeadministrasjon
- 📈 Historiske data og grafer
- 🔍 Søk og filtrer aksjer
- 📱 Responsivt design
- 🌙 Mørk modus
- 💾 Lokal datalagring

## 🛠 Teknologier

- **Frontend Framework**: React 18
- **Språk**: TypeScript
- **Build Tool**: Vite
- **Styling**: CSS Modules / Styled Components
- **State Management**: React Context / Hooks
- **Data Fetching**: Fetch API / Axios
- **Charts**: Recharts / Chart.js

## 📦 Installasjon

### Forutsetninger

- Node.js (versjon 16 eller høyere)
- npm eller yarn

### Steg-for-steg

1. Klon repositoriet:
```bash
git clone <repository-url>
cd "InveStock V1"
```

2. Installer avhengigheter:
```bash
npm install
# eller
yarn install
```

3. Opprett `.env` fil (valgfritt):
```env
VITE_API_KEY=your_api_key_here
```

## 🚀 Bruk

### Kjøre utviklingsserver

**macOS/Linux:**
```bash
./run.command
```

**Eller manuelt:**
```bash
npm run dev
```

Applikasjonen vil være tilgjengelig på `http://localhost:5173`

### Første gang du kjører appen:

1. Sørg for at Node.js er installert: `brew install node`
2. Gjør run.command kjørbar: `chmod +x run.command`
3. Kjør appen: `./run.command`

### Bruke kommandofiler

**macOS/Linux:**
```bash
./run.command
```

**Windows:**
```bash
run.bat
```

Eller dobbeltklikk på filen i Finder/Explorer.

## 📁 Prosjektstruktur

```
InveStock V1/
├── src/
│   ├── components/     # React komponenter
│   ├── pages/          # Sidekomponenter
│   ├── hooks/          # Custom hooks
│   ├── utils/          # Hjelpefunksjoner
│   ├── types/          # TypeScript typer
│   ├── styles/         # Globale stiler
│   ├── App.tsx         # Hovedkomponent
│   └── main.tsx        # Entry point
├── public/             # Statiske filer
├── index.html          # HTML template
├── package.json        # Prosjektavhengigheter
├── tsconfig.json       # TypeScript konfigurasjon
├── vite.config.ts      # Vite konfigurasjon
├── run.command         # macOS kjørefil
└── run.bat             # Windows kjørefil
```

## 💻 Utvikling

### Tilgjengelige Scripts

- `npm run dev` - Start utviklingsserver
- `npm run build` - Bygg for produksjon
- `npm run preview` - Forhåndsvis produksjonsbygg
- `npm run lint` - Kjør linter
- `npm run type-check` - Sjekk TypeScript typer

### Kodestil

Prosjektet følger TypeScript og React beste praksis:
- Funksjonelle komponenter med hooks
- TypeScript for type safety
- CSS Modules for styling isolation
- ESLint for code quality

## 🧪 Testing

```bash
npm run test
# eller
yarn test
```

## 🏗 Bygging

For å bygge applikasjonen for produksjon:

```bash
npm run build
```

Bygget vil bli plassert i `dist/` mappen.

## 📝 Lisens

Dette prosjektet er lisensiert under MIT License.

## 🤝 Bidrag

Bidrag er velkomne! Vennligst opprett en issue eller pull request.

## 📧 Kontakt

For spørsmål eller support, vennligst opprett en issue i repositoriet.

---

Laget med ❤️ av InveStock teamet
