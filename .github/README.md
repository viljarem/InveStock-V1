# InveStock - GitHub Actions Konfigurasjon

Denne mappen inneholder GitHub Actions workflows for automatisering av InveStock-appen.

## 📈 Automatisk Markedsdata Oppdatering

**Fil:** `update-market-data.yml`

### Kjøringsplan
- **Daglig kl 17:00 norsk tid** (mandag-fredag)
- Automatisk tilpasset for sommer/vintertid
- Kan også kjøres manuelt via GitHub interface

### Hva gjør den?
1. 📥 Henter latest kode fra repository
2. 🐍 Setter opp Python 3.11 miljø
3. 📦 Installerer alle dependencies fra requirements.txt
4. 📈 Kjører `update_market_data.py` for å hente ny data:
   - Aksjekurser fra Oslo Børs
   - Brent oljepris
   - USD/NOK valutakurs  
   - Fundamental data for utvalgte aksjer
   - Insider trading data
5. 💾 Committer og pusher oppdaterte datafiler tilbake til repo

### Overvåking
- Full logging av alle operasjoner
- Automatisk håndtering av feil
- Fortsetter selv om noen datakilder feiler
- Committer bare hvis det faktisk er nye data

### Manuell kjøring
Du kan også starte oppdateringen manuelt:
1. Gå til "Actions" fanen i GitHub
2. Velg "Automatisk Markedsdata Oppdatering"  
3. Klikk "Run workflow"
4. Valgfritt: Aktiver "force_all" for å ignorere cache

---

**📅 Opprettet:** Februar 2026  
**🔧 Vedlikehold:** Automatisk via GitHub Actions