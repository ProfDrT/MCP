# Academic Search MCP Server

MCP-Server, der akademische Such-APIs als Tools für Claude Desktop / Cowork bereitstellt.
Funktioniert lokal (stdio) oder als Cloud-Service (Render, Railway, etc.).

## Was der Server kann

| Tool | Beschreibung |
|------|-------------|
| `academic_search_all` | Suche über alle 4 Datenbanken gleichzeitig (dedupliziert) |
| `academic_search_semantic_scholar` | Semantic Scholar (200M+ Papers, Citation Graphs, TLDRs) |
| `academic_search_openalex` | OpenAlex (474M+ Works, breiteste Abdeckung, gut für deutsche Papers) |
| `academic_search_crossref` | CrossRef (150M+ Works, beste DOI-Auflösung) |
| `academic_search_arxiv` | arXiv (2.4M+ Preprints, CS/AI/ML) |
| `academic_snowball` | Vorwärts-/Rückwärts-Snowballing über Semantic Scholar |
| `academic_multi_query` | Mehrere Queries auf einmal + Deduplizierung (ideal für SLR) |
| `academic_export` | Export als BibTeX oder CSV |

## Setup (3 Schritte)

### 1. Python-Umgebung vorbereiten

```bash
cd academic-search-mcp
python3 -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 2. Testen ob es läuft

```bash
python server.py --help
```

Sollte ohne Fehler starten und auf stdin warten. Mit `Ctrl+C` abbrechen.

### 3. In Claude Desktop konfigurieren

Öffne die Claude Desktop Config-Datei:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Füge den Server unter `mcpServers` hinzu:

```json
{
  "mcpServers": {
    "academic-search": {
      "command": "/PFAD/ZU/academic-search-mcp/.venv/bin/python",
      "args": ["/PFAD/ZU/academic-search-mcp/server.py"],
      "env": {
        "OPENALEX_MAILTO": "deine.email@example.com",
        "CROSSREF_MAILTO": "deine.email@example.com"
      }
    }
  }
}
```

**Wichtig:** Ersetze `/PFAD/ZU/` mit dem tatsächlichen Pfad auf deinem Rechner.

### Optional: Semantic Scholar API Key

Für höhere Rate-Limits (100 req/5min → 10.000 req/min):

1. API Key beantragen: https://www.semanticscholar.org/product/api
2. In der Config hinzufügen:

```json
"env": {
  "SEMANTIC_SCHOLAR_API_KEY": "dein-key-hier",
  "OPENALEX_MAILTO": "deine.email@example.com",
  "CROSSREF_MAILTO": "deine.email@example.com"
}
```

## Nutzung in Claude Desktop / Cowork

Nach dem Neustart von Claude Desktop stehen die Tools automatisch zur Verfügung:

> "Suche mir Papers zu generative AI enterprise implementation ab 2022"

Claude wird dann `academic_search_all` aufrufen, die Ergebnisse deduplizieren, und dir eine sortierte Liste mit Citation Counts zurückgeben.

Für eine SLR-Suchstrategie:

> "Führe eine Multi-Query-Suche durch mit diesen Queries: ..."

Claude nutzt dann `academic_multi_query` um mehrere Queries parallel zu suchen und die Ergebnisse zu kombinieren.

---

## Cloud-Deployment (Render)

### 1. GitHub-Repo erstellen

```bash
cd academic-search-mcp
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/DEIN-USER/academic-search-mcp.git
git push -u origin main
```

### 2. Auf Render deployen

1. Gehe zu [render.com](https://render.com) → **New** → **Blueprint**
2. Verbinde dein GitHub-Repo
3. Render erkennt die `render.yaml` automatisch und erstellt den Service
4. Setze die Environment Variables im Dashboard:
   - `OPENALEX_MAILTO` = deine Email
   - `CROSSREF_MAILTO` = deine Email
   - `SEMANTIC_SCHOLAR_API_KEY` = (optional, für höhere Rate Limits)

**Alternativ manuell:**
1. **New** → **Web Service** → GitHub-Repo verbinden
2. **Runtime**: Docker
3. **Instance Type**: Free
4. **Deploy**

Deine URL sieht dann so aus: `https://academic-search-mcp.onrender.com`

### 3. Claude Desktop mit Remote-Server verbinden

```json
{
  "mcpServers": {
    "academic-search": {
      "url": "https://academic-search-mcp.onrender.com/mcp"
    }
  }
}
```

Das war's — kein lokaler Python-Install nötig, keine `.venv`, kein `command`/`args`. Nur die URL.

### Health Check

```bash
curl https://academic-search-mcp.onrender.com/health
# → {"status": "ok", "tools": 8}
```

### Hinweis zum Free Tier

Render's Free Tier fährt den Service nach 15 Minuten Inaktivität herunter. Der erste Request nach einer Pause dauert dann ~30 Sekunden (Cold Start). Für regelmäßige Nutzung lohnt sich der Starter Plan ($7/Monat) — der bleibt always-on.

---

## Projektstruktur

```
academic-search-mcp/
├── server.py              ← MCP Server (FastMCP, stdio + HTTP)
├── academic_search.py     ← Such-Funktionen (4 APIs)
├── requirements.txt       ← Python-Dependencies
├── pyproject.toml         ← Projekt-Metadaten
├── Dockerfile             ← Container für Cloud-Deployment
├── render.yaml            ← Render Blueprint (auto-deploy)
└── README.md              ← Diese Datei
```

## API Rate Limits

| API | Ohne Key | Mit Key |
|-----|----------|---------|
| Semantic Scholar | 100 req / 5 min | 10.000 req / min |
| OpenAlex | 100.000 req / Tag | – (kein Key nötig) |
| CrossRef | Unbegrenzt (mit mailto) | – |
| arXiv | 1 req / 3 sec | – |
