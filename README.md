# RAG-chatbot

# Agentic RAG Prototípus – LangGraph Implementáció

Ez a notebook egy **retrieval-augmented generation (RAG)** prototípust valósít meg **agentikus workflow**-val, amely a [LangGraph](https://github.com/langchain-ai/langgraph) könyvtárra épül.  
A rendszer képes összetett kérdéseket részekre bontani, releváns szövegrészeket keresni egy Wikipedia-alapú tudásbázisból, majd generatív nyelvi modell segítségével választ adni.

---

## 1. Architektúra áttekintés

### 1.1 Fő komponensek

- **Adatbetöltés és előfeldolgozás**
  - `datasets` csomaggal Wikipedia adathalmaz letöltése
  - Dokumentumok `.txt` formátumban mentése (`Datas/` mappába)
  - Egyszerű demó-adatbázis építése a mentett fájlokból

- **Vektortár (VectorStore)**
  - Embedding: `SentenceTransformer` (`all-MiniLM-L6-v2` modell)
  - Indexelés: `FAISS` (L2 normalizált vektorok, belső szorzat alapú keresés)
  - Dokumentumdarabolás helyett teljes fájlszöveg indexelése

- **Generatív modell (GenerationModel)**
  - Alap: `GPT-2` (HuggingFace `pipeline` API)
  - Paraméterezett generálás (`temperature`, `top_p`, `repetition_penalty`, n-gram ismétlés tiltása)
  - Kimenet megtisztítása a prompttól

- **AgentState (munkafolyamat állapot)**
  - Kérdés, rész-kérdések, kontextus, részválaszok, végső válasz

- **LangGraph workflow**
  - **Csomópontok:**
    1. `decompose` – összetett kérdések bontása részfeladatokra
    2. `retrieve` – releváns dokumentumok keresése FAISS-ból
    3. `generate` – válasz generálása részfeladatokra
    4. `aggregate` – részválaszok egyesítése végső válasszá
  - **Élek:** lineáris sorrend `decompose` → `retrieve` → `generate` → `aggregate` → `END`

---

## 2. Működési logika

1. **Adat előkészítés**
   - `save_wiki_to_txt()` letölti a Wikipedia Simple English dumpot és lementi a kiválasztott számú dokumentumot
   - `create_demo_data()` betölti a `Datas/` mappában található `.txt` fájlokat és dokumentumlistát hoz létre

2. **Indexelés**
   - A dokumentumok embedding vektorokká alakítása SentenceTransformerrel
   - L2 normalizált vektorok FAISS indexbe töltése

3. **Kérdés feldolgozás**
   - `decompose_node()` egyszerű szabályokkal bontja a kérdést (vessző, kérdőjel, hossz alapján)

4. **Keresés**
   - `retrieve_node()` minden rész-kérdésre lekérdezi a 3 legrelevánsabb dokumentumot
   - Duplikátumokat eltávolítja

5. **Válaszgenerálás**
   - `generate_node()` a kontextus és a rész-kérdés alapján GPT-2-t futtat
   - Rövidített dokumentumrészleteket használ a promptban

6. **Aggregálás**
   - `aggregate_node()` egyesíti a részválaszokat egy végső szövegbe

7. **Eredmény kiírása**
   - Konzolos logok minden lépéshez (bontás, keresés, generálás, aggregálás)
   - Debug információk a végén (használt dokumentumok száma, részfeladatok száma stb.)

---

## 3. Tervezési döntések

- **Egyszerű szabályalapú feladatbontás**: gyors implementáció, nem igényel extra modellt, de korlátozott rugalmasság
- **FAISS + SentenceTransformer**: hatékony keresés kis-közepes méretű adathalmazokon
- **Kis méretű generatív modell (GPT-2)**: demó célokra gyors és egyszerű, de korlátozott minőség
- **LangGraph workflow**: moduláris felépítés, könnyen bővíthető további csomópontokkal
- **Fájl alapú tárolás**: egyszerűség a prototípus fázisban, nem skálázható nagy adatmennyiségre

---

## 4. Jelenlegi bottleneckek

1. **Embedding és indexelés sebessége**
   - SentenceTransformer CPU-n lassú lehet nagy adatmennyiségnél
   - FAISS index teljes újraépítése minden betöltésnél

2. **Keresési relevancia**
   - Egyszerű embedding keresés, nincs finomhangolt keresési pipeline
   - Dokumentumok nincsenek chunkokra bontva

3. **Generálás minősége**
   - GPT-2 kimenet gyakran irreleváns vagy repetitív
   - Kontextus túl hosszú lehet, prompt kinyirhatja a tokenlimitnél

4. **Feladatbontás primitívsége**
   - Nem kezeli a bonyolult összetett kérdéseket

---

## 5. Teljesítménymérés és tesztelés

**Mérési javaslatok:**
- **Latencia mérés**:
  - Indexelés ideje
  - Keresési idő dokumentumonként
  - Generálás ideje rész-kérdésenként
- **Memóriahasználat**:
  - FAISS index mérete
  - Modell betöltés utáni GPU/CPU memória
- **Válaszminőség**:
  - Kézi értékelés relevancia/koherencia szempontból
  - Automatikus metrikák: ROUGE, BLEU, BERTScore (ha van ground truth)

---

## 6. Továbbfejlesztési javaslatok

- **Chunkolás**: Dokumentumokat kisebb szövegrészekre osztani a relevancia növelésére
- **Erősebb generatív modell**: pl. `gpt-neo`, `flan-t5`, `llama` helyi futtatással
- **Fejlettebb feladatbontás**: LLM-alapú dekompozíció
- **Index perzisztencia**: FAISS index mentése és betöltése újraépítés helyett
- **Kontekstukiválasztás finomhangolása**: relevancia alapján súlyozás, redundancia csökkentés
- **Interaktív UI**: Gradio vagy Streamlit alapú felület a felhasználói teszteléshez

---
