# Friday AI - Screenplay Data Structure Documentation

## Overview

This document defines how screenplays are structured and stored in the Friday AI database for:
1. Semantic search on scene content
2. Export to proper screenplay format (PDF with gray box headers)
3. MCP tool integration
4. Email delivery with proper formatting

---

## Screenplay Formats Supported

### 1. Fountain Format (.fountain)
Standard screenplay format used by professional tools like Highland, Fade In.

**Example from GUSAGUSALU:**
```
EXT. HOUSE-TWO STORIED BUILDING- THIRD FLOOR FRONT BALCONY - MORNING - 8 A.M

A quiet morning. Two pairs of sparrows fly across the sun-lit sky.

KEERTHI
(leaning on the balcony railing)
manchiga undi ga veluturu
(the light feels nice, doesn't it)
```

**Key Elements:**
- Scene headings: `INT./EXT. LOCATION - SUB_LOCATION - TIME`
- Action blocks: Description paragraphs
- Character names: ALL CAPS centered
- Parentheticals: `(direction)` below character name
- Dialogue: Indented below character
- Translations: In parentheses `(english translation)`

### 2. Markdown Draft Format (.md)
Work-in-progress format with mixed Telugu-English content.

**Example from aa_janta_naduma_draft:**
```markdown
## Scene 1: Meet-Cute
**INT. COFFEE SHOP - DAY**

Hero walks in, distracted by phone...

**నిహా** (internal monologue):
> ఈ రోజు మంచి రోజు కాదు...
```

**Key Elements:**
- Scene headers: `## Scene X: Title` or `**INT./EXT.**`
- Character names: Bold `**NAME**` or Telugu script `**నిహా**`
- Internal monologue: Blockquotes
- Mixed Telugu-English dialogue
- Emoji usage for mood

### 3. PDF Celtx Format (.pdf)
Exported from Celtx with specific formatting quirks.

**Quirks to Handle:**
- Spaced-out text: `E X T . H O U S E` → `EXT. HOUSE`
- Gray box scene headings
- Courier Prime font
- Page breaks mid-scene

---

## Database Schema Mapping

### screenplay_projects Table
| Field | Source (Fountain) | Source (Markdown) | Notes |
|-------|------------------|-------------------|-------|
| title | Filename | `# Title` header | Required |
| slug | kebab-case of title | kebab-case of title | Unique, URL-safe |
| logline | Title page if present | First paragraph | Optional |
| primary_language | Detect from content | `te` default | `te`, `en`, `mixed` |
| secondary_language | `en` if translations | `en` | For bilingual scripts |
| status | `draft` default | `draft` | draft/revision/locked |

### screenplay_characters Table
| Field | Source (Fountain) | Source (Markdown) | Notes |
|-------|------------------|-------------------|-------|
| name | Character cue (CAPS) | Bold text or Telugu | Display name |
| full_name | Optional, from notes | Optional | Full character name |
| description | Notes section | Character intro text | Physical/personality |
| role_type | Infer from frequency | Explicit if marked | protagonist/supporting |

**Character Name Normalization:**
- KEERTHI → `KEERTHI`
- **నిహా** → `నిహా` (preserve Telugu)
- Hero → `HERO` (capitalize)
- KEERTHI'S MOTHER → `KEERTHI'S MOTHER` (preserve possessive)

### screenplay_scenes Table
| Field | Source (Fountain) | Source (Markdown) | Notes |
|-------|------------------|-------------------|-------|
| scene_number | Auto-increment | `## Scene X` number | 1, 2, 3... |
| int_ext | `INT.`/`EXT.` prefix | `**INT.**`/`**EXT.**` | INT, EXT, INT/EXT |
| location | After INT./EXT. | After INT./EXT. | Main location |
| sub_location | After `-` | After `-` | Optional sublocation |
| time_of_day | Final element | Final element | MORNING, NIGHT, etc. |
| title | Not in Fountain | `## Scene X: Title` | Optional internal ref |
| tags | Infer from content | Hashtags if present | For search |

**Scene Heading Parsing:**
```
Input: "EXT. HOUSE-TWO STORIED BUILDING- THIRD FLOOR FRONT BALCONY - MORNING - 8 A.M"

Parsed:
- int_ext: "EXT"
- location: "HOUSE-TWO STORIED BUILDING"
- sub_location: "THIRD FLOOR FRONT BALCONY"
- time_of_day: "MORNING - 8 A.M"
```

### scene_elements Table
Elements appear in order within a scene.

| element_type | Content Structure |
|--------------|-------------------|
| action | `{"text": "Description paragraph..."}` |
| dialogue | `{"character": "NAME", "parenthetical": "(V.O.)", "lines": [...]}` |
| transition | `{"text": "CUT TO:"}` |
| shot | `{"text": "CLOSE UP - Hand reaching for glass"}` |

### dialogue_lines Table
For bilingual dialogue with translations.

| Field | Description |
|-------|-------------|
| character_name | Speaker name (normalized) |
| parenthetical | Direction: `(V.O.)`, `(O.S.)`, `(whispering)` |
| text | Original dialogue (Telugu or English) |
| translation | English translation if original is Telugu |
| language | `te`, `en`, or `mixed` |
| line_order | Order within multi-line dialogue block |

---

## Script-Specific Configurations

### GUSAGUSALU (Fountain)
```yaml
project:
  title: "GUSAGUSALU"
  slug: "gusagusalu"
  primary_language: "te"
  secondary_language: "en"
  status: "draft"

parsing:
  format: "fountain"
  translation_pattern: "(english text in parentheses after telugu)"
  scene_count: 27
  main_characters:
    - KEERTHI (protagonist, 13yo girl)
    - KEERTHI'S MOTHER
    - KEERTHI'S DAD
    - KEERTHI'S SISTER
    - PINNY (friend)
```

### Aa Janta Naduma (Markdown)
```yaml
project:
  title: "Aa Janta Naduma"
  slug: "aa-janta-naduma"
  primary_language: "mixed"
  secondary_language: null
  status: "draft"

parsing:
  format: "markdown"
  translation_pattern: null  # Already code-switched
  scene_count: TBD (incomplete draft)
  main_characters:
    - HERO / అర్జున్ (protagonist)
    - HEROINE / నిహా / స్వర్ణ (protagonist)
    - గాయత్రి (supporting)
  notes:
    - Contains placeholder scenes
    - Heavy emoji usage
    - Some scenes are notes/outlines only
```

---

## Export Configuration

### Celtx-Style PDF Export
```python
export_config = {
    "name": "celtx_default",
    "font_family": "Courier Prime",
    "font_size": 12,

    # Page layout (inches)
    "page_width": 8.5,
    "page_height": 11.0,
    "margin_top": 1.0,
    "margin_bottom": 1.0,
    "margin_left": 1.5,  # Binding margin
    "margin_right": 1.0,

    # Scene heading style
    "scene_heading_bg_color": "#CCCCCC",  # Gray box
    "scene_heading_bold": True,

    # Dialogue formatting
    "character_name_caps": True,
    "parenthetical_italics": False,

    # Bilingual support
    "show_translations": True,
    "translation_in_parentheses": True,
}
```

---

## MCP Tool Integration

### scene_search
```json
{
  "query": "Keerthi emotional scene",
  "project_slug": "gusagusalu",
  "top_k": 5
}
```
Returns scenes matching semantic query with:
- Scene heading
- Summary/first action block
- Relevance score

### scene_get
```json
{
  "scene_code": "1",
  "project_slug": "gusagusalu"
}
```
Returns full scene with all elements in order.

### scene_update
```json
{
  "scene_code": "5",
  "project_slug": "gusagusalu",
  "updates": {
    "status": "revision",
    "tags": ["emotional", "turning_point"]
  }
}
```

---

## Processing Pipeline

```
1. Input Detection
   ├── .fountain → Fountain Parser
   ├── .md → Markdown Parser
   └── .pdf → PDF Parser (with spaced-text normalization)

2. Project Creation
   └── Create screenplay_project record

3. Character Extraction
   └── Identify all unique character names
   └── Create screenplay_characters records

4. Scene Parsing
   ├── Parse scene headings (INT/EXT, location, time)
   ├── Extract elements in order (action, dialogue, transition)
   └── Create screenplay_scenes + scene_elements records

5. Dialogue Processing
   ├── Parse character cues
   ├── Extract parentheticals
   ├── Separate dialogue text from translations
   └── Create dialogue_lines records

6. Embedding Generation (optional)
   ├── Generate embeddings for scene summaries
   └── Store in scene_embeddings for semantic search

7. Export Validation
   └── Render to PDF/Fountain to verify formatting
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-10 | Initial structure definition |
